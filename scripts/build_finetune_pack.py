"""Cut a class-balanced pack of vowel audio windows for encoder fine-tuning.

Everything measured so far has trained the *classifier* on top of a frozen
encoder. The encoder -- ``LyngualLabs/mms-300m-yoruba-english``, 300M parameters
-- has never had its weights touched by this project. That matters because
replacing hand-designed features with it moved vowel identity from 0.595 to
0.873, while quadrupling the training data moved it 0.873 to 0.895. The
representation is the ceiling, and fine-tuning it is the one lever not yet
pulled.

Fine-tuning needs raw audio, so the cached ``models/w2v_slices/*.npz`` are no
use: those are the encoder's *output*. This reads the alignments instead and
cuts a window around each vowel.

**Rare classes are exhausted, common ones are subsampled.** Vowels are not
remotely balanced -- ``i`` is 24.2% of the corpus and ``ẹn`` is 0.53% -- so a
uniform sample spends its budget on the classes already solved. Every nasal in
the corpus is taken; the seven common classes are capped. That is the difference
between a pack that carries 10,183 ``ẹn`` examples and one that carries 1,000.

Each clip is a fixed window with the vowel centred, and the manifest records
where inside the window the vowel actually sits, so training can pool over the
vowel exactly as inference does rather than over the padding.

    python scripts/build_finetune_pack.py --out data/finetune_pack
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.audio import resample  # noqa: E402
from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

RATE = 16000
#: Window length. The median vowel runs 90 ms; 150 ms of context on each side
#: gives the encoder something to condition on without the window becoming
#: mostly neighbours.
WINDOW_S = 0.40
WINDOW = int(WINDOW_S * RATE)

#: Taken in full wherever they occur -- these are what the pack exists for.
RARE = ("En", "in", "an", "On", "un")
#: Everything else is capped at this many clips per class.
DEFAULT_CAP = 40_000

ARTEFACT = {"en": "e", "on": "o"}
#: Written incrementally in shards so the pack never has to fit in memory.
SHARD = 50_000


def source_of(speaker: str) -> str:
    if speaker.startswith("slr86__"):
        return "slr86"
    if speaker.startswith("yecs__"):
        return "yecs"
    for name in ("iroyin", "commonvoice", "ljspeech", "thisniyi", "waxal",
                 "michsethowusu"):
        if speaker.startswith(f"studio__{name}"):
            return name
    return "other"


def survey(grids: list[Path]) -> list[tuple]:
    """Every labelled vowel across the alignments, before any sampling."""
    found = []
    for number, grid in enumerate(grids, start=1):
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
        except Exception:
            continue
        for phone in alignment.phones:
            if not is_vowel_phone(phone.label):
                continue
            base = phone.label.split("_")[0]
            label = ARTEFACT.get(base, base)
            found.append((label, grid.parent.name, grid.stem,
                          phone.start_s, phone.end_s))
        if number % 20000 == 0:
            print(f"  surveyed {number:,} alignments, {len(found):,} vowels",
                  flush=True)
    return found


def choose(found: list[tuple], cap: int, seed: int = 0) -> list[tuple]:
    """Keep every rare-class vowel and a capped sample of the common ones."""
    by_label = defaultdict(list)
    for row in found:
        by_label[row[0]].append(row)
    rng = random.Random(seed)
    kept = []
    print(f"\n{'class':<6}{'available':>12}{'kept':>10}")
    for label in sorted(by_label):
        rows = by_label[label]
        if label in RARE or len(rows) <= cap:
            take = rows
        else:
            take = rng.sample(rows, cap)
        kept.extend(take)
        print(f"{label:<6}{len(rows):>12,}{len(take):>10,}"
              + ("   (all: rare)" if label in RARE else ""))
    rng.shuffle(kept)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path,
                        default=PROJECT_ROOT / "output" / "train_aligned")
    parser.add_argument("--audio", type=Path,
                        default=PROJECT_ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "finetune_pack")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help=f"per-class cap for common vowels (default {DEFAULT_CAP:,})")
    parser.add_argument("--limit", type=int, help="only this many alignments, for a trial")
    args = parser.parse_args()

    grids = sorted(args.aligned.rglob("*.TextGrid"))[: args.limit or None]
    print(f"{len(grids):,} alignments")
    found = survey(grids)
    print(f"{len(found):,} labelled vowels")

    kept = choose(found, args.cap)
    print(f"\n{len(kept):,} clips to cut "
          f"({len(kept) * WINDOW * 2 / 1e9:.1f} GB)")

    index = {p.stem: p for p in args.audio.rglob("*.wav")}
    args.out.mkdir(parents=True, exist_ok=True)

    # Group by utterance so each audio file is read once rather than once per
    # vowel; several vowels usually come from the same recording.
    by_utterance = defaultdict(list)
    for row in kept:
        by_utterance[row[2]].append(row)

    manifest = (args.out / "manifest.tsv").open("w", newline="", encoding="utf-8")
    writer = csv.writer(manifest, delimiter="\t")
    writer.writerow(["clip", "shard", "row", "label", "speaker", "source",
                     "vowel_start", "vowel_end", "duration_ms"])

    buffer = np.zeros((SHARD, WINDOW), dtype=np.int16)
    shard = filled = written = skipped = 0
    counts = Counter()

    def flush(count: int) -> None:
        nonlocal shard
        if not count:
            return
        path = args.out / f"clips_{shard:03d}.npy"
        np.save(path, buffer[:count])
        print(f"  wrote {path.name}: {count:,} clips", flush=True)
        shard += 1

    for stem, rows in by_utterance.items():
        wav = index.get(stem)
        if wav is None:
            skipped += len(rows)
            continue
        try:
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            skipped += len(rows)
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != RATE:
            samples = resample(samples, rate, RATE)

        for label, speaker, _, start_s, end_s in rows:
            centre = int((start_s + end_s) / 2 * RATE)
            begin = centre - WINDOW // 2
            clip = np.zeros(WINDOW, dtype="float32")
            lo, hi = max(begin, 0), min(begin + WINDOW, len(samples))
            if hi <= lo:
                skipped += 1
                continue
            clip[lo - begin: hi - begin] = samples[lo:hi]
            buffer[filled] = np.clip(clip * 32767, -32768, 32767).astype(np.int16)

            # Where the vowel sits inside the window, so training pools over the
            # vowel rather than the padding around it.
            vowel_start = max(int(start_s * RATE) - begin, 0)
            vowel_end = min(int(end_s * RATE) - begin, WINDOW)
            writer.writerow([written, shard, filled, label, speaker,
                             source_of(speaker), vowel_start, vowel_end,
                             round((end_s - start_s) * 1000, 1)])
            counts[label] += 1
            filled += 1
            written += 1
            if filled == SHARD:
                flush(filled)
                filled = 0

    flush(filled)
    manifest.close()

    print(f"\n{written:,} clips written to {args.out} "
          f"({sum(p.stat().st_size for p in args.out.glob('*.npy')) / 1e9:.1f} GB)")
    if skipped:
        print(f"{skipped:,} skipped (missing or unreadable audio)")
    print("class balance:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
