"""Pack whole utterances with their vowel spans, for fine-tuning without windows.

The first fine-tuning pack cut a 0.4 s window around every vowel. That was
chosen so classes could be balanced by resampling -- take all 4,504 ``ẹn``, cap
``i`` at 40,000 -- which a whole-utterance pack cannot do, since you take the
utterance and get whatever vowels are in it.

The cost is a train/inference mismatch. wav2vec2 attends across its whole input,
so a vowel inside a 400 ms window is not represented the same way as the same
vowel inside a three-second utterance, and the engine feeds whole utterances.
Measured, that cost less than feared: the windowed fine-tune still improved
whole-utterance features from 0.879 to 0.904 macro F1. But it is a real ceiling,
and closing it is the cleanest remaining gain.

**This pack is also smaller.** Windowing stores overlapping copies of the same
audio once per vowel; eight vowels in an utterance means eight copies of most of
it. 482,541 windows came to 6.2 GB where the same utterances whole are about
5.7 GB.

Balancing moves into the loss instead of the sampling: the utterances come from
``select_extraction_set.py``, which already ranks by rare-vowel content, and the
manifest carries every vowel so a weighted loss can do the rest.

    python scripts/build_utterance_pack.py --out data/utterance_pack
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

RATE = 16000
#: Utterances longer than this are skipped rather than truncated: cutting one
#: mid-word would leave vowel spans pointing past the end of the audio, and the
#: corpus has few of them.
MAX_S = 12.0
MIN_S = 0.3
ARTEFACT = {"en": "e", "on": "o"}
#: Audio is concatenated into shards of roughly this many samples, with the
#: manifest recording each utterance's offset -- variable-length audio does not
#: fit a rectangular array, and one file per utterance would be 60,000 files.
SHARD_SAMPLES = 200 * 60 * RATE  # ~200 minutes


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path,
                        default=PROJECT_ROOT / "output" / "aligned_v3",
                        help="alignments to pack, already chosen by "
                             "select_extraction_set.py")
    parser.add_argument("--audio", type=Path,
                        default=PROJECT_ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "utterance_pack")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    grids = sorted(args.aligned.rglob("*.TextGrid"))[: args.limit or None]
    index = {p.stem: p for p in args.audio.rglob("*.wav")}
    print(f"{len(grids):,} alignments, {len(index):,} audio files")

    args.out.mkdir(parents=True, exist_ok=True)
    utterances = (args.out / "utterances.tsv").open("w", newline="", encoding="utf-8")
    vowels = (args.out / "vowels.tsv").open("w", newline="", encoding="utf-8")
    u_writer = csv.writer(utterances, delimiter="\t")
    v_writer = csv.writer(vowels, delimiter="\t")
    u_writer.writerow(["utterance", "shard", "offset", "samples", "speaker",
                       "source", "vowels"])
    v_writer.writerow(["utterance", "label", "start", "end"])

    buffer: list[np.ndarray] = []
    held = shard = kept = skipped = 0
    counts = Counter()

    def flush() -> None:
        nonlocal buffer, held, shard
        if not buffer:
            return
        path = args.out / f"audio_{shard:03d}.npy"
        np.save(path, np.concatenate(buffer))
        print(f"  {path.name}: {held / RATE / 60:.0f} min", flush=True)
        buffer, held, shard = [], 0, shard + 1

    for number, grid in enumerate(grids, start=1):
        wav = index.get(grid.stem)
        if wav is None:
            skipped += 1
            continue
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            skipped += 1
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != RATE:
            count = int(len(samples) / rate * RATE)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")
        duration = len(samples) / RATE
        if not MIN_S <= duration <= MAX_S:
            skipped += 1
            continue

        spans = [(ARTEFACT.get(p.label.split("_")[0], p.label.split("_")[0]),
                  int(p.start_s * RATE), int(p.end_s * RATE))
                 for p in alignment.phones if is_vowel_phone(p.label)]
        spans = [(label, a, min(b, len(samples)))
                 for label, a, b in spans if a < len(samples) and b > a]
        if not spans:
            skipped += 1
            continue

        speaker = grid.parent.name if grid.parent.name != args.aligned.name \
            else wav.parent.name
        u_writer.writerow([grid.stem, shard, held, len(samples), speaker,
                           source_of(speaker), len(spans)])
        for label, a, b in spans:
            v_writer.writerow([grid.stem, label, a, b])
            counts[label] += 1

        buffer.append(np.clip(samples * 32767, -32768, 32767).astype(np.int16))
        held += len(samples)
        kept += 1
        if held >= SHARD_SAMPLES:
            flush()
        if number % 10000 == 0:
            print(f"  {number:,} alignments, {kept:,} kept", flush=True)

    flush()
    utterances.close()
    vowels.close()

    size = sum(p.stat().st_size for p in args.out.glob("*.npy")) / 1e9
    print(f"\n{kept:,} utterances, {sum(counts.values()):,} vowels, {size:.1f} GB")
    print(f"{skipped:,} skipped (missing audio, out of duration range, or no vowels)")
    print("class balance:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
