"""Pack a speaker-balanced slice of the training corpus for ASR fine-tuning.

The engine's recogniser is ``LyngualLabs/whisper-small-yoruba`` -- someone
else's fine-tune of whisper-small on someone else's Yoruba. Measured on 60
native-speaker studio readings of 3-4 words, it gets 82% of words right and 48%
of phrases entirely right, and a vowel-level fine-tune of its encoder changed
neither figure: 1 of 60 transcripts differed, because that fine-tune optimised
vowel identity rather than transcription.

What has never been tried is fine-tuning it for transcription on *this*
project's corpus: 193,309 utterances from 2,801 speakers, about 118 hours,
already transcript-matched and normalised for acoustic-model training.

This packs a subset of that into something a Kaggle notebook can mount:
16 kHz mono FLAC plus one manifest. FLAC because the whole 118 hours is roughly
8 GB packed and 13 GB as wav, and because it is lossless -- resampling artefacts
have already cost this project one wrong conclusion.

**The split is written into the manifest, by speaker.** Whisper memorises a
voice readily, so a random utterance split would report a number that says
nothing about a new learner. Deciding it here rather than in the notebook means
the split cannot quietly change between runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import soundfile as sf  # noqa: E402

from yoruba_engine.audio import TARGET_RATE_HZ, resample  # noqa: E402

#: Held-out shares, by speaker. Small because the corpus is large: 5% of 2,801
#: speakers is still 140 voices, which is plenty to measure against and leaves
#: the training set nearly whole.
DEV_SHARE, TEST_SHARE = 0.05, 0.05


def split_of(speaker: str) -> str:
    """Which split a speaker belongs to, decided by a stable hash.

    Hashing rather than shuffling means adding audio for a new speaker cannot
    move an existing one across the boundary, so a later run's numbers stay
    comparable with an earlier one's.
    """
    draw = int(hashlib.sha256(speaker.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if draw < TEST_SHARE:
        return "test"
    if draw < TEST_SHARE + DEV_SHARE:
        return "dev"
    return "train"


def source_of(speaker: str) -> str:
    """Which corpus a speaker came from, from the directory prefix."""
    return speaker.split("__", 1)[0] if "__" in speaker else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus",
                        help="MFA-format corpus of paired .wav and .lab files")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "asr_pack")
    parser.add_argument("--hours", type=float, default=40.0,
                        help="Budget for the training split. Dev and test are "
                             "packed whole, since they are small and a truncated "
                             "test set measures less than it appears to.")
    parser.add_argument("--max-seconds", type=float, default=20.0,
                        help="Skip longer utterances. Whisper pads every input "
                             "to 30 s, so a long one costs the same as a short "
                             "one to train on but carries more label noise.")
    parser.add_argument("--min-seconds", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pairs: list[tuple[str, str, Path, Path]] = []
    for lab in sorted(args.corpus.rglob("*.lab")):
        wav = lab.with_suffix(".wav")
        if wav.exists():
            pairs.append((lab.stem, lab.parent.name, wav, lab))
    if not pairs:
        print(f"No .wav/.lab pairs under {args.corpus}", file=sys.stderr)
        return 1
    print(f"{len(pairs)} utterances from {len({s for _, s, _, _ in pairs})} speakers")

    # Group by speaker so the training budget is spent across voices rather than
    # exhausting one speaker at a time.
    by_speaker: dict[str, list] = defaultdict(list)
    for utt, speaker, wav, lab in pairs:
        by_speaker[speaker].append((utt, wav, lab))

    rng = random.Random(args.seed)
    for utterances in by_speaker.values():
        rng.shuffle(utterances)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    budget = args.hours * 3600.0
    written = defaultdict(float)
    counts = defaultdict(int)
    skipped = 0

    speakers = sorted(by_speaker)
    rng.shuffle(speakers)
    # Round-robin across speakers so a truncated pack is still balanced.
    queues = {s: iter(by_speaker[s]) for s in speakers}

    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance", "speaker", "source", "split", "seconds", "text"])

        exhausted: set[str] = set()
        while len(exhausted) < len(speakers):
            for speaker in speakers:
                if speaker in exhausted:
                    continue
                try:
                    utt, wav, lab = next(queues[speaker])
                except StopIteration:
                    exhausted.add(speaker)
                    continue

                split = split_of(speaker)
                if split == "train" and written["train"] >= budget:
                    exhausted.add(speaker)
                    continue

                text = lab.read_text(encoding="utf-8").strip()
                if not text:
                    skipped += 1
                    continue
                try:
                    samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
                except Exception:
                    skipped += 1
                    continue
                if getattr(samples, "ndim", 1) > 1:
                    samples = samples.mean(axis=1)
                seconds = len(samples) / rate
                if not (args.min_seconds <= seconds <= args.max_seconds):
                    skipped += 1
                    continue
                if rate != TARGET_RATE_HZ:
                    samples = resample(samples, rate, TARGET_RATE_HZ)

                sf.write(str(audio_dir / f"{utt}.flac"), samples, TARGET_RATE_HZ)
                writer.writerow([utt, speaker, source_of(speaker), split,
                                 f"{seconds:.2f}", text])
                written[split] += seconds
                counts[split] += 1

    print(f"\nskipped {skipped} (empty, unreadable, or outside "
          f"{args.min_seconds}-{args.max_seconds}s)")
    for split in ("train", "dev", "test"):
        print(f"  {split:<6}{counts[split]:>7} utterances  "
              f"{written[split] / 3600:>6.1f} h")
    print(f"\nWrote {args.out}")
    print("Upload that directory to Kaggle as a dataset, then run "
          "notebooks/finetune_whisper_asr_kaggle.ipynb against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
