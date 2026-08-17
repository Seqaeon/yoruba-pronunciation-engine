"""Stage the studio-domain corpora into an MFA corpus for local alignment.

Why this exists: 94.5% of the vowel classifier's training audio is crowdsourced,
while learners record in studio-ish conditions. Measured by source on held-out
speakers, nasal F1 runs 0.65-0.85 on crowdsourced voices and 0.14-0.39 on studio
ones. More crowdsourced data does not close that; more studio data is the only
thing that lifts the ceiling.

Two rules are enforced here, and both matter more than volume:

**Diacritics are required.** An undiacritized line teaches the classifier that
``ọ`` is ``o``, which is precisely the contrast that fails on real attempts
(77.6% correct on SLR86 held-out, the weakest of any pair measured). A line
without tone marks is worse than no line at all for labelling vowels, however
good its audio.

**Transcripts are deduplicated.** ``thisniyi`` is explicitly ``cv_and_openslr``
and therefore contains SLR86, which is already the local corpus. Merging blind
would double-count utterances and pull the classifier further toward speakers it
already over-represents.

    python scripts/ingest_studio.py --out data/studio_corpus
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402

MIN_DURATION_S, MAX_DURATION_S = 0.30, 15.0
TARGET_RATE = 16000
#: Clips with no speaker label are grouped this many to a pseudo-speaker, so
#: MFA's speaker-adaptive stage has something to estimate a transform from.
BUCKET_SIZE = 50


def has_tone_marks(text: str) -> bool:
    """Whether the line carries acute or grave marks at all."""
    return any(c in "̀́" for c in unicodedata.normalize("NFD", text or ""))


def decode_audio(data: bytes | None, path: Path | None):
    """Read audio from bytes or a file, falling back to ffmpeg for mp3.

    libsndfile handles mp3 only from 1.1 onward, and the IroyinSpeech clips are
    mp3, so the fallback is not optional on older systems.
    """
    try:
        if data is not None:
            return sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        pass
    command = ["ffmpeg", "-v", "quiet", "-i", "pipe:0" if data else str(path),
               "-f", "wav", "-ar", str(TARGET_RATE), "-ac", "1", "pipe:1"]
    result = subprocess.run(command, input=data, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise ValueError("unreadable audio")
    return sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)


class Stager:
    """Writes accepted utterances into an MFA-format corpus."""

    def __init__(self, root: Path, seen_text: set[str]) -> None:
        self.root = root
        self.seen_text = seen_text
        self.kept = Counter()
        self.hours = Counter()
        self.rejected = Counter()
        self.unlabelled = Counter()
        self.speakers: dict[str, set] = {}

    def add(self, source: str, speaker: str | None, utt_id: str, text: str,
            data: bytes | None = None, path: Path | None = None) -> bool:
        text = (text or "").strip()
        if not text:
            self.rejected["empty"] += 1
            return False
        if not has_tone_marks(text):
            self.rejected["no diacritics"] += 1
            return False

        key = " ".join(tokenize(normalize(text), keep_noise=False))
        if not key:
            self.rejected["empty"] += 1
            return False
        if key in self.seen_text:
            self.rejected["duplicate transcript"] += 1
            return False

        try:
            samples, rate = decode_audio(data, path)
        except Exception:
            self.rejected["unreadable"] += 1
            return False

        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        duration = len(samples) / rate if rate else 0.0
        if duration < MIN_DURATION_S:
            self.rejected["too short"] += 1
            return False
        if duration > MAX_DURATION_S:
            self.rejected["too long"] += 1
            return False
        if rate != TARGET_RATE:
            count = int(len(samples) / rate * TARGET_RATE)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")

        if speaker:
            group = f"{source}_{speaker}"
        else:
            group = f"{source}_unk{self.unlabelled[source] // BUCKET_SIZE:04d}"
            self.unlabelled[source] += 1

        directory = self.root / group
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(str(directory / f"{utt_id}.wav"), samples, TARGET_RATE, subtype="PCM_16")
        (directory / f"{utt_id}.lab").write_text(normalize(text), encoding="utf-8")

        self.seen_text.add(key)
        self.kept[source] += 1
        self.hours[source] += duration / 3600
        self.speakers.setdefault(source, set()).add(group)
        return True


def existing_transcripts(config: EngineConfig) -> set[str]:
    """Normalized transcripts already in the local corpus, for deduplication."""
    seen = set()
    index = config.line_index
    if not index.exists():
        return seen
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                seen.add(" ".join(tokenize(normalize(row[2]), keep_noise=False)))
    return seen


def ingest_parquet(stager: Stager, paths: list[Path], source: str) -> None:
    import pyarrow.parquet as pq  # noqa: PLC0415

    for path in paths:
        if not path.exists():
            continue
        parquet = pq.ParquetFile(path)
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=256)):
            data = batch.to_pydict()
            texts = data.get("sentence") or data.get("text") or []
            audios = data.get("audio") or []
            speakers = data.get("speaker_id") or data.get("client_id")
            for row, (text, audio) in enumerate(zip(texts, audios)):
                if not audio or not audio.get("bytes"):
                    continue
                speaker = str(speakers[row]) if speakers else None
                stager.add(source, speaker,
                           f"{source}_{path.stem[-6:]}_{batch_index}_{row}",
                           text, data=audio["bytes"])
        print(f"  {path.name}: running total {stager.kept[source]:,}", flush=True)


def ingest_common_voice(stager: Stager, root: Path, source: str) -> None:
    """Read a Common Voice export: validated.tsv plus clips/."""
    table = root / "validated.tsv"
    if not table.exists():
        print(f"  {table} not found"); return
    with table.open(encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            clip = root / "clips" / (row.get("path") or "")
            if not clip.exists():
                continue
            # client_id is a long hash; the tail is unique enough to name a dir.
            speaker = (row.get("client_id") or "")[-16:] or None
            stager.add(source, speaker, f"{source}_{clip.stem}",
                       row.get("sentence") or "", path=clip)
            if number and number % 500 == 0:
                print(f"  {number} rows, kept {stager.kept[source]:,}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "studio_corpus")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    config = EngineConfig()
    seen = existing_transcripts(config)
    print(f"{len(seen):,} transcripts already in the local corpus (deduped against)\n")

    args.out.mkdir(parents=True, exist_ok=True)
    stager = Stager(args.out, seen)

    print("IroyinSpeech (Common Voice export)")
    ingest_common_voice(stager, args.data / "IroyinSpeech", "iroyin")

    print("\nAt-Tawheed yoruba-tts-ljspeech")
    ingest_parquet(stager, [args.data / "train-00000-of-00001.parquet"], "ljspeech")

    print("\nthisniyi cv_and_openslr")
    ingest_parquet(stager, sorted(args.data.glob("thisniyi_*.parquet")), "thisniyi")

    print(f"\n{'source':<12}{'utterances':>12}{'hours':>9}{'speakers':>10}")
    for source in sorted(stager.kept):
        print(f"{source:<12}{stager.kept[source]:>12,}{stager.hours[source]:>9.2f}"
              f"{len(stager.speakers[source]):>10}")
    print(f"{'TOTAL':<12}{sum(stager.kept.values()):>12,}"
          f"{sum(stager.hours.values()):>9.2f}"
          f"{sum(len(s) for s in stager.speakers.values()):>10}")

    print("\nrejected:")
    for reason, count in stager.rejected.most_common():
        print(f"  {reason:<22}{count:>8,}")
    print(f"\nCorpus written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
