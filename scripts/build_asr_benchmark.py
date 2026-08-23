"""Export the readings every ASR benchmark in this project has used.

Comparing a new recogniser against the shipped one only means something if both
are scored on the same audio with the same metric. This writes that audio out as
a folder small enough to upload anywhere, so a candidate model can be tried on a
rented GPU without cloning the corpus.

Two questions need answering, so two sets are included and deduplicated:

* 60 readings of 3-4 words, the set behind every per-word figure quoted for this
  project -- 82% for the shipped model unprompted, 97% prompted.
* 40 readings that contain a sub-dot, which is the distinction a Yoruba
  recogniser most often loses: 98% for the shipped model, 94% for NCAIR1.

16 kHz mono FLAC, because that is what every model here expects and because the
corpus is 48 kHz, which would triple the upload for no gain.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    import soundfile as sf

    from yoruba_engine.audio import TARGET_RATE_HZ, read_mono
    from yoruba_engine.g2p.orthography import normalize

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "asr_benchmark")
    args = parser.parse_args()

    rows = []
    for line in (args.corpus / "line_index.tsv").read_text(encoding="utf-8").splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        utterance, text = parts[0].strip(), parts[-1].strip()
        wav = args.corpus / "wavs" / f"{utterance}.wav"
        if wav.exists() and "[" not in text:
            rows.append((utterance, text, wav))

    def dots(text: str) -> int:
        return unicodedata.normalize("NFD", normalize(text)).count("̣")

    # Exactly the two samples the quoted figures come from: same filters, same
    # seeds, same order. Reproducing them here rather than picking fresh audio is
    # what makes a new model's number comparable with the old ones.
    words = [r for r in rows if 3 <= len(r[1].split()) <= 4]
    random.Random(7).shuffle(words)
    words = words[:60]

    subdots = [r for r in rows if 3 <= len(r[1].split()) <= 5 and dots(r[1]) > 0]
    random.Random(4).shuffle(subdots)
    subdots = subdots[:40]

    chosen: dict[str, tuple] = {}
    tags: dict[str, set] = {}
    for group, entries in (("words", words), ("subdots", subdots)):
        for utterance, text, wav in entries:
            chosen[utterance] = (text, wav)
            tags.setdefault(utterance, set()).add(group)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with (args.out / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance", "text", "groups", "sub_dots"])
        for utterance, (text, wav) in sorted(chosen.items()):
            samples = read_mono(wav, TARGET_RATE_HZ)
            sf.write(str(audio_dir / f"{utterance}.flac"), samples, TARGET_RATE_HZ)
            total += len(samples) / TARGET_RATE_HZ
            writer.writerow([utterance, text, ",".join(sorted(tags[utterance])),
                             dots(text)])

    size = sum(f.stat().st_size for f in args.out.rglob("*"))
    print(f"{len(chosen)} readings, {total/60:.1f} minutes of audio")
    print(f"  {len(words)} for per-word accuracy, {len(subdots)} containing sub-dots")
    print(f"  {sum(dots(t) for t, _ in chosen.values())} sub-dots in the references")
    print(f"\nWrote {args.out}  ({size/1e6:.1f} MB)")
    print("Upload that folder as a Kaggle dataset and run "
          "notebooks/benchmark_asr_kaggle.ipynb against it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
