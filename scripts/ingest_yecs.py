"""Stage the YECS code-switching corpus as a Yoruba alignment corpus.

YECS is 107 hours over 136 speakers, against the 4 hours of SLR86 the acoustic
model is currently trained on. It is also, for our purposes, the wrong shape:
99.8% of its utterances mix Yoruba and English in one recording, so there is no
subset of Yoruba-only rows to lift out. Separation has to happen inside each
recording.

**English is absorbed rather than removed.** The audio is continuous, so the
English words cannot be cut out before alignment without knowing where they
are -- which is what alignment is for. Instead each English word is replaced in
the transcript by a bracketed token, which the tokenizer collapses to
``[noise]`` and the dictionary maps to MFA's ``spn`` model. The aligner then
places real phone boundaries on the Yoruba words and lets ``spn`` soak up the
English, so every interval this corpus contributes downstream is Yoruba.

**Diacritics were verified, not assumed.** 354,368 YECS Yoruba tokens were
checked against the spellings in the existing corpus: 99.5% agree, and the 0.5%
that lack a tone mark our corpus always writes are mostly proper nouns
(``Yoruba``, ``Ibadan``, ``Bọla``) where marking is optional in practice. That
is a different situation from Yoruba-TTS-Dataset, which had no diacritics at
all and was rejected.

**Audio is symlinked, not copied.** It is already 16 kHz mono PCM_16, which is
exactly the alignment format, and copying would duplicate 14 GB to no purpose.

    python scripts/ingest_yecs.py --limit 2000     # validate on a subset
    python scripts/ingest_yecs.py                  # everything
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402

csv.field_size_limit(10 ** 7)

MIN_DURATION_S, MAX_DURATION_S = 0.30, 15.0

#: Replaces an English word. Any fully bracketed token collapses to the
#: tokenizer's noise marker, which the dictionary already maps to ``spn``.
ENGLISH_TOKEN = "[english]"

#: The test split ships without language tags -- LyngualLabs withholds them for
#: benchmarking -- so English words in it cannot be identified and it is skipped.
SPLITS = ("train", "val")

TONE_MARKS = set("̀́")


def has_tone_marks(text: str) -> bool:
    return any(c in TONE_MARKS for c in unicodedata.normalize("NFD", text))


def build_transcript(tags: list[dict]) -> tuple[str, int, int]:
    """Return (transcript, yoruba words, english words) for one utterance."""
    words, yoruba, english = [], 0, 0
    for tag in tags:
        word = (tag.get("word") or "").strip()
        if not word:
            continue
        if tag.get("language") == "yo":
            words.append(word)
            yoruba += 1
        else:
            words.append(ENGLISH_TOKEN)
            english += 1
    return " ".join(words), yoruba, english


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path,
                        default=PROJECT_ROOT / "data" / "YECS_Corpus")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "yecs_corpus")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many accepted utterances")
    parser.add_argument("--min-yoruba", type=float, default=0.0,
                        metavar="FRACTION",
                        help="require this share of words to be Yoruba. The "
                             "aligner's spn model is greedy, so utterances that "
                             "are mostly English align badly (see module docs).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    kept = Counter()
    rejected = Counter()
    speakers: set[str] = set()
    hours = 0.0
    yoruba_words = english_words = 0
    seen: set[str] = set()

    for split in SPLITS:
        table = args.corpus / "metadata" / f"{split}_metadata.csv"
        if not table.exists():
            print(f"  {table} missing"); continue
        with table.open(encoding="utf-8", newline="") as handle:
            for number, row in enumerate(csv.DictReader(handle)):
                if args.limit and sum(kept.values()) >= args.limit:
                    break
                try:
                    tags = ast.literal_eval(row.get("language_tags") or "[]")
                except (ValueError, SyntaxError):
                    rejected["unparseable tags"] += 1
                    continue

                text, n_yo, n_en = build_transcript(tags)
                if not n_yo:
                    rejected["no Yoruba words"] += 1
                    continue
                if n_yo / max(n_yo + n_en, 1) < args.min_yoruba:
                    rejected["too much English"] += 1
                    continue
                if not has_tone_marks(text):
                    rejected["no diacritics"] += 1
                    continue

                duration = float(row.get("duration") or 0)
                if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
                    rejected["duration out of range"] += 1
                    continue

                clip = args.corpus / "audio" / split / (row.get("file_name") or "")
                if not clip.exists():
                    rejected["audio missing"] += 1
                    continue

                # Prompts are deliberately read by several speakers for acoustic
                # variety, so the transcript alone is not a duplicate key here.
                speaker = row.get("speaker_id") or "unknown"
                key = f"{speaker}\x00{' '.join(tokenize(normalize(text), keep_noise=False))}"
                if key in seen:
                    rejected["duplicate transcript"] += 1
                    continue
                seen.add(key)

                directory = args.out / speaker
                directory.mkdir(parents=True, exist_ok=True)
                stem = Path(row["file_name"]).stem
                link = directory / f"{stem}.wav"
                if not link.exists():
                    link.symlink_to(clip.resolve())
                (directory / f"{stem}.lab").write_text(text, encoding="utf-8")

                kept[split] += 1
                speakers.add(speaker)
                hours += duration / 3600
                yoruba_words += n_yo
                english_words += n_en
                if number and number % 10000 == 0:
                    print(f"  {split}: {number:,} rows, kept {sum(kept.values()):,}",
                          flush=True)

    total = sum(kept.values())
    print(f"\n{'split':<10}{'utterances':>12}")
    for split in SPLITS:
        print(f"{split:<10}{kept[split]:>12,}")
    print(f"{'TOTAL':<10}{total:>12,}   {hours:.1f} h over {len(speakers)} speakers")
    words = yoruba_words + english_words
    print(f"\nwords: {yoruba_words:,} Yoruba ({yoruba_words/max(words,1):.1%}), "
          f"{english_words:,} English absorbed as spn")
    print("\nrejected:")
    for reason, count in rejected.most_common():
        print(f"  {reason:<24}{count:>9,}")
    print(f"\nCorpus written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
