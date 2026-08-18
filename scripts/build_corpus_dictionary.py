"""Generate a pronunciation dictionary covering one staged corpus.

MFA drops any utterance containing a word its dictionary does not define, so a
corpus is only as alignable as its dictionary is complete. The shipped
dictionary covers 49% of word types in held-out Yoruba text, which is fine for
single requests -- those generate coverage on the fly -- but useless for
aligning a corpus of tens of thousands of utterances.

This reuses the same generator the single-request path uses, handing it the
corpus vocabulary in place of one transcript, so corpus alignment and live
requests cannot drift apart in how they pronounce a word.

    python scripts/build_corpus_dictionary.py \
        --corpus data/yecs_corpus --out data/dictionary/yecs.dict
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.mfa import MFAAligner  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s | %(message)s")

    vocabulary: set[str] = set()
    labels = 0
    for lab in args.corpus.rglob("*.lab"):
        labels += 1
        vocabulary.update(lab.read_text(encoding="utf-8").split())
    print(f"{labels:,} transcripts, {len(vocabulary):,} distinct written forms")

    aligner = MFAAligner(EngineConfig())
    aligner.request_dictionary(" ".join(sorted(vocabulary)), args.out)
    entries = sum(1 for _ in args.out.open(encoding="utf-8"))
    print(f"{entries:,} pronunciations written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
