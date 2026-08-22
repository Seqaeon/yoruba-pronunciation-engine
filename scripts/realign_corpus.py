"""Re-align the whole training corpus, to supervise a neural forced aligner.

The corpus TextGrids from acoustic-model training were not kept, and the only
alignment data that survived is data/utterance_pack/vowels.tsv -- 905,870 vowel
spans, no consonants, no silence, no word boundaries. Vowel spans are enough for
tone and vowel identity; they are not enough to replace MFA, which also supplies
word boundaries for the omission check and per-phone scores for GOP.

So this regenerates the full alignments. It is a one-off batch job: MFA's 13.8 s
per utterance is almost entirely per-invocation setup, and aligning a whole
corpus in one call amortises it away.

The dictionary is built to cover the corpus exactly. models/train.dict covers
97.4% of word forms, and MFA drops an utterance containing any out-of-vocabulary
word without saying so -- a failure this project has already been bitten by.
"""

from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.alignment.mfa import MFAAligner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "aligned_corpus")
    parser.add_argument("--dictionary", type=Path,
                        default=ROOT / "data" / "dictionary" / "corpus_full.dict")
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    config = EngineConfig()
    config.mfa.num_jobs = args.jobs
    aligner = MFAAligner(config)

    if not args.dictionary.exists():
        words = set()
        for lab in args.corpus.rglob("*.lab"):
            for word in lab.read_text(encoding="utf-8").split():
                words.add(unicodedata.normalize("NFC", word))
        print(f"generating pronunciations for {len(words)} word forms ...", flush=True)
        aligner.request_dictionary(" ".join(sorted(words)), args.dictionary)
    entries = sum(1 for line in args.dictionary.read_text(encoding="utf-8").splitlines()
                  if line.strip())
    print(f"dictionary: {entries} entries at {args.dictionary}", flush=True)

    utterances = sum(1 for _ in args.corpus.rglob("*.lab"))
    print(f"aligning {utterances} utterances with {args.jobs} jobs ...", flush=True)
    start = time.perf_counter()
    aligner.align_corpus(args.corpus, args.out, dictionary=args.dictionary)
    elapsed = time.perf_counter() - start

    produced = sum(1 for _ in args.out.rglob("*.TextGrid"))
    print(f"\n{produced} TextGrids in {elapsed/60:.1f} min "
          f"({elapsed/max(produced,1):.3f} s per utterance)")
    print(f"{utterances - produced} utterances produced nothing "
          f"({(utterances-produced)/utterances:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
