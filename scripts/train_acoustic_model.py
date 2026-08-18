"""Train a Yoruba acoustic model on every staged corpus.

The shipped model was trained on Kaggle over 37.93 hours from 2,047 speakers:
michsethowusu (24.0 h), Waxal (6.30 h), local TTS (4.16 h) and OpenSLR (3.47 h).
Its weakness is not volume but admission policy -- it was built with
``require_tone_marks`` disabled, so it learned from Waxal's 42% undiacritized
lines, which teaches the model that ``ọ`` and ``o`` are the same vowel.

Aligning YECS with it left 62% of the audio unlabelled and put the median vowel
at 30 ms against 80 ms on SLR86, so its coverage does not extend to this speech.

Training does not use the old model. MFA starts flat, initialising monophones
from the audio itself and iterating align/re-estimate through triphone,
LDA+MLLT and speaker-adaptive training, so a bad one-pass alignment from a
mismatched model cannot propagate into the result.

Whether this actually helps is an empirical question, answered afterwards by
re-running the vowel-duration comparison against SLR86. If YECS-trained
alignment does not bring the median vowel near 80 ms, the extra data is not
earning its place and the old model should stay.

    MFA_JOBS=12 python scripts/train_acoustic_model.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.mfa import MFAAligner  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402

logger = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path,
                        default=PROJECT_ROOT / "data" / "train_corpus")
    parser.add_argument("--dictionary", type=Path,
                        default=PROJECT_ROOT / "data" / "dictionary" / "train.dict")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "yoruba_acoustic_v2.zip")
    parser.add_argument("--aligned", type=Path,
                        default=PROJECT_ROOT / "output" / "train_aligned",
                        help="where to write the alignments training produces")
    parser.add_argument("--temporary-directory", type=Path,
                        default=PROJECT_ROOT / "output" / "mfa_train_tmp")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    config = EngineConfig()
    aligner = MFAAligner(config)
    # The acoustic model is what we are producing, so do not require one.
    aligner.check_available(require_acoustic_model=False)

    utterances = sum(1 for _ in args.corpus.rglob("*.lab"))
    speakers = sum(1 for p in args.corpus.iterdir() if p.is_dir())
    logger.info("Training on %d utterances from %d speakers", utterances, speakers)
    logger.info("MFA jobs: %d", config.mfa.num_jobs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.aligned.mkdir(parents=True, exist_ok=True)

    command = [
        aligner.mfa.executable, "train",
        str(args.corpus), str(args.dictionary), str(args.out),
        "--output_directory", str(args.aligned),
        "--num_jobs", str(config.mfa.num_jobs),
        "--temporary_directory", str(args.temporary_directory),
        "--clean",
    ]
    logger.info("Running: %s", " ".join(command))

    started = time.time()
    result = aligner._run(command, check=False)
    elapsed = (time.time() - started) / 3600

    if result.returncode != 0:
        logger.error("mfa train failed (exit %d)", result.returncode)
        sys.stderr.write(result.stderr[-4000:])
        return 1

    grids = len(list(args.aligned.rglob("*.TextGrid")))
    logger.info("Trained in %.2f h. Model: %s", elapsed, args.out)
    logger.info("%d TextGrids written to %s (%.1f%% of utterances aligned)",
                grids, args.aligned, 100 * grids / max(utterances, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
