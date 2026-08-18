"""Combine the staged corpora into one MFA training corpus.

The shipped acoustic model was trained on 37.93 hours from 2,047 speakers
(michsethowusu, Waxal, local TTS, OpenSLR), but with the tone-mark filter
disabled, so undiacritized lines taught it to conflate ``ọ`` with ``o``. One
pass over YECS with it left 62% of the audio unlabelled and put the median
vowel at 30 ms against 80 ms on SLR86, so its coverage does not reach this
speech. Every corpus combined here requires diacritics.

Training reads every corpus at once. Real directories hold symlinked files
rather than the directories themselves being symlinks: neither ``Path.rglob``
nor ``os.walk`` descends into a symlinked directory by default, so linking at
the directory level produces a corpus that silently reads as empty. Each
subdirectory is one speaker, which is what speaker-adaptive training needs, and
the audio is already 16 kHz mono everywhere, so nothing is copied or converted.

Prefixes keep speakers distinct across corpora, so a YECS ``SPK_0002`` and an
SLR86 ``yof_00295`` cannot collide in the speaker-adaptation statistics.

    python scripts/build_training_corpus.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCES = (
    ("slr86", "mfa_corpus"),
    ("studio", "studio_corpus"),
    ("yecs", "yecs_corpus"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "train_corpus")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{'corpus':<10}{'speakers':>10}{'utterances':>13}")
    grand_speakers = grand_utterances = 0

    for prefix, name in SOURCES:
        root = args.data / name
        if not root.exists():
            print(f"{prefix:<10}{'missing':>10}")
            continue
        speakers = utterances = 0
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            staged = args.out / f"{prefix}__{directory.name}"
            staged.mkdir(exist_ok=True)
            speakers += 1
            for lab in directory.glob("*.lab"):
                audio = lab.with_suffix(".wav")
                if not audio.exists():
                    continue
                for source in (lab, audio):
                    link = staged / source.name
                    if not link.exists():
                        link.symlink_to(source.resolve())
                utterances += 1
        print(f"{prefix:<10}{speakers:>10,}{utterances:>13,}")
        grand_speakers += speakers
        grand_utterances += utterances

    print(f"{'TOTAL':<10}{grand_speakers:>10,}{grand_utterances:>13,}")
    print(f"\nTraining corpus at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
