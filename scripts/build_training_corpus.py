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

**Transcripts are rewritten, not linked.** MFA looks up each written word in the
dictionary exactly as it appears and drops any utterance with a word it cannot
find, so the corpus and the dictionary have to agree character for character.
They did not: the dictionary is generated through ``tokenize(normalize(...))``,
which lowercases, strips edge punctuation and collapses bracketed annotations to
``[noise]``, while the staged corpora kept whatever their source wrote. SLR86
lines end in a full stop and carry ``[external]`` markers; YECS keeps sentence
capitalisation and the literal ``[english]`` token this project substitutes for
English words. Checked against the dictionary, 32% of SLR86 and 100% of YECS
transcripts contained a word MFA could not look up, which would have silently
dropped them from training.

Audio is still symlinked. Only the ``.lab`` files are written, and they are
written in the same canonical form the dictionary is built from.

    python scripts/build_training_corpus.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402

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
                link = staged / audio.name
                if not link.exists():
                    link.symlink_to(audio.resolve())
                text = " ".join(tokenize(
                    normalize(lab.read_text(encoding="utf-8")), keep_noise=True))
                if not text:
                    continue
                (staged / lab.name).write_text(text, encoding="utf-8")
                utterances += 1
        print(f"{prefix:<10}{speakers:>10,}{utterances:>13,}")
        grand_speakers += speakers
        grand_utterances += utterances

    print(f"{'TOTAL':<10}{grand_speakers:>10,}{grand_utterances:>13,}")
    print(f"\nTraining corpus at {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
