"""Choose which utterances to extract wav2vec features from, by what they contain.

The first extraction sampled utterances *evenly* so that speakers stayed
represented. That is the right rule for speaker diversity and the wrong one for
class balance, and it cost the classes that were already failing:

    class   extracted   available in the alignments   unused
    En          2,937        4,504                      35%
    in          9,102       41,338                      78%
    an         12,575       43,205                      71%
    On         17,107       52,732                      68%
    un         18,983       60,937                      69%

Meanwhile 134,557 ``i`` were extracted, a class already at 0.926 F1. Seventy per
cent of the material for the weakest classes was left on disk to make room for
one that needed nothing.

This selects utterances by how many rare vowels they carry, so the extraction
budget goes where the classifier is short. Utterances containing ``ẹn`` are taken
first and in full, then the rest by rare-vowel count, then a spread of ordinary
utterances so common classes and speaker variety are not starved.

Features are still extracted from **whole utterances**, not from cut windows.
wav2vec2 attends across its entire input, so a 400 ms window does not produce
the representation a full utterance does, and the engine feeds full utterances at
inference. Training on windowed features would rebuild the train/inference
mismatch that once made this engine score 0 out of 23 on real audio.

    python scripts/select_extraction_set.py --target 60000
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

RARE = ("En", "in", "an", "On", "un")
#: ``ẹn`` is scarcer than the other nasals by an order of magnitude, so every
#: utterance containing one is taken before anything else is considered.
SCARCEST = "En"
ARTEFACT = {"en": "e", "on": "o"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path,
                        default=PROJECT_ROOT / "output" / "train_aligned")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "output" / "aligned_v3")
    parser.add_argument("--target", type=int, default=60_000,
                        help="how many utterances to extract from")
    parser.add_argument("--spread", type=float, default=0.25,
                        help="share of the budget reserved for ordinary "
                             "utterances, keeping common classes and speaker "
                             "variety from being starved (default 0.25)")
    args = parser.parse_args()

    grids = sorted(args.aligned.rglob("*.TextGrid"))
    print(f"surveying {len(grids):,} alignments")

    profile: dict[Path, Counter] = {}
    for number, grid in enumerate(grids, start=1):
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
        except Exception:
            continue
        counts = Counter()
        for phone in alignment.phones:
            if not is_vowel_phone(phone.label):
                continue
            base = phone.label.split("_")[0]
            counts[ARTEFACT.get(base, base)] += 1
        if counts:
            profile[grid] = counts
        if number % 25000 == 0:
            print(f"  {number:,}", flush=True)

    scarcest = [g for g, c in profile.items() if c[SCARCEST]]
    others = [g for g in profile if g not in set(scarcest)]
    ranked = sorted(others, key=lambda g: -sum(profile[g][r] for r in RARE))

    reserve = int(args.target * args.spread)
    chosen = list(scarcest)
    for grid in ranked:
        if len(chosen) >= args.target - reserve:
            break
        chosen.append(grid)

    # A random tail, so the set is not purely nasal-heavy text.
    remaining = [g for g in ranked if g not in set(chosen)]
    random.Random(0).shuffle(remaining)
    chosen += remaining[: max(args.target - len(chosen), 0)]

    total = Counter()
    for grid in chosen:
        total.update(profile[grid])
    available = Counter()
    for counts in profile.values():
        available.update(counts)

    print(f"\nselected {len(chosen):,} of {len(profile):,} utterances "
          f"({len(scarcest):,} contain {SCARCEST})")
    print(f"\n{'class':<6}{'selected':>12}{'available':>12}{'share':>8}")
    for label in sorted(available):
        share = total[label] / max(available[label], 1)
        print(f"{label:<6}{total[label]:>12,}{available[label]:>12,}{share:>7.0%}"
              + ("  <- rare" if label in RARE else ""))
    print(f"\n{sum(total.values()):,} vowels total")

    args.out.mkdir(parents=True, exist_ok=True)
    for existing in args.out.iterdir():
        existing.unlink()
    for grid in chosen:
        (args.out / grid.name).symlink_to(grid.resolve())
    print(f"\nlinked into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
