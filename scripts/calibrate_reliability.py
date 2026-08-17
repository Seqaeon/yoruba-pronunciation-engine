"""When is a vowel measurement trustworthy enough to judge?

Two ways a vowel verdict goes wrong, and they need different guards.

**Uncertainty.** The classifier spreads probability across several vowels
because the audio is genuinely ambiguous. A learner's ``ọkun`` read as
0.46/0.15/0.15/0.12 across four labels is the model saying it cannot tell. High
entropy catches this.

**Confident nonsense.** The classifier is sure, and wrong, because it is looking
at the wrong audio. Forced alignment must place every prompt word somewhere, so
an omitted word steals time from its neighbours: a final vowel that runs 170 ms
when spoken properly was squeezed to 30 ms in a recording that skipped the
preceding word, and the classifier read the neighbouring consonant with
entropy 0.10. Entropy cannot see this; only duration can.

Both thresholds are measured here on corpus speech, where every word was read
correctly, so any flag raised is a false alarm and any drop in P(expected) is
the measurement degrading rather than the speech.

    python scripts/calibrate_reliability.py --limit 250
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.segmental.classifier import VowelClassifier  # noqa: E402
from yoruba_engine.segmental.verifier import VowelVerifier, words_from_alignment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    classifier = VowelClassifier.load()
    verifier = VowelVerifier(classifier, config.segmental)
    labels = list(classifier.labels)

    index = {p.stem: p for p in args.audio.rglob("*.wav")}
    grids = [g for g in sorted(args.aligned.rglob("*.TextGrid"))[: args.limit]
             if g.stem in index]
    print(f"{len(grids)} utterances of correctly-read speech\n")

    rows = []          # (duration_ms, P(expected), entropy)
    for grid in grids:
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            words = words_from_alignment(alignment)
            result = verifier.verify(index[grid.stem], words)
        except Exception:
            continue
        for word_index, distributions in result.distributions.items():
            if word_index >= len(words):
                continue
            for vowel, distribution in zip(words[word_index].vowels, distributions):
                expected = vowel.label.split("_")[0]
                if expected not in labels:
                    continue
                p = np.asarray(distribution, dtype=float)
                entropy = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum())
                rows.append(((vowel.end_s - vowel.start_s) * 1000,
                             float(p[labels.index(expected)]), entropy))

    print(f"{len(rows):,} vowels measured\n")
    if not rows:
        return 1

    print("How duration affects the measurement:")
    print(f"{'duration':>14}{'n':>8}{'median P(exp)':>15}{'below 0.20':>12}{'median H':>10}")
    buckets = [(0, 30), (30, 50), (50, 80), (80, 120), (120, 1e9)]
    for low, high in buckets:
        group = [r for r in rows if low <= r[0] < high]
        if not group:
            continue
        label = f"{low:.0f}-{high:.0f} ms" if high < 1e9 else f"{low:.0f}+ ms"
        below = sum(1 for _, p, _ in group if p < config.segmental.suspect_probability)
        print(f"{label:>14}{len(group):>8,}"
              f"{statistics.median(p for _, p, _ in group):>15.3f}"
              f"{below / len(group):>11.1%}"
              f"{statistics.median(h for _, _, h in group):>10.2f}")

    print("\nA minimum duration below which a vowel is not judged:")
    print(f"{'floor':>8}{'vowels skipped':>17}{'false flags removed':>21}")
    flagged = [r for r in rows if r[1] < config.segmental.suspect_probability]
    for floor in (0, 20, 30, 40, 50, 60):
        skipped = sum(1 for d, _, _ in rows if d < floor)
        removed = sum(1 for d, _, _ in flagged if d < floor)
        print(f"{floor:>6} ms{skipped / len(rows):>16.1%}"
              f"{removed / max(len(flagged), 1):>20.1%}")

    print("\nAn entropy ceiling above which a vowel is not judged:")
    print(f"{'ceiling':>9}{'vowels skipped':>17}{'false flags removed':>21}")
    for ceiling in (3.0, 2.5, 2.0, 1.5, 1.0):
        skipped = sum(1 for _, _, h in rows if h > ceiling)
        removed = sum(1 for _, _, h in flagged if h > ceiling)
        print(f"{ceiling:>9.1f}{skipped / len(rows):>16.1%}"
              f"{removed / max(len(flagged), 1):>20.1%}")

    print(f"\n{len(flagged):,} of {len(rows):,} correctly-read vowels "
          f"({len(flagged)/len(rows):.1%}) currently fall below the "
          f"{config.segmental.suspect_probability} floor and are false alarms.")
    print("A guard is worth its cost when it removes a much larger share of those\n"
          "than of vowels overall -- otherwise it is just judging less.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
