"""Measure word-duration statistics from the aligned corpus.

Produces ``models/word_duration_stats.json``, which the engine uses to decide
whether a word in a learner's recording was actually spoken or merely squeezed
in by the aligner. See :mod:`yoruba_engine.alignment.durations`.

Run this after training the acoustic model, and again whenever you retrain it:
the statistics describe that model's alignment behaviour, not the language.

Usage::

    python scripts/measure_alignment_stats.py
    python scripts/measure_alignment_stats.py --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.durations import DEFAULT_STATS_PATH  # noqa: E402
from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.g2p.orthography import is_noise_token  # noqa: E402
from yoruba_engine.logging_utils import configure_logging, get_logger  # noqa: E402

logger = get_logger("measure_stats")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--out", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--limit", type=int, help="Use only the first N TextGrids")
    args = parser.parse_args(argv)

    configure_logging(level=logging.INFO)

    textgrids = sorted(args.aligned.rglob("*.TextGrid"))
    if not textgrids:
        raise SystemExit(f"No TextGrids in {args.aligned}. Train the acoustic model first.")
    if args.limit:
        textgrids = textgrids[: args.limit]

    by_phone_count: dict[int, list[float]] = defaultdict(list)
    parsed = 0

    for textgrid in textgrids:
        try:
            alignment = alignment_from_textgrid(textgrid)
        except Exception:  # noqa: BLE001 - one bad file must not stop the run
            continue
        parsed += 1
        for word in alignment.words:
            if is_noise_token(word.label):
                continue
            phone_count = len(alignment.phones_within(word))
            if phone_count:
                by_phone_count[phone_count].append(word.duration_s)

    # Phone counts with too few examples give unstable percentiles; drop them and
    # let the model extrapolate instead.
    p5 = {}
    print(f"{'phones':>7} {'count':>8} {'p5 ms':>8} {'median ms':>10}")
    for phone_count in sorted(by_phone_count):
        durations = sorted(by_phone_count[phone_count])
        if len(durations) < 30:
            continue
        value = durations[int(len(durations) * 0.05)]
        p5[phone_count] = round(value, 4)
        print(f"{phone_count:>7} {len(durations):>8} {value * 1000:>8.0f} "
              f"{statistics.median(durations) * 1000:>10.0f}")

    if not p5:
        raise SystemExit("Not enough aligned words to measure durations.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source_textgrids": parsed,
                "words_measured": sum(len(v) for v in by_phone_count.values()),
                "p5_by_phones": p5,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
