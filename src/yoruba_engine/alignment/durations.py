"""Expected word durations, used to tell spoken words from skipped ones.

Forced alignment must place every word of the prompt somewhere in the audio, so
a word the learner skipped does not disappear -- it is compressed into a sliver
between its neighbours. Comparing a word's aligned duration against what that
word *should* take is therefore the signal for "was this actually said?".

A single flat threshold does not work, because duration scales strongly with
word length. Measured on the training corpus:

===========  ============  ===========
Phone count  Median        5th pct
===========  ============  ===========
1            120 ms        50 ms
2            210 ms        120 ms
3            310 ms        200 ms
5            500 ms        370 ms
8            780 ms        550 ms
===========  ============  ===========

At a flat 60 ms, a five-phone word crushed from 500 ms to 100 ms passes as
"spoken". Scaling the threshold by phone count fixes both ends: short function
words like ``ti`` and the syllabic nasal ``ń`` are not falsely rejected, and long
words must actually occupy time to count.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger("durations")

#: Where ``scripts/measure_alignment_stats.py`` writes the measured table.
DEFAULT_STATS_PATH = PROJECT_ROOT / "models" / "word_duration_stats.json"

#: 5th-percentile duration in seconds by phone count, measured on the training
#: corpus. Used when no stats file is present so the engine still behaves
#: sensibly out of the box.
FALLBACK_P5_BY_PHONES: dict[int, float] = {
    1: 0.05, 2: 0.12, 3: 0.20, 4: 0.28, 5: 0.37, 6: 0.42, 7: 0.53, 8: 0.55,
}


class WordDurationModel:
    """Judges whether an aligned word occupied enough time to have been spoken.

    Args:
        p5_by_phones: 5th-percentile duration per phone count, in seconds.
        fraction: A word counts as spoken when its duration is at least this
            fraction of the 5th percentile for its length. The 5th percentile is
            already near the short end of normal speech, so a fraction of 0.5
            flags only words far below anything the corpus contains.
    """

    def __init__(
        self,
        p5_by_phones: dict[int, float] | None = None,
        fraction: float = 0.5,
    ) -> None:
        self.p5_by_phones = p5_by_phones or dict(FALLBACK_P5_BY_PHONES)
        self.fraction = fraction

    @classmethod
    def load(cls, path: Path | None = None, fraction: float = 0.5) -> WordDurationModel:
        """Load measured statistics, falling back to built-in defaults.

        A missing file is not an error: the fallback table was measured on this
        corpus and is a reasonable prior for any Yoruba speech.
        """
        stats_path = path or DEFAULT_STATS_PATH
        if not stats_path.exists():
            logger.info(
                "No duration stats at %s; using built-in defaults. Generate them "
                "with scripts/measure_alignment_stats.py",
                stats_path,
            )
            return cls(fraction=fraction)

        try:
            payload = json.loads(stats_path.read_text(encoding="utf-8"))
            table = {int(k): float(v) for k, v in payload["p5_by_phones"].items()}
        except (ValueError, KeyError, OSError) as exc:
            logger.warning("Could not read %s (%s); using defaults", stats_path, exc)
            return cls(fraction=fraction)

        logger.info("Loaded word duration stats for %d phone counts", len(table))
        return cls(p5_by_phones=table, fraction=fraction)

    def expected_p5(self, phone_count: int) -> float:
        """The 5th-percentile duration for a word of this many phones.

        Counts beyond the measured range are extrapolated from the longest
        measured entry, scaled per phone, so unusually long words are still
        judged sensibly.
        """
        if phone_count <= 0:
            return self.p5_by_phones.get(1, 0.05)
        if phone_count in self.p5_by_phones:
            return self.p5_by_phones[phone_count]

        longest = max(self.p5_by_phones)
        per_phone = self.p5_by_phones[longest] / longest
        return per_phone * phone_count

    def minimum_duration(self, phone_count: int) -> float:
        """Shortest duration a word of this length may have and count as spoken."""
        return self.fraction * self.expected_p5(phone_count)

    def was_spoken(self, duration_s: float, phone_count: int) -> bool:
        """Whether an aligned word plausibly corresponds to real speech."""
        return duration_s >= self.minimum_duration(phone_count)
