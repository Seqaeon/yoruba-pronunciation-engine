"""Feature extraction for the trained tone classifier.

Deliberately dependency-free: the same function produces the training matrix and
the inference-time vector, so the two can never drift apart. That is the usual
way a trained classifier silently degrades in production, and it is easy to avoid
by simply not writing the extraction twice.

The features encode the three things that carry Yoruba tone:

* **Level** -- where the vowel sits in the speaker's own register.
* **Context** -- how it compares to the vowels around it. Tone is relative; a
  High is High because it is above its neighbours.
* **Shape** -- the rise or fall across the vowel, which distinguishes a High
  approached from a Low from a plain Mid.
"""

from __future__ import annotations

import statistics

from ..datatypes import VowelSegment

#: Names in the order :func:`extract_features` emits them. Used for reporting
#: feature importances after training.
FEATURE_NAMES: tuple[str, ...] = (
    "level",
    "level_z",
    "excursion",
    "onset_rel",
    "offset_rel",
    "delta_prev",
    "delta_next",
    "delta_prev2",
    "position",
    "duration_s",
    "voiced_frames",
    "vowel_index_in_word",
    "vowels_in_word",
    "utterance_spread",
)

#: Number of features per vowel.
FEATURE_COUNT = len(FEATURE_NAMES)


def extract_features(vowels: list[VowelSegment]) -> list[list[float]]:
    """Build a feature row for every measurable vowel in an utterance.

    Args:
        vowels: The utterance's vowels in time order, after pitch extraction and
            declination removal. Vowels with no usable pitch are skipped.

    Returns:
        One row per *measurable* vowel, aligned with :func:`measurable`. Callers
        must use that helper to line predictions back up with the input.
    """
    measured = measurable(vowels)
    if not measured:
        return []

    levels = [v.mean_semitones for v in measured]
    mean_level = statistics.mean(levels)
    spread = statistics.pstdev(levels) if len(levels) > 1 else 0.0
    scale = spread or 1.0

    start = min(v.start_s for v in measured)
    end = max(v.end_s for v in measured)
    span = (end - start) or 1.0

    rows: list[list[float]] = []
    for index, vowel in enumerate(measured):
        level = vowel.mean_semitones

        previous = measured[index - 1].mean_semitones if index > 0 else level
        two_back = measured[index - 2].mean_semitones if index > 1 else previous
        following = measured[index + 1].mean_semitones if index + 1 < len(measured) else level

        excursion = vowel.excursion_st or 0.0
        onset = vowel.onset_semitones if vowel.onset_semitones is not None else level
        offset = vowel.offset_semitones if vowel.offset_semitones is not None else level

        rows.append([
            level,
            (level - mean_level) / scale,
            excursion,
            onset - level,
            offset - level,
            level - previous,
            level - following,
            level - two_back,
            ((vowel.start_s + vowel.end_s) / 2 - start) / span,
            vowel.end_s - vowel.start_s,
            float(len(vowel.contour)),
            float(_index_in_word(vowels, vowel)),
            float(sum(1 for v in vowels if v.word_index == vowel.word_index)),
            spread,
        ])

    return rows


def measurable(vowels: list[VowelSegment]) -> list[VowelSegment]:
    """Return the vowels that carry a usable pitch measurement.

    Feature rows are produced only for these, so predictions must be zipped back
    against this list rather than the full input.
    """
    return [v for v in vowels if v.mean_semitones is not None]


def _index_in_word(vowels: list[VowelSegment], target: VowelSegment) -> int:
    """Position of ``target`` among the vowels of its own word.

    Word-initial and word-final vowels behave differently: utterance-final
    lengthening and phrase-boundary effects both land on the last vowel.
    """
    position = 0
    for vowel in vowels:
        if vowel is target:
            return position
        if vowel.word_index == target.word_index:
            position += 1
    return position
