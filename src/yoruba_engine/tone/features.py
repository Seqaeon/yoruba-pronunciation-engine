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


#: How many points of each unit's pitch contour to sample.
#:
#: The 14 summary features compress a whole contour into ``excursion``, one
#: number -- and permutation importance ranks it first, worth +0.092 accuracy on
#: held-out speakers. That is the model saying shape matters more than anything
#: else it is given, through the narrowest channel it has.
#:
#: Eight points is a compromise. A Yoruba vowel runs about 95 ms, which at
#: Praat's default step is roughly nine or ten pitch measurements, so eight
#: samples is near the real resolution rather than interpolating detail that was
#: never measured.
CONTOUR_POINTS = 8


def contour_shape(vowel: VowelSegment, points: int = CONTOUR_POINTS) -> list[float]:
    """Resample one unit's pitch contour to a fixed number of points.

    Returned **relative to the unit's own mean**, so this describes shape and
    nothing else. Height is already carried by ``level`` and ``level_z``, which
    have had declination removed; the raw contour has not, and subtracting the
    unit's mean makes that irrelevant rather than wrong.

    A unit with no usable pitch returns zeros, which is the same thing the
    summary features do for it -- and such units are excluded from scoring by
    :func:`measurable` anyway.
    """
    values = [p.semitones for p in vowel.contour if p.semitones is not None]
    if not values:
        return [0.0] * points
    centre = sum(values) / len(values)
    if len(values) == 1:
        return [0.0] * points

    # Linear resampling onto a fixed grid. The contour is short and already
    # smoothed by the pitch tracker, so anything fancier would be inventing
    # structure.
    out: list[float] = []
    last = len(values) - 1
    for index in range(points):
        position = index * last / (points - 1)
        low = int(position)
        high = min(low + 1, last)
        weight = position - low
        sample = values[low] * (1 - weight) + values[high] * weight
        out.append(float(sample - centre))
    return out


def extract_sequence_features(vowels: list[VowelSegment],
                              points: int = CONTOUR_POINTS) -> list[list[float]]:
    """The summary features with each unit's contour shape appended.

    One function so training and inference cannot drift. This project has
    already been bitten by two implementations of one table disagreeing; a
    feature vector assembled twice would be the same failure with no visible
    symptom, since a model fed the wrong column order still returns confident
    predictions.
    """
    rows = extract_features(vowels)
    if not rows:
        return rows
    return [row + contour_shape(vowel, points)
            for row, vowel in zip(rows, measurable(vowels))]


#: Width of the vector :func:`extract_sequence_features` produces.
SEQUENCE_FEATURE_COUNT = FEATURE_COUNT + CONTOUR_POINTS
