"""Stage 5: tone detection and comparison.

Classifies each vowel's normalized pitch as High, Mid, or Low, then compares a
learner's tone pattern against the reference.

Two design points worth understanding before tuning thresholds:

**Register normalization.** Tone is relative, not absolute. A low tone from a
bass-voiced man may sit at a higher absolute F0 than a high tone from a soprano.
All classification therefore happens in semitones relative to the *speaker's own*
median F0, computed per utterance.

**Downdrift.** In connected Yoruba speech, F0 declines gradually across an
utterance, so a High tone late in a sentence can be absolutely lower than a Mid
tone early on. :class:`ToneAnalyzer` optionally fits and removes this declination
trend before classifying. Leave it enabled for sentences; it is a no-op on
single words.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from ..config import ToneConfig
from ..logging_utils import get_logger
from ..datatypes import Tone, ToneError, VowelSegment, WordAnalysis

logger = get_logger("tone")


@dataclass
class ToneComparison:
    """Result of comparing two tone patterns."""

    errors: list[ToneError]
    matched: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.matched / self.total if self.total else 0.0


class ToneAnalyzer:
    """Assigns H/M/L labels to vowel segments and compares tone patterns."""

    def __init__(
        self,
        config: ToneConfig | None = None,
        remove_downdrift: bool = True,
        classifier: object | None = None,
    ) -> None:
        """
        Args:
            config: Classification thresholds.
            remove_downdrift: Fit and subtract the utterance-level F0 declination
                before classifying. Recommended for multi-word utterances.
            classifier: An optional trained
                :class:`~yoruba_engine.tone.classifier.ToneClassifier`. When
                given it replaces the threshold rules, which are otherwise the
                default. Both paths honour the same contract: unmeasurable
                vowels stay :attr:`Tone.UNKNOWN` with zero confidence.
        """
        self.config = config or ToneConfig()
        self.remove_downdrift = remove_downdrift
        self.classifier = classifier

    def analyse(self, words: list[WordAnalysis]) -> list[WordAnalysis]:
        """Detect the tone of every vowel, mutating and returning ``words``.

        Args:
            words: Word analyses carrying extracted pitch contours.

        Returns:
            The same list, with ``detected_tone``, ``mean_semitones``,
            ``slope_st_per_s`` and ``confidence`` populated on each vowel.
        """
        vowels = [v for word in words for v in word.vowels]
        if not vowels:
            logger.warning("No vowel segments to analyse")
            return words

        for vowel in vowels:
            self._measure(vowel)

        if self.remove_downdrift:
            self._remove_declination(vowels)

        self._score_in_context(vowels)

        if self.classifier is not None:
            # The classifier consumes the same measurements the rules do, so
            # everything above (trimming, declination removal, three-point
            # sampling) still applies.
            self.classifier.predict(vowels)
            for vowel in vowels:
                if vowel.mean_semitones is None:
                    vowel.detected_tone = Tone.UNKNOWN
                    vowel.confidence = 0.0
        else:
            for vowel in vowels:
                self._classify(vowel)

        detected = [v.detected_tone for v in vowels if v.detected_tone is not Tone.UNKNOWN]
        logger.info(
            "Classified %d/%d vowels (H=%d M=%d L=%d)",
            len(detected), len(vowels),
            detected.count(Tone.HIGH), detected.count(Tone.MID), detected.count(Tone.LOW),
        )
        return words

    def _measure(self, vowel: VowelSegment) -> None:
        """Compute mean semitones and F0 slope over the vowel's contour."""
        usable = [p for p in vowel.contour if p.semitones is not None]
        if len(usable) < self.config_min_frames:
            vowel.mean_semitones = None
            vowel.slope_st_per_s = None
            vowel.confidence = 0.0
            return

        semitones = [p.semitones for p in usable]
        vowel.mean_semitones = statistics.median(semitones)  # median resists octave jumps
        vowel.slope_st_per_s = self._slope(usable)

    @property
    def config_min_frames(self) -> int:
        """Minimum voiced frames required to classify a vowel."""
        return 3

    @staticmethod
    def _slope(points: list) -> float:
        """Least-squares slope of semitones over time, in st/s."""
        times = [p.time_s for p in points]
        values = [p.semitones for p in points]
        mean_t = sum(times) / len(times)
        mean_v = sum(values) / len(values)
        denominator = sum((t - mean_t) ** 2 for t in times)
        if denominator == 0:
            return 0.0
        return sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denominator

    def _remove_declination(self, vowels: list[VowelSegment]) -> None:
        """Subtract the linear F0 downtrend across the utterance.

        Fits a line through (vowel midpoint, mean semitones) across all measured
        vowels and removes it. With fewer than four vowels the fit is too
        unstable to trust, so it is skipped.
        """
        measured = [v for v in vowels if v.mean_semitones is not None]
        if len(measured) < 4:
            return

        times = [(v.start_s + v.end_s) / 2 for v in measured]
        values = [v.mean_semitones for v in measured]
        mean_t = sum(times) / len(times)
        mean_v = sum(values) / len(values)
        denominator = sum((t - mean_t) ** 2 for t in times)
        if denominator == 0:
            return

        slope = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values)) / denominator

        # Only correct a genuine downtrend. A positive slope is not downdrift and
        # removing it would destroy real tonal information.
        if slope >= 0:
            return

        logger.debug("Removing declination of %.2f st/s", slope)
        for vowel, time_s in zip(measured, times):
            vowel.mean_semitones -= slope * (time_s - mean_t)

    def _score_in_context(self, vowels: list[VowelSegment]) -> None:
        """Combine each vowel's own level with its neighbours and its slope.

        Yoruba tone is a *relative* system: what marks a syllable as High is
        largely that it sits above the syllables around it, not that it reaches
        an absolute frequency. Classifying each vowel in isolation throws that
        away, and it is the single largest source of Low/Mid confusion.

        The score is the vowel's normalized level, pushed up when it is higher
        than its neighbours, and adjusted by the rise or fall within the vowel:

            score = level
                  + prev_weight * (level - previous level)
                  + next_weight * (level - next level)
                  + slope_weight * (offset - onset)

        Measured on the training corpus, this recovers about 1.7 points of
        accuracy over thresholding the level alone (61.2% -> 62.9% held out).
        Weights live in :class:`~yoruba_engine.config.ToneConfig`.
        """
        measured = [v for v in vowels if v.mean_semitones is not None]

        for index, vowel in enumerate(measured):
            level = vowel.mean_semitones
            score = level

            if self.config.prev_weight and index > 0:
                score += self.config.prev_weight * (level - measured[index - 1].mean_semitones)
            if self.config.next_weight and index + 1 < len(measured):
                score += self.config.next_weight * (level - measured[index + 1].mean_semitones)

            excursion = vowel.excursion_st
            if self.config.slope_weight and excursion is not None:
                score += self.config.slope_weight * excursion

            vowel.tone_score = score

    def _classify(self, vowel: VowelSegment) -> None:
        """Assign H/M/L from the contextual score, and rate the confidence."""
        if vowel.mean_semitones is None:
            vowel.detected_tone = Tone.UNKNOWN
            vowel.confidence = 0.0
            return

        value = vowel.tone_score if vowel.tone_score is not None else vowel.mean_semitones
        if value >= self.config.high_threshold_st:
            vowel.detected_tone = Tone.HIGH
            margin = value - self.config.high_threshold_st
        elif value <= self.config.low_threshold_st:
            vowel.detected_tone = Tone.LOW
            margin = self.config.low_threshold_st - value
        else:
            vowel.detected_tone = Tone.MID
            margin = min(
                self.config.high_threshold_st - value,
                value - self.config.low_threshold_st,
            )

        # Confidence rises with distance from the decision boundary and falls
        # when the contour is steep (suggesting a bad segmentation) or short.
        boundary_confidence = min(1.0, max(0.0, margin) / 2.0)
        slope_penalty = 1.0
        if vowel.slope_st_per_s is not None:
            excess = abs(vowel.slope_st_per_s) - self.config.max_level_slope_st_per_s
            if excess > 0:
                slope_penalty = max(0.3, 1.0 - excess / 40.0)
        frame_confidence = min(1.0, len(vowel.contour) / 8.0)

        vowel.confidence = boundary_confidence * slope_penalty * frame_confidence

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        reference: list[WordAnalysis],
        learner: list[WordAnalysis],
        word_pairs: list[tuple[int | None, int | None]] | None = None,
    ) -> ToneComparison:
        """Compare learner tone patterns against the reference, word by word.

        Args:
            reference: Analysed reference words.
            learner: Analysed learner words.
            word_pairs: Optional index alignment from the text comparison stage,
                as ``(reference_index, learner_index)`` with ``None`` for
                insertions/deletions. When omitted, words are paired positionally.

        Returns:
            The comparison, listing every tonal mismatch.
        """
        if word_pairs is None:
            length = min(len(reference), len(learner))
            word_pairs = [(i, i) for i in range(length)]

        errors: list[ToneError] = []
        matched = 0
        total = 0

        for ref_index, learner_index in word_pairs:
            if ref_index is None or learner_index is None:
                continue  # a missing/extra word is a text error, not a tone error
            if ref_index >= len(reference) or learner_index >= len(learner):
                continue

            ref_word = reference[ref_index]
            learner_word = learner[learner_index]

            for vowel_index, (ref_vowel, learner_vowel) in enumerate(
                zip(ref_word.vowels, learner_word.vowels)
            ):
                # An unmeasurable vowel is not evidence of an error.
                if (
                    ref_vowel.detected_tone is Tone.UNKNOWN
                    or learner_vowel.detected_tone is Tone.UNKNOWN
                ):
                    continue

                total += 1
                if ref_vowel.detected_tone == learner_vowel.detected_tone:
                    matched += 1
                elif learner_vowel.confidence >= self.config.min_confidence:
                    errors.append(
                        ToneError(
                            word=ref_word.word,
                            word_index=ref_index,
                            vowel_index=vowel_index,
                            expected=ref_vowel.detected_tone,
                            detected=learner_vowel.detected_tone,
                        )
                    )
                else:
                    # Low-confidence disagreement: count it against the score but
                    # do not tell the learner something we are unsure about.
                    logger.debug(
                        "Suppressing low-confidence tone error on %r (conf %.2f)",
                        ref_word.word, learner_vowel.confidence,
                    )

        logger.info("Tone comparison: %d/%d vowels matched, %d errors", matched, total, len(errors))
        return ToneComparison(errors=errors, matched=matched, total=total)


def expected_pattern_from_text(word: str) -> list[Tone]:
    """Return the tone pattern a word *should* have, from its orthography.

    This is the ground truth when a reference recording is unavailable: Yoruba
    spelling encodes tone directly. Use it to validate the acoustic tone
    detector against the reference corpus.
    """
    from ..g2p.orthography import tone_pattern  # noqa: PLC0415 - avoids a cycle

    return tone_pattern(word)
