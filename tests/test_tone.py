"""Tests for tone detection and comparison.

Vowel segments are constructed directly with synthetic pitch contours, so these
tests exercise the classification logic without needing audio or MFA.
"""

from __future__ import annotations

import pytest

from yoruba_engine.config import ToneConfig
from yoruba_engine.datatypes import PitchPoint, Tone, VowelSegment, WordAnalysis
from yoruba_engine.tone.analyzer import ToneAnalyzer


def make_vowel(
    semitones: float,
    label: str = "a_M",
    start: float = 0.0,
    duration: float = 0.15,
    frames: int = 10,
    slope: float = 0.0,
) -> VowelSegment:
    """Build a vowel with a synthetic contour centred on ``semitones``."""
    contour = []
    for i in range(frames):
        time_s = start + (i / max(1, frames - 1)) * duration
        value = semitones + slope * (time_s - start - duration / 2)
        contour.append(PitchPoint(time_s=time_s, f0_hz=200.0, semitones=value))
    segment = VowelSegment(
        label=label, start_s=start, end_s=start + duration, word_index=0, contour=contour
    )
    # Mirror what PitchExtractor._sample_shape does, so segments built here
    # behave like ones that came out of real audio.
    values = [p.semitones for p in contour]
    if len(values) >= 3:
        count = len(values)
        segment.onset_semitones = values[int(count * 0.2)]
        segment.mid_semitones = values[int(count * 0.5)]
        segment.offset_semitones = values[min(count - 1, int(count * 0.8))]
    return segment


class TestClassification:
    @pytest.mark.parametrize(
        "semitones,expected",
        [(4.0, Tone.HIGH), (0.0, Tone.MID), (-4.0, Tone.LOW), (1.5, Tone.HIGH), (-1.5, Tone.LOW)],
    )
    def test_thresholds(self, semitones, expected):
        # Explicit symmetric thresholds: the shipped defaults are corpus-tuned
        # and asymmetric, which would make these boundary cases meaningless.
        analyzer = ToneAnalyzer(
            ToneConfig(high_threshold_st=1.2, low_threshold_st=-1.2),
            remove_downdrift=False,
        )
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[make_vowel(semitones)])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is expected

    def test_unmeasurable_vowel_is_unknown_not_mid(self):
        """A vowel with too few voiced frames must not be silently called Mid."""
        analyzer = ToneAnalyzer(remove_downdrift=False)
        vowel = make_vowel(0.0, frames=1)
        vowel.contour = vowel.contour[:1]
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[vowel])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.UNKNOWN
        assert word.vowels[0].confidence == 0.0

    def test_confidence_higher_far_from_boundary(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        clear = WordAnalysis(word="a", start_s=0, end_s=1, vowels=[make_vowel(6.0)])
        borderline = WordAnalysis(word="b", start_s=0, end_s=1, vowels=[make_vowel(1.25)])
        analyzer.analyse([clear])
        analyzer.analyse([borderline])
        assert clear.vowels[0].confidence > borderline.vowels[0].confidence

    def test_median_resists_a_single_octave_jump(self):
        """One doubled-F0 frame must not flip the classification."""
        analyzer = ToneAnalyzer(remove_downdrift=False)
        vowel = make_vowel(-4.0)
        vowel.contour[3].semitones = 8.0  # spurious octave error
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[vowel])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.LOW

    def test_custom_thresholds_respected(self):
        analyzer = ToneAnalyzer(ToneConfig(high_threshold_st=5.0, low_threshold_st=-5.0),
                                remove_downdrift=False)
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[make_vowel(3.0)])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.MID


class TestDowndrift:
    #: Declination is tested in isolation: contextual weights are zeroed so that
    #: neighbour effects cannot mask or mimic the downdrift correction.
    NO_CONTEXT = dict(prev_weight=0.0, next_weight=0.0, slope_weight=0.0)

    #: An H-M-L-H utterance with 2 st of downdrift accumulating per syllable.
    #: The final High ends up at -2 st absolute -- below the Low threshold --
    #: and is misread as Low unless declination is removed first.
    DOWNDRIFT_CASE = [4.0, -2.0, -8.0, -2.0]

    def test_final_high_is_misread_as_low_without_correction(self):
        analyzer = ToneAnalyzer(ToneConfig(**self.NO_CONTEXT), remove_downdrift=False)
        vowels = [make_vowel(v, start=i * 0.3) for i, v in enumerate(self.DOWNDRIFT_CASE)]
        word = WordAnalysis(word="test", start_s=0, end_s=2, vowels=vowels)
        analyzer.analyse([word])
        assert word.vowels[-1].detected_tone is Tone.LOW

    def test_declination_removal_recovers_the_final_high(self):
        analyzer = ToneAnalyzer(ToneConfig(**self.NO_CONTEXT), remove_downdrift=True)
        vowels = [make_vowel(v, start=i * 0.3) for i, v in enumerate(self.DOWNDRIFT_CASE)]
        word = WordAnalysis(word="test", start_s=0, end_s=2, vowels=vowels)
        analyzer.analyse([word])
        assert word.vowels[-1].detected_tone is Tone.HIGH

    def test_declination_removal_preserves_relative_ordering(self):
        """Detrending must not reorder the tones relative to one another."""
        analyzer = ToneAnalyzer(ToneConfig(**self.NO_CONTEXT), remove_downdrift=True)
        vowels = [make_vowel(v, start=i * 0.3) for i, v in enumerate(self.DOWNDRIFT_CASE)]
        word = WordAnalysis(word="test", start_s=0, end_s=2, vowels=vowels)
        analyzer.analyse([word])
        # The third syllable was the lowest before correction and must remain so.
        corrected = [v.mean_semitones for v in word.vowels]
        assert corrected.index(min(corrected)) == 2

    def test_not_applied_to_short_utterances(self):
        """With under four vowels the fit is unstable, so it is skipped."""
        analyzer = ToneAnalyzer(ToneConfig(**self.NO_CONTEXT), remove_downdrift=True)
        vowels = [make_vowel(4.0, start=0.0), make_vowel(-4.0, start=0.3)]
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=vowels)
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.HIGH
        assert word.vowels[1].detected_tone is Tone.LOW

    def test_rising_trend_is_not_treated_as_downdrift(self):
        analyzer = ToneAnalyzer(ToneConfig(**self.NO_CONTEXT), remove_downdrift=True)
        vowels = [make_vowel(-4.0 + 2.0 * i, start=i * 0.3) for i in range(4)]
        word = WordAnalysis(word="test", start_s=0, end_s=2, vowels=vowels)
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.LOW
        assert word.vowels[-1].detected_tone is Tone.HIGH


class TestContextualScoring:
    """Yoruba tone is relative, so neighbours and vowel shape feed the decision."""

    def test_score_is_recorded(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[make_vowel(0.0)])
        analyzer.analyse([word])
        assert word.vowels[0].tone_score is not None

    def test_vowel_above_its_neighbours_is_pushed_up(self):
        """Same absolute level, different surroundings, different verdict."""
        config = ToneConfig(prev_weight=0.5, next_weight=0.5, slope_weight=0.0)
        analyzer = ToneAnalyzer(config, remove_downdrift=False)

        vowels = [make_vowel(v, start=i * 0.2) for i, v in enumerate([-3.0, 0.0, -3.0])]
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=vowels)
        analyzer.analyse([word])
        raised = word.vowels[1].tone_score

        flat = [make_vowel(v, start=i * 0.2) for i, v in enumerate([0.0, 0.0, 0.0])]
        flat_word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=flat)
        analyzer.analyse([flat_word])

        assert raised > flat_word.vowels[1].tone_score

    def test_rising_vowel_scores_above_a_level_one(self):
        config = ToneConfig(prev_weight=0.0, next_weight=0.0, slope_weight=0.5)
        analyzer = ToneAnalyzer(config, remove_downdrift=False)

        rising = WordAnalysis(word="a", start_s=0, end_s=1,
                              vowels=[make_vowel(0.0, slope=20.0)])
        level = WordAnalysis(word="b", start_s=0, end_s=1,
                             vowels=[make_vowel(0.0, slope=0.0)])
        analyzer.analyse([rising])
        analyzer.analyse([level])

        assert rising.vowels[0].tone_score > level.vowels[0].tone_score

    def test_zero_weights_reduce_to_the_plain_level(self):
        config = ToneConfig(prev_weight=0.0, next_weight=0.0, slope_weight=0.0)
        analyzer = ToneAnalyzer(config, remove_downdrift=False)
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[make_vowel(1.5)])
        analyzer.analyse([word])
        assert word.vowels[0].tone_score == pytest.approx(word.vowels[0].mean_semitones)

    def test_three_point_samples_populated(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        vowel = make_vowel(0.0, slope=10.0)
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[vowel])
        analyzer.analyse([word])
        assert vowel.excursion_st > 0

    def test_unmeasurable_vowel_gets_no_score(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        vowel = make_vowel(0.0, frames=1)
        vowel.contour = vowel.contour[:1]
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[vowel])
        analyzer.analyse([word])
        assert vowel.tone_score is None
        assert vowel.detected_tone is Tone.UNKNOWN


class TestComparison:
    def _word(self, name: str, semitones: list[float]) -> WordAnalysis:
        return WordAnalysis(
            word=name, start_s=0, end_s=1,
            vowels=[make_vowel(s, start=i * 0.2) for i, s in enumerate(semitones)],
        )

    def test_identical_patterns_score_perfect(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        reference = [self._word("òrí", [-4.0, 4.0])]
        learner = [self._word("òrí", [-4.0, 4.0])]
        analyzer.analyse(reference)
        analyzer.analyse(learner)

        comparison = analyzer.compare(reference, learner)
        assert comparison.accuracy == 1.0
        assert comparison.errors == []

    def test_detects_the_brief_example(self):
        """Reference Òrí (Low-High) vs learner Órí (High-High)."""
        analyzer = ToneAnalyzer(remove_downdrift=False)
        reference = [self._word("òrí", [-4.0, 4.0])]
        learner = [self._word("órí", [4.0, 4.0])]
        analyzer.analyse(reference)
        analyzer.analyse(learner)

        comparison = analyzer.compare(reference, learner)
        assert len(comparison.errors) == 1
        error = comparison.errors[0]
        assert error.vowel_index == 0
        assert error.expected is Tone.LOW
        assert error.detected is Tone.HIGH
        assert "first vowel" in error.describe()

    def test_unknown_tones_are_not_counted_as_errors(self):
        analyzer = ToneAnalyzer(remove_downdrift=False)
        reference = [self._word("test", [4.0])]
        learner = [self._word("test", [4.0])]
        analyzer.analyse(reference)
        analyzer.analyse(learner)
        learner[0].vowels[0].detected_tone = Tone.UNKNOWN

        comparison = analyzer.compare(reference, learner)
        assert comparison.total == 0
        assert comparison.errors == []

    def test_low_confidence_mismatch_is_suppressed_from_feedback(self):
        """We count it against the score but do not tell the learner."""
        analyzer = ToneAnalyzer(remove_downdrift=False)
        reference = [self._word("test", [4.0])]
        learner = [self._word("test", [-4.0])]
        analyzer.analyse(reference)
        analyzer.analyse(learner)
        learner[0].vowels[0].confidence = 0.01

        comparison = analyzer.compare(reference, learner)
        assert comparison.errors == []
        assert comparison.total == 1
        assert comparison.matched == 0
