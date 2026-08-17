"""Tests for word alignment, scoring, and feedback generation."""

from __future__ import annotations

import pytest

from yoruba_engine.compare.engine import ComparisonEngine, align_words
from yoruba_engine.config import ScoringConfig
from yoruba_engine.datatypes import PitchPoint, Tone, UtteranceAnalysis, VowelSegment, WordAnalysis
from yoruba_engine.exceptions import ComparisonError
from yoruba_engine.tone.analyzer import ToneAnalyzer


class TestAlignWords:
    def test_identical_sequences(self):
        result = align_words(["mo", "wà"], ["mo", "wà"])
        assert result.matches == 2
        assert result.error_rate == 0.0

    def test_substitution(self):
        result = align_words(["mo", "wà"], ["mo", "lọ"])
        assert result.substitutions == 1
        assert result.matches == 1

    def test_deletion(self):
        result = align_words(["mo", "fẹ́", "jẹun"], ["mo", "jẹun"])
        assert result.deletions == 1
        assert (1, None) in result.pairs

    def test_insertion(self):
        result = align_words(["mo", "jẹun"], ["mo", "fẹ́", "jẹun"])
        assert result.insertions == 1
        assert any(ref is None for ref, _ in result.pairs)

    def test_tone_differences_are_not_word_errors(self):
        """òrí vs órí is a tone error, and must not also count as a word error."""
        result = align_words(["òrí"], ["órí"])
        assert result.matches == 1
        assert result.substitutions == 0

    def test_dot_below_difference_is_a_word_error(self):
        """ẹ and e are different phonemes, so this is a genuine word error."""
        result = align_words(["ẹsẹ̀"], ["ese"])
        assert result.substitutions == 1

    def test_empty_learner_input(self):
        result = align_words(["mo", "wà"], [])
        assert result.deletions == 2
        assert result.error_rate == 1.0

    def test_empty_reference(self):
        result = align_words([], ["mo"])
        assert result.insertions == 1
        assert result.error_rate == 0.0  # no reference to measure against


def make_utterance(
    utt_id: str, words: list[tuple[str, list[float]]]
) -> UtteranceAnalysis:
    """Build an analysed utterance from (word, [semitones per vowel]) pairs."""
    analyses = []
    for word_index, (word, tones) in enumerate(words):
        vowels = []
        for vowel_index, semitones in enumerate(tones):
            start = word_index * 0.5 + vowel_index * 0.15
            contour = [
                PitchPoint(time_s=start + i * 0.01, f0_hz=200.0, semitones=semitones)
                for i in range(10)
            ]
            vowels.append(
                VowelSegment(
                    label="a_M", start_s=start, end_s=start + 0.1,
                    word_index=word_index, contour=contour,
                )
            )
        analyses.append(
            WordAnalysis(
                word=word, start_s=word_index * 0.5, end_s=word_index * 0.5 + 0.4, vowels=vowels
            )
        )

    ToneAnalyzer(remove_downdrift=False).analyse(analyses)
    return UtteranceAnalysis(
        utterance_id=utt_id,
        audio_path=__import__("pathlib").Path(f"{utt_id}.wav"),
        transcript=" ".join(w for w, _ in words),
        words=analyses,
        speaker_median_hz=200.0,
    )


class TestComparisonEngine:
    def test_perfect_match_scores_one(self):
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("òrí", [-4.0, 4.0])])

        result = ComparisonEngine().compare(reference, learner)
        assert result.text_similarity == 1.0
        assert result.tone_similarity == 1.0
        assert result.overall_score == 1.0
        assert result.tone_errors == []

    def test_brief_example_tone_mismatch(self):
        """Reference Òrí vs learner Órí: right word, wrong tone."""
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("órí", [4.0, 4.0])])

        result = ComparisonEngine().compare(reference, learner)
        assert result.text_similarity == 1.0  # same word, tone-insensitively
        assert result.tone_similarity == 0.5  # one of two vowels wrong
        assert len(result.tone_errors) == 1
        assert result.tone_errors[0].expected is Tone.LOW
        assert result.tone_errors[0].detected is Tone.HIGH

    def test_wrong_word_lowers_text_score(self):
        reference = make_utterance("ref", [("mo", [0.0]), ("wà", [-4.0])])
        learner = make_utterance("learner", [("mo", [0.0]), ("lọ", [-4.0])])

        result = ComparisonEngine().compare(reference, learner)
        assert result.text_similarity == 0.5

    def test_scoring_weights_are_applied(self):
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("órí", [4.0, 4.0])])

        engine = ComparisonEngine(ScoringConfig(text_weight=0.2, tone_weight=0.8))
        result = engine.compare(reference, learner)
        assert result.overall_score == pytest.approx(0.2 * 1.0 + 0.8 * 0.5)

    def test_empty_reference_raises(self):
        reference = make_utterance("ref", [])
        learner = make_utterance("learner", [("mo", [0.0])])
        with pytest.raises(ComparisonError, match="no aligned words"):
            ComparisonEngine().compare(reference, learner)

    def test_feedback_is_generated(self):
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("órí", [4.0, 4.0])])

        result = ComparisonEngine().compare(reference, learner)
        assert result.feedback
        assert any("Tone mismatch" in line for line in result.feedback)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(text_weight=0.5, tone_weight=0.9)


class TestPatternDescription:
    def test_matches_the_brief_output_format(self):
        text = ComparisonEngine.describe_pattern_mismatch(
            "òrí", [Tone.LOW, Tone.HIGH], [Tone.HIGH, Tone.HIGH]
        )
        assert "Expected: Low-High" in text
        assert "Detected: High-High" in text
        assert "first vowel" in text


class TestVerification:
    """Forced alignment cannot fail, so a learner who says something entirely
    different still produces an alignment. The pipeline must not report that as
    a perfect score."""

    def _analysis(self, utt_id, words, coverage):
        analysis = make_utterance(utt_id, words)
        analysis.audio_duration_s = 3.0
        analysis.speech_coverage = coverage
        return analysis

    def _pipeline(self):
        from yoruba_engine.pipeline import AssessmentPipeline

        # No MFA or model needed: only _verify_attempt is exercised.
        return AssessmentPipeline.__new__(AssessmentPipeline)

    def test_low_coverage_marks_the_attempt_unverified(self):
        from yoruba_engine.config import EngineConfig
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline()
        pipeline.config = EngineConfig()

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7)
        learner = self._analysis("learner", [("òrí", [-4.0, 4.0])], 0.10)
        result = ComparisonEngine().compare(reference, learner)
        assert result.overall_score == 1.0  # before verification

        AssessmentPipeline._verify_attempt(pipeline, result, learner)

        assert result.verified is False
        assert result.overall_score == 0.0
        assert "could not confirm" in result.feedback[0].lower()
        assert result.verification_note is not None

    def test_normal_coverage_passes_untouched(self):
        from yoruba_engine.config import EngineConfig
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline()
        pipeline.config = EngineConfig()

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7)
        learner = self._analysis("learner", [("òrí", [-4.0, 4.0])], 0.7)
        result = ComparisonEngine().compare(reference, learner)

        AssessmentPipeline._verify_attempt(pipeline, result, learner)

        assert result.verified is True
        assert result.overall_score == 1.0
        assert result.learner_speech_coverage == 0.7

    def test_unknown_coverage_does_not_fail_the_learner(self):
        """If duration could not be measured, do not punish the attempt."""
        from yoruba_engine.config import EngineConfig
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline()
        pipeline.config = EngineConfig()

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7)
        learner = self._analysis("learner", [("òrí", [-4.0, 4.0])], None)
        result = ComparisonEngine().compare(reference, learner)

        AssessmentPipeline._verify_attempt(pipeline, result, learner)
        assert result.verified is True
