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

        engine = ComparisonEngine(
            ScoringConfig(text_weight=0.2, tone_weight=0.8, vowel_weight=0.0))
        result = engine.compare(reference, learner)
        assert result.overall_score == pytest.approx(0.2 * 1.0 + 0.8 * 0.5)

    def test_vowel_accuracy_is_scored_when_it_was_measured(self):
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("òrí", [-4.0, 4.0])])

        engine = ComparisonEngine(
            ScoringConfig(text_weight=0.4, tone_weight=0.4, vowel_weight=0.2))
        result = engine.compare(reference, learner, vowel_accuracy=0.5)
        assert result.overall_score == pytest.approx(0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5)

    def test_unmeasured_vowels_do_not_count_as_zero(self):
        """A recording whose vowels could not be judged must score as it did
        before vowel identity was weighed, not be punished for a measurement
        that never happened."""
        reference = make_utterance("ref", [("òrí", [-4.0, 4.0])])
        learner = make_utterance("learner", [("òrí", [-4.0, 4.0])])

        engine = ComparisonEngine(
            ScoringConfig(text_weight=0.4, tone_weight=0.4, vowel_weight=0.2))
        result = engine.compare(reference, learner, vowel_accuracy=None)
        assert result.overall_score == pytest.approx(1.0)

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

    def _pipeline(self, recognizes: bool = False):
        """A pipeline with only what ``_verify_attempt`` reads.

        ``recognizes`` selects which evidence the check uses: a recogniser makes
        it compare the heard words with the prompt, and its absence falls back
        to alignment coverage.
        """
        from yoruba_engine.config import EngineConfig
        from yoruba_engine.pipeline import AssessmentPipeline
        from yoruba_engine.asr.providers import (
            ASRTranscriptProvider, PromptedTranscriptProvider,
        )

        # No MFA or model needed: only _verify_attempt is exercised.
        pipeline = AssessmentPipeline.__new__(AssessmentPipeline)
        pipeline.config = EngineConfig()
        pipeline.transcript_provider = (
            ASRTranscriptProvider() if recognizes else PromptedTranscriptProvider()
        )
        return pipeline

    def test_low_coverage_marks_the_attempt_unverified(self):
        from yoruba_engine.config import EngineConfig
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline()

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

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7)
        learner = self._analysis("learner", [("òrí", [-4.0, 4.0])], None)
        result = ComparisonEngine().compare(reference, learner)

        AssessmentPipeline._verify_attempt(pipeline, result, learner)
        assert result.verified is True

    def test_recognised_transcript_gates_on_words_not_coverage(self):
        """With a recogniser, low coverage is not evidence of a wrong phrase.

        A learner who leaves silence at either end produces low coverage while
        saying exactly the right words. Before recognition that was
        indistinguishable from saying something else; now it is not, and the
        attempt must survive.
        """
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline(recognizes=True)

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7)
        learner = self._analysis("learner", [("òrí", [-4.0, 4.0])], 0.10)
        result = ComparisonEngine().compare(reference, learner)

        AssessmentPipeline._verify_attempt(pipeline, result, learner)

        assert result.verified is True
        assert result.overall_score == 1.0
        # Still reported, even though it did not decide the outcome.
        assert result.learner_speech_coverage == 0.10

    def test_recognised_transcript_rejects_a_different_phrase(self):
        """Words that are not the prompt's fail, however well they align."""
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline(recognizes=True)

        reference = self._analysis("ref", [("òrí", [-4.0, 4.0]), ("omi", [0.0, 0.0])], 0.7)
        learner = self._analysis("learner", [("bàbá", [-4.0, 4.0]), ("ilé", [0.0, 0.0])], 0.7)
        result = ComparisonEngine().compare(reference, learner)
        assert result.text_similarity < 0.4

        AssessmentPipeline._verify_attempt(pipeline, result, learner)

        assert result.verified is False
        assert result.overall_score == 0.0
        assert "could not confirm" in result.feedback[0].lower()

    def test_a_mangled_attempt_is_still_an_attempt(self):
        """Every word wrong is not the same as reading a different phrase.

        A real recognition of a genuine attempt at 'àbùbùtán bíi omi òkun' came
        back as 'àbògùtàn omi ọkàn': one word of four. Gating on word agreement
        scored that 0.25 and withheld the score, telling a learner who had
        attempted the prompt that they had not -- which is both false and the
        opposite of useful. It must verify.
        """
        from yoruba_engine.pipeline import AssessmentPipeline

        pipeline = self._pipeline(recognizes=True)

        result = ComparisonEngine().compare(
            self._analysis("ref", [("òrí", [-4.0, 4.0])], 0.7),
            self._analysis("learner", [("òrí", [-4.0, 4.0])], 0.7),
        )
        result.reference_transcript = "àbùbùtán bíi omi òkun"
        result.learner_transcript = "àbògùtàn omi ọkàn"

        AssessmentPipeline._verify_attempt(
            pipeline, result, self._analysis("learner", [("òrí", [-4.0, 4.0])], 0.7))

        assert result.verified is True
        assert result.overall_score == 1.0


class TestPhraseResemblance:
    """Distinguishing 'said it badly' from 'said something else'."""

    def test_a_mangled_rendition_still_resembles_the_prompt(self):
        from yoruba_engine.compare.engine import phrase_resemblance

        assert phrase_resemblance(
            "àbùbùtán bíi omi òkun", "àbògùtàn omi ọkàn") > 0.6

    def test_a_different_phrase_does_not(self):
        from yoruba_engine.compare.engine import phrase_resemblance

        assert phrase_resemblance(
            "Ẹ káàbọ̀", "àbùbùtàn bí omi òkun") < 0.5

    def test_tone_marks_do_not_count_against_the_learner(self):
        """The recogniser's tone marks drift; a tone error is reported
        separately, not read as a different phrase."""
        from yoruba_engine.compare.engine import phrase_resemblance

        assert phrase_resemblance("àbùbùtán", "àbùbùtàn") == 1.0

    def test_an_empty_transcript_resembles_nothing(self):
        from yoruba_engine.compare.engine import phrase_resemblance

        assert phrase_resemblance("Ẹ káàbọ̀", "") == 0.0



class TestResegmentation:
    """A recogniser's word boundaries are guesswork; the prompt's are not."""

    def test_a_merged_phrase_is_split_to_the_prompt(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert resegment_to_prompt("Ẹ káàbọ̀", "ekaabo").split() == ["e", "kaabo"]

    def test_an_over_split_phrase_is_joined(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert len(resegment_to_prompt("Ẹ káàbọ̀", "e ka bo").split()) == 2

    def test_correct_spacing_is_left_alone(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        prompt = "àbùbùtán bíi omi òkun"
        assert resegment_to_prompt(prompt, "àbùbùtàn bí omi òkun") == \
            "àbùbùtàn bí omi òkun"

    def test_a_wrong_sound_survives_the_respacing(self):
        """This must not quietly correct the learner towards the prompt."""
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert resegment_to_prompt("Ẹ káàbọ̀", "e ka mo").split()[1] == "kamo"

    def test_a_single_word_prompt_is_untouched(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert resegment_to_prompt("Jọ̀wọ́", "jo wo") == "jo wo"

    def test_nothing_heard_stays_nothing(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert resegment_to_prompt("Ẹ káàbọ̀", "") == ""

    def test_a_word_that_was_not_said_yields_no_token(self):
        """Fewer words out than the prompt has is the honest outcome, not a
        padded one: the learner did not produce that word."""
        from yoruba_engine.asr.segment import resegment_to_prompt

        out = resegment_to_prompt("Kí ni orúkọ rẹ", "orúkọ").split()
        assert len(out) < 4

    def test_tone_marks_are_preserved_in_the_output(self):
        from yoruba_engine.asr.segment import resegment_to_prompt

        assert "ọ́" in resegment_to_prompt("Ọwọ́ mi", "ọwọ́mi")
