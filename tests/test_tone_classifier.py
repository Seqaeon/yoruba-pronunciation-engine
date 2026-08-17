"""Tests for the trained tone classifier and its feature extraction.

No trained model or audio is needed: features are computed from synthetic vowel
segments, and the classifier is exercised with a stub estimator. What matters
here is the contract between the analyzer and the classifier, not the accuracy
of any particular fitted model.
"""

from __future__ import annotations

import pytest

from yoruba_engine.config import ToneConfig
from yoruba_engine.datatypes import Tone, WordAnalysis
from yoruba_engine.exceptions import ToneAnalysisError
from yoruba_engine.tone.analyzer import ToneAnalyzer
from yoruba_engine.tone.classifier import (
    TONE_ORDER,
    ToneClassifier,
    check_feature_count,
)
from yoruba_engine.tone.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    extract_features,
    measurable,
)

from test_tone import make_vowel


class StubModel:
    """Always predicts one tone, with a fixed probability."""

    def __init__(self, tone: Tone = Tone.HIGH, probability: float = 0.9) -> None:
        self.index = TONE_ORDER.index(tone)
        self.probability = probability
        self.seen_rows: list[list[float]] = []

    def predict_proba(self, rows):
        self.seen_rows = list(rows)
        distribution = [(1 - self.probability) / 2] * 3
        distribution[self.index] = self.probability
        return [distribution for _ in rows]


class TestFeatures:
    def test_row_per_measurable_vowel(self):
        vowels = [make_vowel(v, start=i * 0.2) for i, v in enumerate([0.0, 2.0, -2.0])]
        ToneAnalyzer(remove_downdrift=False).analyse(
            [WordAnalysis(word="w", start_s=0, end_s=1, vowels=vowels)]
        )
        rows = extract_features(vowels)
        assert len(rows) == 3
        assert all(len(row) == FEATURE_COUNT for row in rows)

    def test_names_match_row_width(self):
        assert len(FEATURE_NAMES) == FEATURE_COUNT

    def test_unmeasurable_vowels_are_excluded(self):
        good = make_vowel(0.0)
        bad = make_vowel(0.0, frames=1)
        bad.contour = bad.contour[:1]
        vowels = [good, bad]
        ToneAnalyzer(remove_downdrift=False).analyse(
            [WordAnalysis(word="w", start_s=0, end_s=1, vowels=vowels)]
        )
        assert len(measurable(vowels)) == 1
        assert len(extract_features(vowels)) == 1

    def test_empty_input(self):
        assert extract_features([]) == []

    def test_context_features_reflect_neighbours(self):
        vowels = [make_vowel(v, start=i * 0.2) for i, v in enumerate([0.0, 4.0, 0.0])]
        ToneAnalyzer(remove_downdrift=False).analyse(
            [WordAnalysis(word="w", start_s=0, end_s=1, vowels=vowels)]
        )
        rows = extract_features(vowels)
        delta_prev = FEATURE_NAMES.index("delta_prev")
        # The middle vowel sits 4 semitones above the one before it.
        assert rows[1][delta_prev] == pytest.approx(4.0, abs=0.5)

    def test_check_feature_count_rejects_wrong_width(self):
        with pytest.raises(ToneAnalysisError, match="Retrain"):
            check_feature_count([[0.0, 1.0]])


class TestClassifierIntegration:
    def _word(self, levels):
        return WordAnalysis(
            word="test", start_s=0, end_s=1,
            vowels=[make_vowel(v, start=i * 0.2) for i, v in enumerate(levels)],
        )

    def test_classifier_overrides_the_rules(self):
        """A level that the rules would call Low is reported as the model says."""
        classifier = ToneClassifier(model=StubModel(Tone.HIGH))
        analyzer = ToneAnalyzer(classifier=classifier, remove_downdrift=False)

        word = self._word([-6.0])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.HIGH

    def test_confidence_comes_from_the_probability(self):
        classifier = ToneClassifier(model=StubModel(Tone.LOW, probability=0.77))
        analyzer = ToneAnalyzer(classifier=classifier, remove_downdrift=False)

        word = self._word([0.0])
        analyzer.analyse([word])
        assert word.vowels[0].confidence == pytest.approx(0.77)

    def test_unmeasurable_vowel_stays_unknown(self):
        """Both classifier and rule paths must honour the same contract."""
        classifier = ToneClassifier(model=StubModel(Tone.HIGH))
        analyzer = ToneAnalyzer(classifier=classifier, remove_downdrift=False)

        vowel = make_vowel(0.0, frames=1)
        vowel.contour = vowel.contour[:1]
        word = WordAnalysis(word="test", start_s=0, end_s=1, vowels=[vowel])
        analyzer.analyse([word])

        assert vowel.detected_tone is Tone.UNKNOWN
        assert vowel.confidence == 0.0

    def test_model_receives_one_row_per_measurable_vowel(self):
        model = StubModel()
        analyzer = ToneAnalyzer(classifier=ToneClassifier(model=model), remove_downdrift=False)
        analyzer.analyse([self._word([0.0, 1.0, 2.0])])
        assert len(model.seen_rows) == 3

    def test_no_classifier_falls_back_to_rules(self):
        analyzer = ToneAnalyzer(
            ToneConfig(prev_weight=0, next_weight=0, slope_weight=0),
            remove_downdrift=False,
        )
        word = self._word([-6.0])
        analyzer.analyse([word])
        assert word.vowels[0].detected_tone is Tone.LOW

    def test_load_missing_model_raises_with_guidance(self, tmp_path):
        with pytest.raises(ToneAnalysisError, match="train_tone_classifier"):
            ToneClassifier.load(tmp_path / "absent.joblib")


class TestMinimalPairIndex:
    """Minimal-pair discrimination reframes an open-set question ("is this
    vowel implausible?") as a two-way one ("òkun or òkan?"), which is far
    easier and produces feedback naming the word actually heard."""

    def _dictionary(self, tmp_path, entries):
        path = tmp_path / "d.dict"
        path.write_text("\n".join(f"{w}\t{p}" for w, p in entries), encoding="utf-8")
        return path

    def test_finds_words_differing_in_one_vowel(self, tmp_path):
        from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex

        index = MinimalPairIndex.from_dictionary(self._dictionary(tmp_path, [
            ("òkun", "o k un"), ("òkan", "o k an"), ("òkú", "o k u"),
            ("omi", "o m i"),
        ]))
        competitors = {c.word for c in index.competitors("òkun")}
        assert "òkan" in competitors
        assert "òkú" in competitors
        assert "omi" not in competitors      # differs in a consonant too

    def test_identical_vowel_is_not_a_competitor(self, tmp_path):
        from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex

        index = MinimalPairIndex.from_dictionary(self._dictionary(tmp_path, [
            ("òkun", "o k un"), ("okun", "o k un"),
        ]))
        assert all(c.competing_phone != "un" for c in index.competitors("òkun"))

    def test_probabilistic_dictionary_format(self, tmp_path):
        from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex

        path = tmp_path / "d.dict"
        path.write_text("òkun\t0.99\t0.05\t2.4\t0.7\to k un\n"
                        "òkan\t0.99\t0.05\t2.4\t0.7\to k an\n", encoding="utf-8")
        index = MinimalPairIndex.from_dictionary(path)
        assert {c.word for c in index.competitors("òkun")} == {"òkan"}

    def test_competitor_wins_when_audio_supports_it(self, tmp_path):
        from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex, best_competitor

        index = MinimalPairIndex.from_dictionary(self._dictionary(tmp_path, [
            ("òkun", "o k un"), ("òkan", "o k an"),
        ]))
        labels = {"o": 0, "un": 1, "an": 2}
        # Second vowel looks strongly like 'un' though the prompt wanted 'an'.
        probabilities = [{"o": 0.9, "un": 0.05, "an": 0.05},
                         {"o": 0.05, "un": 0.90, "an": 0.02}]

        verdict = best_competitor("òkan", index.competitors("òkan"), probabilities, labels)
        assert verdict is not None
        assert verdict.competitor == "òkun"
        assert verdict.decisive
        assert "òkun" in verdict.describe()

    def test_no_verdict_when_prompt_is_the_best_explanation(self, tmp_path):
        from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex, best_competitor

        index = MinimalPairIndex.from_dictionary(self._dictionary(tmp_path, [
            ("òkun", "o k un"), ("òkan", "o k an"),
        ]))
        labels = {"o": 0, "un": 1, "an": 2}
        probabilities = [{"o": 0.9, "un": 0.05, "an": 0.05},
                         {"o": 0.05, "un": 0.90, "an": 0.02}]

        assert best_competitor("òkun", index.competitors("òkun"), probabilities, labels) is None
