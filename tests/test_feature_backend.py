"""Tests for how the verifier chooses its feature backend.

This is the mechanism that prevents a feature/model mismatch, and that class of
bug has cost this project real time three times over: per-utterance cepstral
normalization once scored a correctly spoken word at p = 0.002; a 33-feature
model raised ``ValueError`` when handed 39 features; and both calibration
scripts silently measured a different feature path than production used.

The rule is that the *model* declares what it needs and the caller obeys, so
these tests assert the declaration is honoured rather than that some particular
backend is used.
"""

from __future__ import annotations

from yoruba_engine.config import SegmentalConfig
from yoruba_engine.datatypes import Interval
from yoruba_engine.segmental.verifier import VowelVerifier, words_from_alignment


class FakeAlignment:
    def __init__(self, words, phones):
        self.words = words
        self.phones = phones


class FakeClassifier:
    def __init__(self, metadata=None):
        self.labels = ["a", "an", "o", "O", "un"]
        self.model = object()
        self.metadata = metadata or {}


class TestBackendSelection:
    def test_wav2vec_metadata_selects_learned_features(self):
        verifier = VowelVerifier(
            FakeClassifier({"features_kind": "wav2vec2 layer 16",
                            "wav2vec_layer": 16}),
            SegmentalConfig())
        assert verifier.uses_wav2vec

    def test_absent_metadata_falls_back_to_praat(self):
        """A model that says nothing is an older one, trained on measurements."""
        verifier = VowelVerifier(FakeClassifier(), SegmentalConfig())
        assert not verifier.uses_wav2vec
        assert verifier.wav2vec is None

    def test_hand_designed_metadata_selects_praat(self):
        verifier = VowelVerifier(
            FakeClassifier({"features_kind": "39 hand-designed"}), SegmentalConfig())
        assert not verifier.uses_wav2vec

    def test_declared_layer_is_honoured(self):
        """The layer is part of the contract: layer 16 and layer 24 produce
        different representations, and swapping them silently would degrade
        accuracy without any error."""
        verifier = VowelVerifier(
            FakeClassifier({"features_kind": "wav2vec2 layer 8", "wav2vec_layer": 8}),
            SegmentalConfig())
        if verifier.wav2vec is not None:      # only when the checkpoint is present
            assert verifier.wav2vec.layer == 8

    def test_missing_checkpoint_disables_rather_than_crashes(self, monkeypatch):
        """An assessment must not fail outright because a 1.26 GB model is
        absent -- omission and tone checking still work without it."""
        from yoruba_engine.segmental import wav2vec as module

        monkeypatch.setattr(module.Wav2VecExtractor, "available",
                            property(lambda self: False))
        verifier = VowelVerifier(
            FakeClassifier({"features_kind": "wav2vec2 layer 16"}), SegmentalConfig())
        assert verifier.uses_wav2vec
        assert verifier.wav2vec is None

    def test_learned_path_returns_empty_when_backend_is_absent(self, monkeypatch):
        from yoruba_engine.segmental import wav2vec as module

        monkeypatch.setattr(module.Wav2VecExtractor, "available",
                            property(lambda self: False))
        verifier = VowelVerifier(
            FakeClassifier({"features_kind": "wav2vec2 layer 16"}), SegmentalConfig())
        result = verifier.verify(__import__("pathlib").Path("missing.wav"), [])
        assert result.checks == []
        assert result.mispronounced == []


class TestWordsFromAlignment:
    def _alignment(self):
        return FakeAlignment(
            words=[Interval("òkun", 0.0, 0.30), Interval("omi", 0.30, 0.60)],
            phones=[Interval("o", 0.00, 0.10), Interval("k", 0.10, 0.20),
                    Interval("un", 0.20, 0.30),
                    Interval("o", 0.30, 0.40), Interval("m", 0.40, 0.50),
                    Interval("i", 0.50, 0.60)],
        )

    def test_assigns_phones_to_the_containing_word(self):
        words = words_from_alignment(self._alignment())
        assert [w.phone_count for w in words] == [3, 3]

    def test_keeps_only_vowels_as_vowel_segments(self):
        words = words_from_alignment(self._alignment())
        assert [v.label for v in words[0].vowels] == ["o", "un"]
        assert [v.label for v in words[1].vowels] == ["o", "i"]

    def test_preserves_word_index_on_each_vowel(self):
        words = words_from_alignment(self._alignment())
        assert {v.word_index for v in words[1].vowels} == {1}

    def test_preserves_timings(self):
        words = words_from_alignment(self._alignment())
        assert words[0].vowels[1].start_s == 0.20
        assert words[0].vowels[1].end_s == 0.30

    def test_empty_alignment_yields_nothing(self):
        assert words_from_alignment(FakeAlignment([], [])) == []
