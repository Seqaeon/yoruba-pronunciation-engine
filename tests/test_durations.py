"""Tests for the word-duration model that decides whether a word was spoken."""

from __future__ import annotations

import json

import pytest

from yoruba_engine.alignment.durations import (
    FALLBACK_P5_BY_PHONES,
    WordDurationModel,
)


class TestExpectedDuration:
    def test_scales_with_word_length(self):
        """The whole point: a one-phone word and an eight-phone word cannot
        share a threshold."""
        model = WordDurationModel()
        assert model.expected_p5(1) < model.expected_p5(4) < model.expected_p5(8)

    def test_extrapolates_beyond_measured_range(self):
        model = WordDurationModel(p5_by_phones={1: 0.05, 2: 0.12})
        assert model.expected_p5(6) > model.expected_p5(2)

    def test_zero_or_negative_falls_back_to_single_phone(self):
        model = WordDurationModel()
        assert model.expected_p5(0) == model.expected_p5(1)
        assert model.expected_p5(-3) == model.expected_p5(1)


class TestWasSpoken:
    def test_short_function_word_is_accepted(self):
        """'ti' is two phones and legitimately brief; it must not be discarded.
        Measured median on the corpus is 180 ms."""
        model = WordDurationModel()
        assert model.was_spoken(0.180, phone_count=2) is True
        assert model.was_spoken(0.120, phone_count=2) is True

    def test_syllabic_nasal_is_accepted(self):
        """'ń' is a single phone and can be very short indeed."""
        model = WordDurationModel()
        assert model.was_spoken(0.060, phone_count=1) is True

    def test_long_word_crushed_to_a_sliver_is_rejected(self):
        """A five-phone word normally runs ~500 ms. At 100 ms it was not said --
        this is the case a flat 60 ms threshold would wave through."""
        model = WordDurationModel()
        assert model.was_spoken(0.100, phone_count=5) is False

    def test_flat_threshold_would_have_passed_it(self):
        """Documents why the model is length-aware at all."""
        model = WordDurationModel()
        crushed = 0.100
        assert crushed > 0.06  # would pass a flat 60 ms rule
        assert model.was_spoken(crushed, phone_count=5) is False

    def test_fraction_controls_strictness(self):
        lenient = WordDurationModel(fraction=0.1)
        strict = WordDurationModel(fraction=0.95)
        assert lenient.was_spoken(0.05, phone_count=3) is True
        assert strict.was_spoken(0.05, phone_count=3) is False


class TestLoading:
    def test_missing_file_uses_builtin_defaults(self, tmp_path):
        model = WordDurationModel.load(tmp_path / "absent.json")
        assert model.p5_by_phones == FALLBACK_P5_BY_PHONES

    def test_reads_measured_stats(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text(json.dumps({"p5_by_phones": {"1": 0.04, "2": 0.12}}), encoding="utf-8")
        model = WordDurationModel.load(path)
        assert model.expected_p5(1) == pytest.approx(0.04)

    def test_corrupt_file_falls_back_rather_than_raising(self, tmp_path):
        path = tmp_path / "stats.json"
        path.write_text("not json at all", encoding="utf-8")
        model = WordDurationModel.load(path)
        assert model.p5_by_phones == FALLBACK_P5_BY_PHONES


class TestPhoneSetGuard:
    """A dictionary/model phone mismatch is the one failure MFA does not report.

    Aligning tone-suffixed phones against a tone-stripped model exits zero and
    writes TextGrids where every segment carries the same meaningless label --
    observed as an entire corpus aligned to the single phone 'm'.
    """

    def _model(self, tmp_path, phones):
        import zipfile
        path = tmp_path / "model.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "model/phones.txt",
                "\n".join(f"{p} {i}" for i, p in enumerate(phones)),
            )
        return path

    def _aligner(self, tmp_path, model_phones, dict_lines):
        from yoruba_engine.alignment.mfa import MFAAligner
        from yoruba_engine.config import EngineConfig

        config = EngineConfig()
        config.mfa.acoustic_model = self._model(tmp_path, model_phones)
        config.mfa.dictionary = tmp_path / "d.dict"
        config.mfa.dictionary.write_text("\n".join(dict_lines), encoding="utf-8")
        return MFAAligner(config)

    def test_matching_phone_sets_pass(self, tmp_path):
        aligner = self._aligner(
            tmp_path, ["o", "r", "i", "spn"], ["òrí\to r i", "[noise]\tspn"]
        )
        aligner.check_phone_sets_match()      # must not raise

    def test_tone_suffix_mismatch_is_caught(self, tmp_path):
        from yoruba_engine.exceptions import AlignmentError

        aligner = self._aligner(
            tmp_path, ["o", "r", "i", "spn"], ["òrí\to_L r i_H"]
        )
        with pytest.raises(AlignmentError, match="tone-stripped"):
            aligner.check_phone_sets_match()

    def test_probabilistic_dictionary_format_is_parsed(self, tmp_path):
        """MFA rewrites dictionaries with probability columns before the phones;
        those numbers must not be mistaken for phone labels."""
        aligner = self._aligner(
            tmp_path, ["o", "r", "i", "spn"], ["òrí\t0.99\t0.05\t2.42\t0.72\to r i"]
        )
        aligner.check_phone_sets_match()

    def test_unreadable_model_is_not_fatal(self, tmp_path):
        """MFA does its own checks; an unreadable archive must not block the run."""
        from yoruba_engine.alignment.mfa import MFAAligner
        from yoruba_engine.config import EngineConfig

        config = EngineConfig()
        config.mfa.acoustic_model = tmp_path / "not-a-zip.zip"
        config.mfa.acoustic_model.write_text("garbage", encoding="utf-8")
        config.mfa.dictionary = tmp_path / "d.dict"
        config.mfa.dictionary.write_text("òrí\to r i", encoding="utf-8")
        MFAAligner(config).check_phone_sets_match()
