"""Tests for Yoruba orthography handling.

These are the highest-value tests in the suite: every downstream stage depends on
tone marks being read correctly, and Unicode normalization bugs here are silent
-- they produce a plausible-looking Mid tone rather than an error.
"""

from __future__ import annotations

import unicodedata

import pytest

from yoruba_engine.datatypes import Tone
from yoruba_engine.g2p.orthography import (
    NOISE_TOKEN,
    decompose_char,
    grapheme_clusters,
    has_tone_marks,
    is_noise_token,
    normalize,
    strip_tones,
    tokenize,
    tone_pattern,
)


class TestNormalize:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize("  Mo   FẸ́   ") == normalize("mo fẹ́")

    def test_empty_input(self):
        assert normalize("") == ""

    def test_cedilla_folds_onto_dot_below(self):
        """Some input methods emit a cedilla where a dot below is meant."""
        with_cedilla = unicodedata.normalize("NFC", "ȩ")
        assert normalize(with_cedilla) == normalize("ẹ")

    def test_idempotent(self):
        once = normalize("Ẹ̀kọ́")
        assert normalize(once) == once


class TestGraphemeClusters:
    def test_keeps_stacked_marks_with_base(self):
        """ẹ̀ has no precomposed form; it must stay one cluster."""
        clusters = grapheme_clusters(normalize("ẹ̀"))
        assert len(clusters) == 1

    def test_round_trips(self):
        text = normalize("ọ̀rẹ́ mi")
        assert "".join(grapheme_clusters(text)) == text

    def test_plain_ascii(self):
        assert grapheme_clusters("abc") == ["a", "b", "c"]


class TestDecomposeChar:
    @pytest.mark.parametrize(
        "char,expected_tone",
        [("á", Tone.HIGH), ("à", Tone.LOW), ("a", Tone.MID), ("ā", Tone.MID)],
    )
    def test_tone_extraction(self, char, expected_tone):
        assert decompose_char(char).tone is expected_tone

    def test_dot_below_detected(self):
        assert decompose_char("ẹ").has_dot_below is True
        assert decompose_char("e").has_dot_below is False

    def test_dot_below_and_tone_together(self):
        result = decompose_char(normalize("ẹ̀"))
        assert result.has_dot_below is True
        assert result.tone is Tone.LOW
        assert result.letter == "ẹ"


class TestTonePattern:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("òrí", [Tone.LOW, Tone.HIGH]),
            ("ọmọ", [Tone.MID, Tone.MID]),
            ("ẹsẹ̀", [Tone.MID, Tone.LOW]),
            ("ẹ̀kọ́", [Tone.LOW, Tone.HIGH]),
            ("ìwé", [Tone.LOW, Tone.HIGH]),
            ("gbogbo", [Tone.MID, Tone.MID]),
        ],
    )
    def test_known_words(self, word, expected):
        assert tone_pattern(word) == expected

    def test_nasal_vowel_is_one_tone_bearing_unit(self):
        """The n in 'ìwọ̀n' nasalizes the vowel; it is not its own syllable."""
        assert len(tone_pattern("ìwọ̀n")) == 2

    def test_n_before_vowel_is_a_consonant(self):
        assert tone_pattern("ọ̀nà") == [Tone.LOW, Tone.LOW]

    def test_word_initial_syllabic_nasal_bears_tone(self):
        assert tone_pattern("ń") == [Tone.HIGH]


class TestStripTones:
    def test_removes_tone_keeps_dot_below(self):
        assert strip_tones("ẹ̀kọ́") == normalize("ẹkọ")

    def test_no_op_on_untoned_text(self):
        assert strip_tones("ọmọ") == normalize("ọmọ")


class TestHasToneMarks:
    def test_detects_marks(self):
        assert has_tone_marks("òrí") is True

    def test_undiacritized_text(self):
        """The corpus-blocker check: plain text has no tone information."""
        assert has_tone_marks("ori") is False

    def test_mid_only_word_reads_as_unmarked(self):
        """All-Mid words are legitimately unmarked, which is why the audit
        measures coverage across the corpus rather than per word."""
        assert has_tone_marks("ọmọ") is False


class TestTokenize:
    def test_strips_punctuation(self):
        assert tokenize("Ṣé o wà?  Bẹ́ẹ̀ ni.") == [
            normalize("ṣé"), "o", normalize("wà"), normalize("bẹ́ẹ̀"), "ni",
        ]

    def test_preserves_internal_hyphen(self):
        assert tokenize("ilé-ìwé") == [normalize("ilé-ìwé")]

    def test_empty(self):
        assert tokenize("") == []


class TestNoiseAnnotations:
    """The source corpora annotate non-speech events as [snap], [breath], etc.

    These must survive tokenization as a single canonical token: MFA needs to
    model the time they occupy, or it stretches the neighbouring vowel across
    them and corrupts the F0 measurement tone detection relies on.
    """

    @pytest.mark.parametrize(
        "marker", ["[snap]", "[breath]", "[external]", "[abrupt]", "[hesitation]"]
    )
    def test_markers_become_the_noise_token(self, marker):
        assert tokenize(f"{marker} mo wà") == [NOISE_TOKEN, "mo", normalize("wà")]

    def test_marker_at_end_of_line(self):
        assert tokenize("mo wà [external]") == ["mo", normalize("wà"), NOISE_TOKEN]

    def test_marker_fused_to_punctuation(self):
        assert tokenize("wà.[breath] mo") == [normalize("wà"), NOISE_TOKEN, "mo"]

    def test_keep_noise_false_drops_them(self):
        assert tokenize("[snap] mo wà", keep_noise=False) == ["mo", normalize("wà")]

    def test_real_words_are_untouched(self):
        assert NOISE_TOKEN not in tokenize("mo fẹ́ jẹun")

    def test_is_noise_token(self):
        assert is_noise_token(NOISE_TOKEN) is True
        assert is_noise_token("snap") is False

    def test_mfa_bracketed_label_counts_as_noise(self):
        """MFA rewrites bracketed dictionary entries to its own generic label,
        so alignment output never contains the token that went in."""
        assert is_noise_token("[bracketed]") is True
        assert is_noise_token("[snap]") is True

    def test_real_word_with_a_bracket_is_not_noise(self):
        assert is_noise_token("[") is False
        assert is_noise_token("ọmọ") is False

    def test_bracketed_marker_does_not_leak_as_a_word(self):
        """Naive punctuation stripping would turn [snap] into the word 'snap',
        which then needs a bogus Yoruba pronunciation."""
        assert "snap" not in tokenize("[snap] mo wà")
