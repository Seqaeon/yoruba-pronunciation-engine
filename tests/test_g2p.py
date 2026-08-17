"""Tests for grapheme-to-phoneme conversion and dictionary generation."""

from __future__ import annotations

import pytest

from yoruba_engine.datatypes import Tone
from yoruba_engine.exceptions import G2PError
from yoruba_engine.g2p.dictionary import (
    NOISE_PHONE,
    PronunciationDictionary,
    grapheme_to_phonemes,
)
from yoruba_engine.g2p.orthography import NOISE_TOKEN


def symbols(word: str) -> list[str]:
    return [p.symbol for p in grapheme_to_phonemes(word)]


class TestGraphemeToPhonemes:
    def test_simple_word(self):
        assert symbols("òrí") == ["o_L", "r", "i_H"]

    def test_gb_is_a_single_phoneme(self):
        """gb is a labial-velar stop, not g followed by b."""
        assert symbols("gbogbo") == ["gb", "o_M", "gb", "o_M"]

    def test_dot_below_vowels_are_distinct(self):
        assert symbols("ọmọ") == ["O_M", "m", "O_M"]
        assert symbols("omo") == ["o_M", "m", "o_M"]

    def test_sh_digraph_maps_to_s_hook(self):
        assert symbols("she") == symbols("ṣe")

    def test_nasal_vowel_is_one_phone(self):
        phones = grapheme_to_phonemes("ìwọ̀n")
        assert phones[-1].is_nasal is True
        assert phones[-1].symbol == "On_L"
        assert sum(p.is_vowel for p in phones) == 2

    def test_n_before_vowel_stays_a_consonant(self):
        assert symbols("ọ̀nà") == ["O_L", "n", "a_L"]

    def test_tone_carried_on_vowel_phones(self):
        phones = grapheme_to_phonemes("ẹ̀kọ́")
        vowels = [p for p in phones if p.is_vowel]
        assert [v.tone for v in vowels] == [Tone.LOW, Tone.HIGH]

    def test_hyphen_is_not_a_phoneme(self):
        assert "-" not in symbols("ilé-ìwé")

    def test_empty_word_raises(self):
        with pytest.raises(G2PError):
            grapheme_to_phonemes("")

    def test_foreign_letter_raises(self):
        """c, q, v, x, z are not in the Yoruba alphabet and need an override."""
        with pytest.raises(G2PError, match="Unmappable"):
            grapheme_to_phonemes("computer")


class TestPronunciationDictionary:
    def test_builds_entries(self):
        dictionary = PronunciationDictionary()
        dictionary.build(["òrí", "ọmọ", "gbogbo"])
        assert dictionary.entries["òrí"] == ["o_L r i_H"]
        # Three words plus the always-present noise entry.
        assert len(dictionary.entries) == 4

    def test_noise_entry_is_always_defined(self):
        """Transcripts contain [noise]; MFA would call it OOV without an entry."""
        dictionary = PronunciationDictionary()
        dictionary.build(["ọmọ"])
        assert dictionary.entries[NOISE_TOKEN] == [NOISE_PHONE]

    def test_collects_failures_without_raising(self):
        dictionary = PronunciationDictionary()
        dictionary.build(["ọmọ", "xylophone"])
        assert "ọmọ" in dictionary.entries
        assert "xylophone" in dictionary.failures

    def test_overrides_take_precedence(self):
        dictionary = PronunciationDictionary(overrides={"ọmọ": ["custom phones"]})
        dictionary.build(["ọmọ"])
        assert dictionary.entries["ọmọ"] == ["custom phones"]

    def test_override_rescues_unmappable_word(self):
        dictionary = PronunciationDictionary(overrides={"computer": ["k o_M m p u_M t a_M"]})
        dictionary.build(["computer"])
        assert not dictionary.failures

    def test_phone_inventory(self):
        dictionary = PronunciationDictionary()
        dictionary.build(["òrí"])
        assert dictionary.phone_inventory() == {"o_L", "r", "i_H", NOISE_PHONE}

    def test_write_is_sorted_and_tab_separated(self, tmp_path):
        dictionary = PronunciationDictionary()
        dictionary.build(["ọmọ", "òrí"])
        path = tmp_path / "yoruba.dict"
        dictionary.write(path)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert all("\t" in line for line in lines)
        assert lines == sorted(lines)

    def test_load_overrides_roundtrip(self, tmp_path):
        override_file = tmp_path / "overrides.tsv"
        override_file.write_text(
            "# comment line\nọmọ\tO_M m O_M\nọmọ\tO_M m O_H\n", encoding="utf-8"
        )
        dictionary = PronunciationDictionary.load_overrides(override_file)
        assert len(dictionary.overrides["ọmọ"]) == 2

    def test_missing_override_file_is_not_an_error(self, tmp_path):
        dictionary = PronunciationDictionary.load_overrides(tmp_path / "nope.tsv")
        assert dictionary.overrides == {}


class TestVowelInventory:
    """Standard Yoruba has 12 vowels: 7 oral and 5 nasal.

    There is no *en or *on. A sequence that looks like one is either a syllabic
    nasal carrying its own tone, or a sub-dot dropped during typing.
    """

    def test_seven_oral_vowels(self):
        from yoruba_engine.g2p.orthography import ORAL_VOWELS
        assert len(ORAL_VOWELS) == 7
        assert set(ORAL_VOWELS) == {"a", "e", "ẹ", "i", "o", "ọ", "u"}

    def test_five_nasal_vowels(self):
        from yoruba_engine.g2p.orthography import NASAL_VOWELS
        assert len(NASAL_VOWELS) == 5
        assert set(NASAL_VOWELS) == {"an", "ẹn", "in", "ọn", "un"}

    @pytest.mark.parametrize(
        "word,expected_nasal",
        [("fún", "un_H"), ("ìwọ̀n", "On_L"), ("jẹun", "un_M")],
    )
    def test_legal_nasal_vowels_are_produced(self, word, expected_nasal):
        assert expected_nasal in symbols(word)

    def test_e_and_o_never_nasalize(self):
        """A nasalizing n after e or o means a dropped sub-dot, so the intended
        nasal vowel is recovered instead of inventing *en / *on."""
        assert "en_M" not in symbols("wen")
        assert "on_M" not in symbols("won")
        assert "On_M" in symbols("won")

    def test_tone_marked_n_is_a_syllabic_nasal(self):
        """oǹdó is o + ǹ + dó: the ǹ carries its own Low tone and must not be
        absorbed as nasalization, which would also lose a syllable."""
        phones = symbols("oǹdó")
        assert "n_L" in phones
        assert "on_M" not in phones

    def test_syllabic_nasal_counts_as_a_tone_bearing_unit(self):
        from yoruba_engine.g2p.orthography import tone_pattern
        assert len(tone_pattern("oǹdó")) == 3

    def test_n_before_a_vowel_is_still_a_consonant(self):
        assert symbols("ẹnu") == ["E_M", "n", "u_M"]
