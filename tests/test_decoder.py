"""Tests for minimal-pair word decoding.

These cover the judgement calls rather than the arithmetic: when the decoder
must stay silent, when a candidate set is wide enough to contain the right
answer, and whether the frequency prior can be outvoted by acoustics.
"""

from __future__ import annotations

import pytest

from yoruba_engine.segmental.decoder import WordDecoder, WordVerdict
from yoruba_engine.segmental.minimal_pairs import MinimalPairIndex

LABELS = ["a", "an", "e", "i", "o", "O", "u", "un"]


class FakeModel:
    """Stands in for the estimator; the decoder only reads ``labels``."""


class FakeClassifier:
    def __init__(self):
        self.labels = LABELS
        self.model = FakeModel()
        self.metadata = {}


def distribution(**probabilities):
    """Build a probability vector over LABELS from named entries."""
    row = [0.001] * len(LABELS)
    for label, value in probabilities.items():
        row[LABELS.index(label)] = value
    return row


def build_index(entries):
    """Build a MinimalPairIndex from ``{word: [phones]}`` without a file."""
    index = MinimalPairIndex()
    for word, phones in entries.items():
        vowels = [p for p in phones if p in set(LABELS)]
        index.vowels_of[word] = vowels
        index.phones_of[word] = phones
    return index


WORDS = {
    "òkun": ["o", "k", "un"],
    "ọkàn": ["O", "k", "an"],
    "ọkun": ["O", "k", "un"],
    "ọkà":  ["O", "k", "a"],
    "omi":  ["o", "m", "i"],
}


def make_decoder(margin=2.0, frequencies=None, prior_weight=0.5):
    return WordDecoder(FakeClassifier(), build_index(WORDS), margin=margin,
                       frequencies=frequencies or {}, prior_weight=prior_weight)


class TestCandidateSet:
    def test_includes_words_differing_in_two_vowels(self):
        """òkun and ọkàn differ in *both* vowels, so they are not a minimal pair.

        The pair this engine exists to catch would be invisible to a strict
        one-vowel index, which is why the decoder groups by consonant skeleton.
        """
        decoder = make_decoder()
        verdict = decoder.decode("òkun", [distribution(O=0.9), distribution(an=0.9)])
        assert "ọkàn" in verdict.scores

    def test_excludes_words_with_a_different_skeleton(self):
        decoder = make_decoder()
        verdict = decoder.decode("òkun", [distribution(o=0.9), distribution(un=0.9)])
        assert "omi" not in verdict.scores


class TestAbstention:
    def test_names_a_substitution_when_the_margin_is_clear(self):
        decoder = make_decoder(margin=2.0)
        verdict = decoder.decode(
            "òkun", [distribution(O=0.95, o=0.01), distribution(an=0.95, un=0.01)])
        assert verdict.decoded == "ọkàn"
        assert verdict.substituted

    def test_stays_silent_when_the_margin_is_thin(self):
        """A near-tie must not be reported.

        Measured on corpus speech, a decoder that always names its best guess
        contradicts a fifth of correctly spoken words.
        """
        decoder = make_decoder(margin=2.0)
        verdict = decoder.decode(
            "òkun", [distribution(O=0.32, o=0.30), distribution(un=0.31, an=0.30)])
        assert not verdict.substituted

    def test_prompt_word_winning_is_always_confident(self):
        decoder = make_decoder()
        verdict = decoder.decode(
            "òkun", [distribution(o=0.9), distribution(un=0.9)])
        assert verdict.decoded == "òkun"
        assert verdict.confident and not verdict.substituted

    def test_unknown_word_gives_no_opinion(self):
        decoder = make_decoder()
        verdict = decoder.decode("àbùbùtu", [distribution(a=0.9)])
        assert not verdict.confident
        assert not verdict.substituted

    def test_no_distributions_gives_no_opinion(self):
        assert not make_decoder().decode("òkun", []).confident

    def test_vowel_count_mismatch_is_not_a_substitution(self):
        """Three distributions against a two-vowel word means the word was not
        really spoken -- an omission, which is not the decoder's business."""
        decoder = make_decoder()
        verdict = decoder.decode(
            "òkun", [distribution(o=0.9), distribution(un=0.9), distribution(a=0.9)])
        assert not verdict.substituted


class TestFrequencyPrior:
    def test_prior_breaks_a_tie_toward_the_commoner_word(self):
        """ọkàn occurs 202 times in the corpus; ọkà 28. Acoustically identical
        vowel sequences should resolve to the word people actually say."""
        counts = {"ọkàn": 202, "ọkà": 28}
        decoder = make_decoder(margin=0.0, frequencies=counts)
        # 'an' and 'a' given equal probability: only the prior separates them.
        verdict = decoder.decode(
            "òkun", [distribution(O=0.9), distribution(an=0.5, a=0.5)])
        assert verdict.decoded == "ọkàn"

    def test_prior_penalises_words_that_never_occur(self):
        """ọkun appears zero times in 209,265 tokens, and was being named."""
        counts = {"òkun": 51, "ọkun": 0}
        decoder = make_decoder(margin=0.0, frequencies=counts)
        assert decoder.prior("ọkun") == pytest.approx(0.0)
        assert decoder.prior("òkun") > 0

    def test_prior_cannot_override_clear_acoustics(self):
        """A rare word with strong acoustic support must still win."""
        counts = {"òkun": 5000, "ọkàn": 1}
        decoder = make_decoder(margin=2.0, frequencies=counts)
        verdict = decoder.decode(
            "òkun", [distribution(O=0.99, o=1e-6), distribution(an=0.99, un=1e-6)])
        assert verdict.decoded == "ọkàn"

    def test_no_frequencies_means_no_prior(self):
        assert make_decoder(frequencies={}).prior("òkun") == 0.0


class TestTranscript:
    def test_uses_the_prompt_where_not_confident(self):
        """An unconfident verdict must never assert an error that was not
        measured."""
        decoder = make_decoder()
        verdicts = [
            WordVerdict("òkun", "ọkàn", margin=0.1, confident=False),
            WordVerdict("omi", "omi", margin=0.0, confident=True),
        ]
        assert decoder.transcript(verdicts) == "òkun omi"

    def test_uses_the_decoded_word_where_confident(self):
        decoder = make_decoder()
        verdicts = [WordVerdict("òkun", "ọkàn", margin=4.0, confident=True)]
        assert decoder.transcript(verdicts) == "ọkàn"
