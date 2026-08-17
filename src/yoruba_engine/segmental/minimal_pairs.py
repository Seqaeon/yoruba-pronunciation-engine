"""Minimal-pair discrimination.

Thresholding the probability of the expected vowel asks a hard open-set
question: "is this vowel implausible?" But in prompted practice the real
question is far easier and far more useful: "did the learner say ``òkun`` or
``òkan``?" -- a choice between two specific words.

That distinction matters because the alternatives are known in advance. A
classifier only 66% accurate at naming a vowel out of twelve can still be
decisive at telling ``un`` from ``an``, because the comparison is a likelihood
ratio between two hypotheses rather than a threshold on one.

It also produces feedback worth reading. Instead of "check this word", the
engine can say *"you said òkàn (heart), the prompt was òkun (ocean)"* -- which
is the difference that actually matters to a learner, since a single vowel
changes the meaning entirely.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..logging_utils import get_logger

logger = get_logger("segmental.minimal_pairs")

#: The vowel phones a word can differ in. Consonant minimal pairs exist too, but
#: this check is driven by a vowel classifier.
VOWELS = frozenset({"a", "e", "E", "i", "o", "O", "u", "an", "En", "in", "On", "un"})


@dataclass(frozen=True)
class Competitor:
    """A real word differing from the prompt word in one vowel."""

    word: str
    position: int          # index among the word's vowels
    expected_phone: str    # what the prompt requires there
    competing_phone: str   # what this competitor has instead


class MinimalPairIndex:
    """Finds the real words a prompt word could be confused with.

    Built from the pronunciation dictionary, so competitors are always genuine
    Yoruba words rather than hypothetical strings. Confusing ``òkun`` with a
    non-word is not an error a learner makes; confusing it with ``òkàn`` is.
    """

    def __init__(self) -> None:
        self.by_pattern: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        self.vowels_of: dict[str, list[str]] = {}
        self.phones_of: dict[str, list[str]] = {}

    @classmethod
    def from_dictionary(cls, path: Path) -> MinimalPairIndex:
        """Build the index from an MFA pronunciation dictionary.

        Handles MFA's probabilistic format, where numeric columns precede the
        phones.
        """
        import re  # noqa: PLC0415

        numeric = re.compile(r"^-?\d+(\.\d+)?$")
        index = cls()

        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                word = fields[0]
                phones: list[str] = []
                for field in fields[1:]:
                    if not numeric.match(field.strip()):
                        phones.extend(field.split())
                if not phones:
                    continue

                vowels = [p for p in phones if p in VOWELS]
                if not vowels:
                    continue
                index.vowels_of.setdefault(word, vowels)
                index.phones_of.setdefault(word, phones)

                # Wildcard each vowel in turn; words sharing a pattern differ in
                # exactly that position.
                for position in range(len(vowels)):
                    wildcard = _nth_vowel_index(phones, position)
                    pattern = tuple(
                        "*" if i == wildcard else p for i, p in enumerate(phones)
                    )
                    index.by_pattern[pattern].append((word, vowels[position]))

        logger.info("Minimal-pair index: %d words, %d patterns",
                    len(index.vowels_of), len(index.by_pattern))
        return index

    def competitors(self, word: str, phones: list[str] | None = None) -> list[Competitor]:
        """Real words differing from ``word`` in exactly one vowel.

        Args:
            word: The prompt word.
            phones: Its full phone sequence. Defaults to the dictionary's, which
                is normally what you want -- the patterns were built from full
                sequences, so passing only the vowels finds nothing.
        """
        if phones is None:
            phones = self.phones_of.get(word)
        if not phones:
            return []

        vowels = [p for p in phones if p in VOWELS]
        found: list[Competitor] = []

        for position in range(len(vowels)):
            pattern = tuple(
                "*" if i == _nth_vowel_index(phones, position) else p
                for i, p in enumerate(phones)
            )
            for other, other_vowel in self.by_pattern.get(pattern, []):
                if other == word or other_vowel == vowels[position]:
                    continue
                found.append(Competitor(
                    word=other,
                    position=position,
                    expected_phone=vowels[position],
                    competing_phone=other_vowel,
                ))
        return found


def _nth_vowel_index(phones: list[str], n: int) -> int:
    """Index in ``phones`` of the n-th vowel."""
    seen = -1
    for index, phone in enumerate(phones):
        if phone in VOWELS:
            seen += 1
            if seen == n:
                return index
    return -1


@dataclass
class PairVerdict:
    """The outcome of comparing a prompt word against its competitors."""

    prompt_word: str
    competitor: str
    expected_phone: str
    competing_phone: str
    log_ratio: float

    @property
    def decisive(self) -> bool:
        """Whether the competitor is enough more likely to be worth reporting.

        A log ratio of 2.3 is roughly a 10x likelihood difference.
        """
        return self.log_ratio >= 2.3

    def describe(self) -> str:
        return (
            f"'{self.prompt_word}' sounded like '{self.competitor}' "
            f"({self.competing_phone} rather than {self.expected_phone})"
        )


def best_competitor(
    prompt_word: str,
    competitors: list[Competitor],
    vowel_probabilities: list[dict[str, float]],
    label_index: dict[str, int],
) -> PairVerdict | None:
    """Return the competitor the audio supports most, if any beats the prompt.

    Args:
        prompt_word: The word the learner was asked to say.
        competitors: Its minimal-pair competitors.
        vowel_probabilities: One probability distribution per vowel of the word,
            as ``{phone: probability}``.
        label_index: Classifier label positions, used to skip phones the model
            does not know.

    Returns:
        The strongest verdict, or ``None`` when the prompt word is the best
        explanation of the audio.
    """
    best: PairVerdict | None = None

    for competitor in competitors:
        if competitor.position >= len(vowel_probabilities):
            continue
        distribution = vowel_probabilities[competitor.position]
        expected = distribution.get(competitor.expected_phone)
        competing = distribution.get(competitor.competing_phone)
        if not expected or not competing:
            continue
        if competitor.expected_phone not in label_index or competitor.competing_phone not in label_index:
            continue

        ratio = math.log(competing) - math.log(expected)
        if ratio > 0 and (best is None or ratio > best.log_ratio):
            best = PairVerdict(
                prompt_word=prompt_word,
                competitor=competitor.word,
                expected_phone=competitor.expected_phone,
                competing_phone=competitor.competing_phone,
                log_ratio=ratio,
            )

    return best
