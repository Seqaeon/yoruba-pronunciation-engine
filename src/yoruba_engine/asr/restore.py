"""Restore tone diacritics to recognizer output.

Yoruba ASR models are trained on undiacritized text and return words like
``ori`` where the corpus writes ``òrí``. That is a problem for two reasons:

* The pronunciation dictionary is keyed on diacritized words, so an
  undiacritized transcript is entirely out of vocabulary and cannot be aligned.
* Tone comparison needs the aligner to place vowel boundaries; without an
  alignment there is nothing to measure.

Restoration here is deliberately shallow: map each recognized word to the
diacritized corpus word with the same letters, preferring the most frequent when
several exist. It resolves the majority of tokens because most Yoruba words have
one common diacritization, and it needs no model.

It is *not* a tone-marking oracle. Genuine ambiguity exists -- ``owo`` is
``owó`` (money), ``ọwọ́`` (hand) and ``ọ̀wọ̀`` (respect) -- and the fallback picks
the most frequent, which is sometimes wrong. That limitation is why the engine
measures tone **acoustically** from the audio and never trusts the restored
spelling for tone: restoration exists only to make alignment possible.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from ..logging_utils import get_logger
from ..g2p.orthography import normalize, strip_tones, tokenize

logger = get_logger("asr.restore")


class DiacriticRestorer:
    """Maps undiacritized words onto the corpus's diacritized vocabulary."""

    def __init__(self, candidates: dict[str, list[tuple[str, int]]] | None = None) -> None:
        """
        Args:
            candidates: Mapping of tone-stripped key to ``(word, frequency)``
                pairs, most frequent first.
        """
        self.candidates = candidates or {}

    @classmethod
    def from_word_frequency(cls, path: Path) -> DiacriticRestorer:
        """Build from the ``word_frequency.csv`` written by the Stage 1 audit.

        Raises:
            FileNotFoundError: If the frequency file is missing. Run
                ``yoruba-engine audit`` to produce it.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `yoruba-engine audit` to generate it."
            )

        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                word = normalize(row.get("word", ""))
                if not word:
                    continue
                try:
                    frequency = int(row.get("frequency", 0))
                except ValueError:
                    frequency = 0
                grouped[strip_tones(word)][word] += frequency

        candidates = {
            key: counter.most_common() for key, counter in grouped.items()
        }
        logger.info("Diacritic restorer loaded %d word forms from %s", len(candidates), path)
        return cls(candidates=candidates)

    def restore_word(self, word: str) -> tuple[str, bool]:
        """Restore one word.

        Args:
            word: A recognized word, with or without diacritics.

        Returns:
            ``(restored_word, was_resolved)``. When the word is not in the corpus
            vocabulary it is returned unchanged with ``False``, because inventing
            a spelling would produce an out-of-vocabulary token that breaks
            alignment.
        """
        normalized = normalize(word)
        if not normalized:
            return normalized, False

        key = strip_tones(normalized)
        options = self.candidates.get(key)
        if not options:
            return normalized, False

        # An already-diacritized form that exists in the corpus is kept as-is.
        for candidate, _ in options:
            if candidate == normalized:
                return candidate, True

        best, _ = options[0]
        if len(options) > 1:
            logger.debug(
                "Ambiguous restoration for %r: chose %r from %s",
                normalized, best, [c for c, _ in options[:4]],
            )
        return best, True

    def restore(self, text: str) -> tuple[str, list[str]]:
        """Restore a whole transcript.

        Args:
            text: Recognizer output.

        Returns:
            ``(restored_text, unresolved_words)``. Unresolved words are returned
            so the caller can decide what to do -- typically drop them from the
            text used for alignment, since they cannot be pronounced by the
            dictionary.
        """
        restored: list[str] = []
        unresolved: list[str] = []

        for token in tokenize(text, keep_noise=False):
            word, resolved = self.restore_word(token)
            restored.append(word)
            if not resolved:
                unresolved.append(word)

        if unresolved:
            logger.info(
                "%d/%d recognized words are not in the corpus vocabulary: %s",
                len(unresolved), len(restored), unresolved[:8],
            )
        return " ".join(restored), unresolved

    def alignable_text(self, text: str) -> str:
        """Restore, then drop words the dictionary cannot pronounce.

        The result is safe to hand to the forced aligner. Dropping a word costs
        the alignment of that word only; leaving it in makes MFA fail on the
        whole utterance.
        """
        restored: list[str] = []
        for token in tokenize(text, keep_noise=False):
            word, resolved = self.restore_word(token)
            if resolved:
                restored.append(word)
        return " ".join(restored)
