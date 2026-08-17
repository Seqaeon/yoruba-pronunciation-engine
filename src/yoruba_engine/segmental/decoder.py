"""Derive what the learner actually said, rather than assuming the prompt.

Forced alignment answers "where in this audio is each word of the prompt", not
"which word was spoken". Asked to align ``òkun`` against a recording of
``ọkàn``, MFA obliges: it stretches the prompt over the audio and reports word
boundaries with no complaint. Read naively, that makes every attempt look
correct, and a substitution shows up only as an oddly short word -- reported as
an omission, which is the wrong feedback for a learner who said a real word.

This module closes that gap without a speech recogniser. The prompt bounds the
search: a learner reading ``òkun`` produces ``òkun`` or something confusable
with it, so the candidate set is every real Yoruba word sharing its consonant
skeleton -- ``ọkàn``, ``ọkun``, ``òkè``, ``ẹkún`` and the rest of ``* k *``.
Scoring those against the vowel classifier's per-vowel distributions gives both
a decoded word and a margin.

Note the skeleton, not :class:`MinimalPairIndex` alone: that index finds words
differing in exactly one vowel, and ``òkun`` (o, un) against ``ọkàn`` (ọ, an)
differs in two. The pair this engine was built to catch is not a minimal pair.

**The margin is the point.** Measured on SLR86 held-out speakers, the ``o``/``ọ``
contrast is right 77.6% of the time, so a decoder that always names its best
guess would tell a learner "you said ọkun" wrongly about one time in four.
Being told you made a mistake you did not make is worse feedback than silence,
so a verdict below the calibrated margin is withheld rather than guessed --
see :meth:`WordDecoder.decode`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..logging_utils import get_logger
from .minimal_pairs import VOWELS, MinimalPairIndex

logger = get_logger("decoder")

#: Log-odds a candidate must beat the prompt word by before the decoder will
#: claim a substitution.
#:
#: Calibrated by ``scripts/calibrate_decoder.py`` against 2,801 words of aligned
#: corpus speech. Every one was read correctly by a native speaker, so any
#: substitution claimed there is a false alarm:
#:
#:     margin   0.0    1.0    2.0    2.5    3.0    4.0    5.0
#:     wrong   31.2%  16.7%   7.4%   4.8%   3.2%   1.3%   0.7%
#:
#: A decoder that always names its best guess (margin 0) contradicts a third of
#: correctly spoken words, which is why abstention is not optional here.
#:
#: 3.0 is chosen rather than the 4.4%-at-2.5 that a 5% target would pick,
#: because real substitutions in the recorded test attempts won by 4.0 and 4.5
#: log units -- comfortably clear of 3.0 -- so the extra strictness costs
#: little recall and halves the false-alarm rate. Telling a learner they made a
#: mistake they did not make is the expensive error.
DEFAULT_MARGIN = 3.0

#: Weight on the unigram prior, in the same log units as the acoustic score.
#:
#: Acoustics alone let the decoder name words that do not occur in the language
#: as used: a learner who said ``ọkàn`` was decoded as ``ọ̀kà``, which appears
#: zero times in 209,265 corpus tokens, while ``ọkàn`` appears 202. Adding
#: ``PRIOR_WEIGHT * log(count + 1)`` breaks ties toward words people say.
#:
#: 0.5 puts the gap between a 200-occurrence word and a 28-occurrence one at
#: about 1.0 log units -- enough to decide a close call, not enough to override
#: clear acoustic evidence, which is the intended balance.
PRIOR_WEIGHT = 0.5


@dataclass
class WordVerdict:
    """What the decoder concluded about one word of the prompt."""

    word: str
    #: The best-scoring candidate, which may be the prompt word itself.
    decoded: str
    #: ``decoded`` score minus the prompt word's score, in log units. Zero when
    #: the prompt word won.
    margin: float
    #: Whether the margin cleared the threshold. A verdict that is not
    #: confident must be reported as "unclear", never as a substitution.
    confident: bool
    #: Every candidate considered, best first, for debugging and reports.
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def substituted(self) -> bool:
        """Whether the learner said a different real word, confidently."""
        return self.confident and self.decoded != self.word


class WordDecoder:
    """Scores minimal-pair candidates against measured vowel probabilities."""

    def __init__(
        self,
        classifier,
        index: MinimalPairIndex,
        margin: float = DEFAULT_MARGIN,
        frequencies: dict[str, int] | None = None,
        prior_weight: float = PRIOR_WEIGHT,
    ) -> None:
        """
        Args:
            classifier: A loaded :class:`VowelClassifier`.
            index: Minimal-pair index built from the pronunciation dictionary.
            margin: Log-odds a competitor must win by to be reported.
            frequencies: Word counts for the unigram prior. Without them the
                decoder scores on acoustics alone and will happily name words
                that never occur.
            prior_weight: How much the prior counts against the acoustics.
        """
        self.classifier = classifier
        self.index = index
        self.margin = margin
        self.frequencies = frequencies or {}
        self.prior_weight = prior_weight
        self._position = {label: i for i, label in enumerate(classifier.labels)}

        # MinimalPairIndex finds words differing in exactly ONE vowel, which is
        # too narrow here: òkun (o, un) and ọkàn (ọ, an) differ in two, and that
        # is precisely the confusion this engine exists to catch. Group instead
        # by consonant skeleton -- every vowel wildcarded -- so any real word
        # reachable by getting the vowels wrong is a candidate.
        self._by_skeleton: dict[tuple, list[str]] = defaultdict(list)
        for candidate, phones in index.phones_of.items():
            skeleton = tuple("*" if phone in VOWELS else phone for phone in phones)
            self._by_skeleton[skeleton].append(candidate)

    @classmethod
    def load(cls, dictionary: Path, classifier=None, margin: float = DEFAULT_MARGIN,
             frequencies: Path | None = None):
        """Build a decoder from the shipped dictionary, classifier and counts."""
        import json  # noqa: PLC0415

        from .classifier import VowelClassifier  # noqa: PLC0415

        counts = {}
        # decoder.py lives at src/yoruba_engine/segmental/, so the project root
        # is four levels up; the dictionary path is not a reliable anchor.
        path = frequencies or Path(__file__).resolve().parents[3] / "models" / "word_frequency.json"
        try:
            counts = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.info("No word frequencies at %s (%s); decoding on acoustics "
                        "alone, which names rare words too readily", path, exc)

        return cls(
            classifier=classifier or VowelClassifier.load(),
            index=MinimalPairIndex.from_dictionary(dictionary),
            margin=margin,
            frequencies=counts,
        )

    def prior(self, word: str) -> float:
        """Log prior for a candidate, zero when no counts are loaded."""
        if not self.frequencies:
            return 0.0
        return self.prior_weight * math.log(self.frequencies.get(word, 0) + 1)

    def _score(self, vowels: list[str], distributions: list) -> float | None:
        """Total log probability of a candidate's vowel sequence.

        Returns ``None`` when the candidate has a different number of vowels
        than the aligner found, which makes the two incomparable.
        """
        if len(vowels) != len(distributions):
            return None
        total = 0.0
        for vowel, distribution in zip(vowels, distributions):
            position = self._position.get(vowel)
            if position is None:
                return None          # a phone the classifier never saw
            total += math.log(max(float(distribution[position]), 1e-9))
        return total

    def decode(self, word: str, distributions: list) -> WordVerdict:
        """Decide which real word the measured vowels best support.

        Args:
            word: The prompt word, normalized.
            distributions: One probability vector per vowel the aligner placed
                inside the word, in order.

        Returns:
            A verdict. When no candidate is comparable, or when the winning
            margin is below the threshold, ``confident`` is ``False`` and the
            caller must not report a substitution.
        """
        expected_vowels = self.index.vowels_of.get(word)
        if not expected_vowels or not distributions:
            return WordVerdict(word, word, 0.0, confident=False)

        candidates = {word: expected_vowels}
        phones = self.index.phones_of.get(word) or []
        skeleton = tuple("*" if phone in VOWELS else phone for phone in phones)
        rivals = {c.word for c in self.index.competitors(word)}
        rivals.update(self._by_skeleton.get(skeleton, ()))
        for rival in rivals:
            vowels = self.index.vowels_of.get(rival)
            if vowels:
                candidates[rival] = vowels

        scores = {}
        for candidate, vowels in candidates.items():
            score = self._score(vowels, distributions)
            if score is not None:
                scores[candidate] = score + self.prior(candidate)

        own = scores.get(word)
        if own is None:
            # The aligner's vowel count disagrees with the dictionary, usually
            # because the word was not really spoken. Not a substitution.
            return WordVerdict(word, word, 0.0, confident=False, scores=scores)

        best = max(scores, key=scores.__getitem__)
        margin = scores[best] - own
        return WordVerdict(
            word=word,
            decoded=best,
            margin=margin,
            confident=(best == word) or (margin >= self.margin),
            scores=dict(sorted(scores.items(), key=lambda item: -item[1])),
        )

    def transcript(self, verdicts: list[WordVerdict]) -> str:
        """Assemble the learner's transcript from per-word verdicts.

        Words the decoder is not confident about keep the prompt's spelling
        rather than a guess, so the transcript never asserts an error that was
        not measured. Substitutions it *is* confident about are written as the
        word the learner actually produced.
        """
        return " ".join(v.decoded if v.confident else v.word for v in verdicts)
