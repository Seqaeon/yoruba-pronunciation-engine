"""Verify that a learner produced the vowels the prompt calls for.

This is what catches a *substitution* -- saying ``òkan`` when the prompt was
``òkun``. Forced alignment cannot: the word has the right shape and a plausible
duration, so it aligns without complaint, and duration-based detection only ever
finds omissions.

The check is deliberately conservative. It does not ask "which vowel was this?"
and compare to the expected one -- the classifier is 73.9% accurate, so that
would contradict the learner far too often. It asks instead how much probability
the classifier assigns to the vowel that *should* be there, and objects only when
that is very low. Measured on correctly-produced corpus vowels, only 5.2% fall
below 0.05, so a single flagged vowel is uncommon and two in one word is rare.

Telling a learner they said the wrong word when they did not is worse than
missing a real error, so the thresholds are set to favour silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import SegmentalConfig
from ..logging_utils import get_logger
from ..datatypes import WordAnalysis
from .classifier import vowel_features
from .formants import FormantExtractor, normalize_vowel_space
from .mfcc import MFCCExtractor, cepstral_mean_variance, normalize_vector

logger = get_logger("segmental.verifier")


@dataclass
class VowelCheck:
    """One vowel's verification result."""

    word_index: int
    vowel_index: int
    expected: str
    predicted: str
    expected_probability: float

    #: Threshold used for the ``suspect`` convenience property. The verifier
    #: itself reads the configured value; this is for reporting only.
    suspect_threshold: float = 0.02

    @property
    def suspect(self) -> bool:
        """Whether the expected vowel was assigned implausibly little probability."""
        return self.expected_probability < self.suspect_threshold


@dataclass
class SegmentalResult:
    """Verification across a whole utterance."""

    checks: list[VowelCheck] = field(default_factory=list)
    mispronounced: list[int] = field(default_factory=list)
    measured_vowels: int = 0

    #: Full probability vector per measured vowel, grouped by word index and
    #: kept in speaking order. The verifier itself only reads P(expected), but
    #: :class:`~yoruba_engine.segmental.decoder.WordDecoder` needs the whole
    #: distribution to weigh competing words, and extracting formants and MFCCs
    #: is the slow part of an assessment -- doing it twice would double it.
    distributions: dict[int, list] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Fraction of measured vowels that were not suspect."""
        if not self.checks:
            return 1.0
        return sum(not c.suspect for c in self.checks) / len(self.checks)


class VowelVerifier:
    """Checks aligned vowels against what the prompt requires."""

    def __init__(self, classifier, config: SegmentalConfig | None = None) -> None:
        """
        Args:
            classifier: A trained
                :class:`~yoruba_engine.segmental.classifier.VowelClassifier`.
            config: Thresholds; defaults to :class:`SegmentalConfig`.
        """
        self.classifier = classifier
        self.config = config or SegmentalConfig()
        self.formants = FormantExtractor()
        self.mfcc = MFCCExtractor()
        self._label_index = {label: i for i, label in enumerate(classifier.labels)}

    def verify(self, audio_path: Path, words: list[WordAnalysis]) -> SegmentalResult:
        """Check every measurable vowel in the utterance.

        Args:
            audio_path: The learner's recording.
            words: Word analyses from pitch extraction, whose vowel labels carry
                the phone the *prompt* requires.

        Returns:
            The verification result. On any measurement failure it returns an
            empty result rather than raising: an unverifiable recording must not
            be reported as a wrong one.
        """
        result = SegmentalResult()

        try:
            formant_track = self.formants.analyse(audio_path)
            times, frames = self.mfcc.analyse(audio_path)
        except Exception as exc:
            logger.info("Segmental check unavailable for %s: %s", audio_path.name, exc)
            return result

        # Collect measurements first: both normalizations are relative to the
        # speaker's own range across the whole utterance.
        entries = []
        for word_index, word in enumerate(words):
            for vowel_index, vowel in enumerate(word.vowels):
                formant_values = self.formants.measure_segment(
                    formant_track, vowel.start_s, vowel.end_s
                )
                cepstral_values = self.mfcc.measure_segment(
                    times, frames, vowel.start_s, vowel.end_s
                )
                if formant_values and cepstral_values:
                    entries.append(
                        (word_index, vowel_index, vowel.label, formant_values, cepstral_values)
                    )

        if not entries:
            return result

        speaker_f1, speaker_f2 = normalize_vowel_space([e[3] for e in entries])

        # Cepstral normalization is deliberately NOT applied to a learner
        # recording.
        #
        # It removes vocal tract length and channel response, but only when the
        # mean is taken over enough of one speaker's material that it does not
        # track the vowels themselves. Measured on the training corpus, sweeping
        # the group size on a single speaker so only sample count varies:
        #
        #   vowels in group    2      3      5     10     20     40     80  none
        #   macro F1        0.227  0.282  0.300  0.342  0.362  0.390  0.375 0.342
        #
        # One learner utterance yields a handful of vowels, which lands in the
        # range where normalizing is worse than doing nothing: on a two-vowel
        # word the mean IS the vowel, so subtracting it deletes the identity
        # being measured. The current classifier is trained to match -- per
        # speaker only where that speaker has at least 40 vowels, nothing
        # otherwise -- so training and inference agree.
        #
        # Models trained before this was understood used per-utterance mean and
        # variance. They say so in their metadata and still get it: feeding an
        # estimator features normalized differently from its training set costs
        # far more than the normalization was ever worth.
        if self.classifier.metadata.get("cepstral_normalization") == "per_utterance":
            means, deviations = cepstral_mean_variance([e[4] for e in entries])
        else:
            width = len(entries[0][4])
            means, deviations = [0.0] * width, [1.0] * width

        rows = [
            vowel_features(formant_values, speaker_f1, speaker_f2)
            + normalize_vector(cepstral_values, means, deviations)
            for _, _, _, formant_values, cepstral_values in entries
        ]
        probabilities = self.classifier.model.predict_proba(rows)

        for (word_index, _, _, _, _), distribution in zip(entries, probabilities):
            result.distributions.setdefault(word_index, []).append(distribution)

        for (word_index, vowel_index, label, _, _), distribution in zip(entries, probabilities):
            expected = label.split("_")[0]
            position = self._label_index.get(expected)
            if position is None:
                continue          # a phone the classifier never saw; no opinion

            best = max(range(len(distribution)), key=distribution.__getitem__)
            result.checks.append(VowelCheck(
                word_index=word_index,
                vowel_index=vowel_index,
                expected=expected,
                predicted=self.classifier.labels[best],
                expected_probability=float(distribution[position]),
            ))

        result.measured_vowels = len(result.checks)
        result.mispronounced = self._flag_words(result.checks, words)
        return result

    def _flag_words(self, checks: list[VowelCheck], words: list[WordAnalysis]) -> list[int]:
        """Decide which words to query.

        A word is queried when any single vowel falls below the strict
        threshold, or when two vowels are weakly suspect.

        An earlier version demanded two suspect vowels in every case. That was
        wrong for the exact situation this check exists to catch: minimal pairs
        such as ``òkun``/``òkan`` differ in one segment by definition, so the
        rule could never fire on them.
        """
        by_word: dict[int, list[VowelCheck]] = {}
        for check in checks:
            by_word.setdefault(check.word_index, []).append(check)

        flagged = []
        for word_index, word_checks in by_word.items():
            strong = [c for c in word_checks
                      if c.expected_probability < self.config.suspect_probability]
            weak = [c for c in word_checks
                    if c.expected_probability < self.config.weak_probability]

            if strong or len(weak) >= self.config.min_weak_vowels:
                flagged.append(word_index)

        if flagged:
            logger.info(
                "Segmental check flagged %d words: %s",
                len(flagged), [words[i].word for i in flagged if i < len(words)],
            )
        return flagged
