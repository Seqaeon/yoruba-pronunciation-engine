"""Stage 6: the comparison and scoring engine.

Aligns the learner's word sequence to the reference using edit distance, scores
segmental and tonal accuracy separately, then combines them.

Text and tone are scored separately and reported separately because they are
pedagogically distinct failures. Saying the wrong word is a vocabulary problem;
saying the right word with the wrong tone is a phonological one -- and for a
tonal language the second is the harder skill and the more common error for
speakers whose first language is not tonal.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ScoringConfig, ToneConfig
from ..exceptions import ComparisonError
from ..g2p.orthography import is_noise_token, strip_tones
from ..logging_utils import get_logger
from ..tone.analyzer import ToneAnalyzer
from ..datatypes import ComparisonResult, Tone, UtteranceAnalysis

logger = get_logger("compare")


@dataclass
class WordAlignmentResult:
    """Edit-distance alignment between two word sequences."""

    pairs: list[tuple[int | None, int | None]]
    substitutions: int
    insertions: int
    deletions: int
    matches: int

    @property
    def total_reference(self) -> int:
        return self.matches + self.substitutions + self.deletions

    @property
    def error_rate(self) -> float:
        """Word error rate against the reference length."""
        if self.total_reference == 0:
            return 0.0
        return (self.substitutions + self.insertions + self.deletions) / self.total_reference


def align_words(reference: list[str], learner: list[str]) -> WordAlignmentResult:
    """Align two word sequences with Levenshtein distance.

    Words are compared tone-insensitively here: a tonal difference is a *tone*
    error, not a word error, and counting it twice would double-penalize the
    learner.

    Args:
        reference: Reference word tokens.
        learner: Learner word tokens.

    Returns:
        The alignment, with ``None`` marking insertions and deletions.
    """
    ref_keys = [strip_tones(w) for w in reference]
    learner_keys = [strip_tones(w) for w in learner]

    rows, cols = len(ref_keys) + 1, len(learner_keys) + 1
    cost = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        cost[i][0] = i
    for j in range(cols):
        cost[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution = cost[i - 1][j - 1] + (ref_keys[i - 1] != learner_keys[j - 1])
            cost[i][j] = min(substitution, cost[i - 1][j] + 1, cost[i][j - 1] + 1)

    pairs: list[tuple[int | None, int | None]] = []
    matches = substitutions = insertions = deletions = 0
    i, j = len(ref_keys), len(learner_keys)

    # Ties are common and the choice between them matters. Aligning
    #
    #     reference  abubutan  bii  omi  okun
    #     learner    abubutan  omi  n    kun
    #
    # costs three either way: pair every position and substitute three times, or
    # drop ``bii``, match ``omi`` against ``omi`` and substitute around it. The
    # second is the true account -- the learner omitted a word -- and it finds
    # two exact matches where the first finds one.
    #
    # A traceback that takes the diagonal first always picks the positional
    # reading, which is harmless while the transcript is the prompt echoed back
    # and both sequences match by construction, and wrong as soon as a
    # recogniser returns what was really said. Taking the diagonal only when it
    # is an exact match, and otherwise letting a gap absorb the cost, prefers
    # the reading with more matches at no extra cost.
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            same = ref_keys[i - 1] == learner_keys[j - 1]
            if same and cost[i][j] == cost[i - 1][j - 1]:
                pairs.append((i - 1, j - 1))
                matches += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            pairs.append((i - 1, None))
            deletions += 1
            i -= 1
            continue
        if j > 0 and cost[i][j] == cost[i][j - 1] + 1:
            pairs.append((None, j - 1))
            insertions += 1
            j -= 1
            continue
        # Nothing cheaper remains, so this is a genuine substitution.
        pairs.append((i - 1, j - 1))
        substitutions += 1
        i, j = i - 1, j - 1

    pairs.reverse()
    return WordAlignmentResult(
        pairs=pairs,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        matches=matches,
    )


class ComparisonEngine:
    """Scores a learner rendition against a native-speaker reference."""

    def __init__(
        self,
        scoring: ScoringConfig | None = None,
        tone_config: ToneConfig | None = None,
    ) -> None:
        self.scoring = scoring or ScoringConfig()
        self.tone_analyzer = ToneAnalyzer(tone_config)

    def compare(
        self,
        reference: UtteranceAnalysis,
        learner: UtteranceAnalysis,
    ) -> ComparisonResult:
        """Produce the full scored comparison.

        Args:
            reference: Analysed native-speaker rendition.
            learner: Analysed learner rendition.

        Returns:
            Scores, word alignment, tone errors, and generated feedback.

        Raises:
            ComparisonError: If the reference has no analysable words.
        """
        if not reference.words:
            raise ComparisonError(
                f"Reference utterance {reference.utterance_id} has no aligned words; "
                "cannot compare against it."
            )

        # Non-speech events are annotated in the reference corpus but are never
        # part of what a learner is asked to say. Scoring them would penalize the
        # learner for not reproducing the reference speaker's breathing.
        reference_words = [w for w in reference.words if not is_noise_token(w.word)]
        learner_words = [w for w in learner.words if not is_noise_token(w.word)]

        if not reference_words:
            raise ComparisonError(
                f"Reference utterance {reference.utterance_id} contains only "
                "non-speech annotations; cannot compare against it."
            )

        reference_tokens = [w.word for w in reference_words]
        learner_tokens = [w.word for w in learner_words]

        alignment = align_words(reference_tokens, learner_tokens)
        text_similarity = max(0.0, 1.0 - alignment.error_rate)

        tone_comparison = self.tone_analyzer.compare(
            reference_words, learner_words, word_pairs=alignment.pairs
        )
        tone_similarity = tone_comparison.accuracy

        overall = (
            self.scoring.text_weight * text_similarity
            + self.scoring.tone_weight * tone_similarity
        )

        word_pairs = [
            (
                reference_tokens[i] if i is not None else None,
                learner_tokens[j] if j is not None else None,
            )
            for i, j in alignment.pairs
        ]

        result = ComparisonResult(
            utterance_id=reference.utterance_id,
            reference_transcript=reference.transcript,
            learner_transcript=learner.transcript,
            text_similarity=text_similarity,
            tone_similarity=tone_similarity,
            overall_score=overall,
            word_pairs=word_pairs,
            tone_errors=tone_comparison.errors,
        )
        result.feedback = self._build_feedback(result, alignment, tone_comparison)

        logger.info(
            "Compared %s: text %.2f, tone %.2f, overall %.2f",
            reference.utterance_id, text_similarity, tone_similarity, overall,
        )
        return result

    def _build_feedback(
        self,
        result: ComparisonResult,
        alignment: WordAlignmentResult,
        tone_comparison,
    ) -> list[str]:
        """Turn scores and errors into learner-facing guidance.

        Ordered most-actionable first, and capped so a struggling beginner is not
        handed twenty simultaneous corrections.
        """
        feedback: list[str] = []

        if result.overall_score >= 0.9:
            feedback.append("Excellent. Your pronunciation closely matches the reference.")
        elif result.overall_score >= self.scoring.pass_threshold:
            feedback.append("Good attempt. A few details to refine below.")
        else:
            feedback.append("Keep practising. Listen to the reference again and repeat slowly.")

        if alignment.substitutions:
            wrong = [
                f"'{learner}' instead of '{ref}'"
                for ref, learner in result.word_pairs
                if ref and learner and strip_tones(ref) != strip_tones(learner)
            ]
            if wrong:
                feedback.append(f"Different words: {', '.join(wrong[:3])}.")
        if alignment.deletions:
            missed = [ref for ref, learner in result.word_pairs if ref and learner is None]
            feedback.append(
                f"These words were not detected in your recording: {', '.join(missed[:4])}. "
                "Say each one clearly and try again."
            )
        if alignment.insertions:
            extra = [learner for ref, learner in result.word_pairs if learner and ref is None]
            feedback.append(f"Extra words detected: {', '.join(extra[:3])}.")

        for error in result.tone_errors[:3]:
            feedback.append(error.describe())

        if len(result.tone_errors) > 3:
            feedback.append(f"...and {len(result.tone_errors) - 3} further tone differences.")

        if tone_comparison.total == 0:
            feedback.append(
                "Tone could not be measured. Record in a quieter place and speak "
                "at a steady volume."
            )
        elif tone_comparison.accuracy < 0.6:
            feedback.append(
                "Focus on tone rather than the consonants and vowels: in Yoruba, "
                "pitch changes the meaning of the word."
            )

        return feedback

    @staticmethod
    def describe_pattern_mismatch(
        word: str,
        expected: list[Tone],
        detected: list[Tone],
    ) -> str:
        """Format a word-level tone comparison in the brief's example style.

        Produces output of the form::

            Expected: Low-High
            Detected: High-High
        """
        names = {Tone.HIGH: "High", Tone.MID: "Mid", Tone.LOW: "Low", Tone.UNKNOWN: "?"}
        expected_str = "-".join(names[t] for t in expected)
        detected_str = "-".join(names[t] for t in detected)
        lines = [f"Word: {word}", f"Expected: {expected_str}", f"Detected: {detected_str}"]

        for index, (exp, det) in enumerate(zip(expected, detected)):
            if exp != det:
                ordinal = ["first", "second", "third", "fourth"]
                which = ordinal[index] if index < len(ordinal) else f"{index + 1}th"
                lines.append(f"Tone mismatch on {which} vowel.")
        return "\n".join(lines)
