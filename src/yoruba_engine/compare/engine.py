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


def phrase_resemblance(prompt: str, heard: str) -> float:
    """How much a recognised transcript sounds like the prompt, 0 to 1.

    This answers a narrower question than :func:`text similarity
    <ComparisonEngine>`: not "how many words were right" but "was this an
    attempt at this phrase at all". The two must not be confused, because a
    learner who mangles every word is still attempting the prompt and must be
    coached, not accused of reading something else.

    Word-level agreement cannot make that distinction. A genuine attempt at
    ``àbùbùtán bíi omi òkun`` came back from the recogniser as ``àbògùtàn omi
    ọkàn``: one word of four matches exactly, which scores 0.25 -- the same
    range a completely different phrase scores. Compared as characters with
    tone marks removed, the same pair scores 0.67, because the sounds are
    nearly all there in nearly the right order.

    Measured on the project's four attempt recordings against their prompt, and
    on the same recognitions against eight unrelated app phrases:

        genuine attempts   0.67 - 0.97
        different phrases  0.13 - 0.43

    That is a wide gap, but it comes from four recordings by one speaker, so
    treat the threshold as provisional rather than tuned.

    Tone marks are stripped because the recogniser's are unreliable -- it writes
    ``àbùbùtàn`` for ``àbùbùtán`` -- and a tone error is reported in its own
    section rather than being read as a different phrase. Spaces are dropped
    because word boundaries are exactly what a recogniser gets wrong when a
    learner runs words together.
    """
    from difflib import SequenceMatcher  # noqa: PLC0415

    from ..g2p.orthography import normalize  # noqa: PLC0415

    def key(text: str) -> str:
        return strip_tones(normalize(text or "")).replace(" ", "")

    left, right = key(prompt), key(heard)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _collapse_repeats(word: str) -> str:
    """Write each run of the same letter once: ``bii`` becomes ``bi``.

    Yoruba writes a doubled vowel where two tone-bearing units share a quality,
    and a recogniser will not reproduce that consistently -- Whisper returned
    ``bí`` for ``bíi`` and ``ọkàn`` correctly in the same breath. Treating the
    two spellings as different words reports a substitution the learner did not
    make.

    Only runs collapse. A substitution never does, which is what keeps this
    safe: ``okun`` and ``okan`` differ by one letter and stay different, and
    they are exactly the contrast the engine exists to catch. So does ``ẹ``
    against ``e``, since the sub-dot is part of the letter rather than a repeat.
    """
    out: list[str] = []
    for character in word:
        if not out or out[-1] != character:
            out.append(character)
    return "".join(out)


def align_words(reference: list[str], learner: list[str],
                tolerate_spelling: bool = False) -> WordAlignmentResult:
    """Align two word sequences with Levenshtein distance.

    Words are compared tone-insensitively here: a tonal difference is a *tone*
    error, not a word error, and counting it twice would double-penalize the
    learner.

    Args:
        reference: Reference word tokens.
        learner: Learner word tokens.
        tolerate_spelling: Also ignore repeated letters, for transcripts that
            came from a recogniser rather than from the prompt. Off by default,
            so prompted practice keeps exact matching.

    Returns:
        The alignment, with ``None`` marking insertions and deletions.
    """
    def key(word: str) -> str:
        stripped = strip_tones(word)
        if not tolerate_spelling:
            return stripped
        # With a recogniser, also ignore sub-dots. They are phonemic -- ọ is /ɔ/
        # and o is /o/ -- so this is a real concession, made because the error it
        # prevents is worse than the one it allows.
        #
        # A learner reading `ó dàbọ̀` was transcribed `ó dàbò` and scored 50% on
        # a two-word phrase, told they had said the wrong word. They had not:
        # they said the right word and produced one vowel as /o/, which the vowel
        # classifier confirms at 0.91 against 0.01 for /ɔ/. Word identity is not
        # the channel that error belongs in, and it is already reported by the
        # vowel panel, by CHECK VOWELS, and by the minimal-pair check.
        #
        # The cost is that a genuine minimal pair -- `òkun` against `ọkun` --
        # scores as the same word here. That pair differs by exactly the same
        # single sub-dot, so no rule can separate them at this level; the vowel
        # classifier is what distinguishes them, and it still does.
        import unicodedata  # noqa: PLC0415

        decomposed = unicodedata.normalize("NFD", stripped)
        without_dots = "".join(c for c in decomposed if not unicodedata.combining(c))
        return _collapse_repeats(unicodedata.normalize("NFC", without_dots))

    ref_keys = [key(w) for w in reference]
    learner_keys = [key(w) for w in learner]

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
        tolerate_spelling: bool = False,
        vowel_accuracy: float | None = None,
    ) -> ComparisonResult:
        """Produce the full scored comparison.

        Args:
            tolerate_spelling: Ignore repeated letters when matching words.
            vowel_accuracy: Share of the prompt's vowels the learner confidently
                produced, from :meth:`AssessmentPipeline._score_vowel_identity`.
                ``None`` leaves it out of the score rather than counting it zero.
                Set for transcripts from a recogniser, whose spelling of a
                correctly spoken word need not match the prompt's.
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

        alignment = align_words(reference_tokens, learner_tokens,
                                tolerate_spelling=tolerate_spelling)
        text_similarity = max(0.0, 1.0 - alignment.error_rate)

        # On a one-word prompt the word-level score has only two values, and a
        # recogniser out of its depth on half a second of audio swings it the
        # whole way. Two of the igba reference clips -- recordings a native
        # speaker confirms contain exactly the right word -- came back as `ti
        # gbà` and `tí gbà`, an inserted consonant and a split, and scored zero
        # text where an unmistakably wrong word would also score zero.
        #
        # Characters carry the distinction words cannot at this length: those
        # two resemble the prompt at 0.89, while the five igba words resemble
        # each other at most 0.80 once tones are stripped. So a single-word
        # prompt is scored on how much of it was heard rather than on whether
        # the recogniser spelled it exactly, which is the same question
        # phrase_resemblance already answers for verification.
        #
        # Longer prompts keep the word-level score. There the recogniser has
        # enough context to be trusted, and naming which word was wrong is the
        # feedback a learner needs.
        # Measured against the transcript, not against the aligned words. When a
        # one-word attempt is aligned to the prompt rather than to a mis-spelled
        # recognition, those two differ, and scoring the aligned words would
        # compare the prompt with itself and return 1.0 for anything that got
        # that far -- including a genuine substitution such as `dàbò` for
        # `dàbọ̀`, which is exactly the error this project most needs to keep
        # reporting. The transcript is what the recogniser actually heard.
        if len(reference_tokens) == 1:
            heard = learner.transcript or " ".join(learner_tokens)
            if heard:
                text_similarity = phrase_resemblance(reference_tokens[0], heard)

        tone_comparison = self.tone_analyzer.compare(
            reference_words, learner_words, word_pairs=alignment.pairs
        )
        tone_similarity = tone_comparison.accuracy

        # Vowel identity joins the score only when it could be judged. Folding
        # in a None as zero would punish a learner for a measurement that never
        # happened; dropping the weight and renormalising keeps a recording with
        # no usable pitch scoring exactly as it did before.
        components = [(self.scoring.text_weight, text_similarity),
                      (self.scoring.tone_weight, tone_similarity)]
        if vowel_accuracy is not None:
            components.append((self.scoring.vowel_weight, vowel_accuracy))
        total_weight = sum(weight for weight, _ in components)
        overall = (sum(weight * value for weight, value in components) / total_weight
                   if total_weight else 0.0)

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
            vowel_accuracy=vowel_accuracy,
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
