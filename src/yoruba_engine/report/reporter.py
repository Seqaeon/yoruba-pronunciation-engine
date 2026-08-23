"""Stage 7: report generation.

Emits the same assessment as plain text (for CLI use), JSON (for the future
FastAPI layer and the mobile client), and optionally HTML.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..g2p.orthography import is_noise_token
from ..datatypes import ComparisonResult, Tone, UtteranceAnalysis

logger = get_logger("report")

_TONE_NAMES = {Tone.HIGH: "High", Tone.MID: "Mid", Tone.LOW: "Low", Tone.UNKNOWN: "?"}


def _serialize(obj: Any) -> Any:
    """JSON encoder hook for dataclasses, enums, and paths."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Tone):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize {type(obj).__name__}")


def _same_word(reference: str | None, learner: str | None) -> bool:
    """Whether two spellings name the same word, ignoring tone and repeats.

    Matches the comparison the scorer performs, so the report cannot mark a
    word DIFFERENT that the score counted as correct. Substitutions still
    differ: ``òkun`` and ``ọkun`` collapse to ``okun`` and ``ọkun``, which are
    not equal, because the sub-dot belongs to the letter rather than being a
    repeat.
    """
    if not reference or not learner:
        return False
    from ..compare.engine import _collapse_repeats  # noqa: PLC0415
    from ..g2p.orthography import strip_tones  # noqa: PLC0415

    return (_collapse_repeats(strip_tones(reference))
            == _collapse_repeats(strip_tones(learner)))


def _expected_pattern(word) -> str:
    """The tones a word should carry, from its spelling.

    Falls back to whatever was detected in the reference recording only when the
    prompt has no usable marks, which happens with an undiacritized corpus.
    """
    from ..g2p.orthography import normalize, tone_pattern  # noqa: PLC0415
    from ..datatypes import Tone  # noqa: PLC0415

    written = tone_pattern(normalize(word.word))
    if len(written) == len(word.vowels) and all(t is not Tone.UNKNOWN for t in written):
        return "-".join(_TONE_NAMES[t] for t in written)
    return word.pattern_string()


def _diacritics_only(reference: str | None, learner: str | None) -> bool:
    """Whether two spellings differ only in tone marks and sub-dots.

    ``dàbọ̀`` and ``dàbò`` differ by one sub-dot; ``òkun`` and ``ọkàn`` differ by
    a whole vowel. The first is the same word produced with the wrong vowel
    quality, the second is a different word, and a report that calls both
    DIFFERENT is telling a learner the wrong thing half the time.

    This does not excuse the error -- the vowel and tone panels still report it.
    It decides which of the two mistakes the learner is told they made.
    """
    if not reference or not learner:
        return False
    import unicodedata  # noqa: PLC0415

    from ..compare.engine import _collapse_repeats  # noqa: PLC0415
    from ..g2p.orthography import normalize, strip_tones  # noqa: PLC0415

    def bare(word: str) -> str:
        decomposed = unicodedata.normalize("NFD", strip_tones(normalize(word)))
        return _collapse_repeats(
            "".join(c for c in decomposed if not unicodedata.combining(c)).lower())

    return reference != learner and bare(reference) == bare(learner)


def alignment_rows(result: ComparisonResult) -> list[dict[str, str | None]]:
    """The word-alignment table, as rows of reference, learner and status.

    Both the text report and the JSON payload render this table, and they were
    computing it separately: the report derived a status column while the
    payload emitted bare word pairs, leaving any consumer of the JSON -- the
    app's practice screen among them -- to reimplement the rules and get a
    different answer. Deriving both from one function is what keeps the screen
    and the report describing the same assessment.

    ``learner`` is what to show in that column, which is not always the aligned
    token: a substituted word shows what the learner actually said, and a word
    the vowel check flagged shows ``(unclear)``, because printing the prompt
    word there would claim the learner produced it.
    """
    # A word the acoustic model fitted badly must not print as "ok" -- that is
    # exactly the case the vowel check cannot see. Words the duration check says
    # were never produced are excluded: they are reported as NOT SAID.
    unspoken = set(result.undetected_words)
    flagged_by_gop = [w for w in result.poorly_pronounced if w not in unspoken]

    # The vowel check names words on the side that was aligned. In prompted
    # mode that is the prompt's own words; with a recogniser it is the words the
    # recogniser heard, because those are what MFA was given. Testing only the
    # reference side meant the check's verdict never reached this table in the
    # mode the app now runs in -- it silently never matched.
    flagged = set(result.mispronounced_words) | set(flagged_by_gop)

    def suspect(reference_word, learner_word):
        return reference_word in flagged or learner_word in flagged

    rows: list[dict[str, str | None]] = []
    for ref_word, learner_word in result.word_pairs:
        shown = learner_word
        if ref_word is None:
            status = "INSERTED"
        elif learner_word is None:
            # Only duration evidence says a word was absent. A word the vowel
            # check flagged is present but wrong, and saying it was missing
            # tells the learner to repeat something they did say.
            status = "NOT SAID"
        elif ref_word in result.substitutions:
            shown = result.substitutions[ref_word]
            status = "SAID DIFFERENTLY"
        elif suspect(ref_word, learner_word):
            # The vowels in this slot do not match what was aligned there. What
            # to show depends on whether anything independent named it.
            #
            # With a recogniser, the learner word is a real hypothesis about
            # what was said -- 'àbùbùtùn' for a prompted 'àbùbùtán' -- and the
            # status already says not to trust it fully. Replacing it with
            # "(unclear)" throws away the only concrete thing the learner can
            # act on, which is what it was doing.
            #
            # Without one, the learner word is the prompt echoed back, and
            # printing it would assert the learner produced the very word the
            # vowels contradict. That is the case "(unclear)" exists for.
            # Whether a name exists depends on the mode, not on whether the
            # name happens to equal the prompt.
            #
            # With a recogniser the learner word is an independent hypothesis
            # about what was said, and it is worth showing even when it matches
            # the prompt exactly -- "you said ẹ, but check its vowel" is a
            # different and more useful message than "(unclear)". Testing word
            # equality instead blanked precisely the words the recogniser got
            # right, which is how ``ẹ`` came back as unclear while the transcript
            # underneath read ``... ni ẹ``.
            #
            # Without one, the learner word is the prompt echoed back, and
            # printing it would assert the learner produced the very word the
            # vowels contradict. That is what "(unclear)" is for.
            shown = learner_word if result.text_similarity_applicable else "(unclear)"
            status = "CHECK VOWELS"
        elif ref_word == learner_word:
            status = "ok"
        elif _same_word(ref_word, learner_word):
            # The scorer already counted these as the same word, and the status
            # column must not contradict it. A recogniser writes ``bí`` where
            # the prompt has ``bíi``; the learner said the word, and only the
            # spelling of a repeated vowel differs. Tone marks differ freely
            # here too -- a tonal error is reported in its own section rather
            # than as a wrong word.
            status = "ok"
        elif _diacritics_only(ref_word, learner_word):
            # The same word with a different vowel or tone, not a different
            # word. `dàbọ̀` heard as `dàbò` is one sub-dot: /ɔ/ produced as /o/.
            # Calling that DIFFERENT tells a learner they said the wrong word,
            # which is both false and the wrong correction -- they said the
            # right word and shaped one vowel wrong, which is what CHECK VOWELS
            # means and what the tone and vowel panels then explain.
            status = "CHECK VOWELS"
        else:
            # Two independent signals agree: the recogniser wrote a different
            # word, and the vowel check did not object to it -- the audio in
            # this slot is consistent with what was heard. One signal alone is
            # not enough to tell a learner they said the wrong word, because a
            # recogniser gets 82% of words right on native studio speech, and
            # the 18% would all land here as accusations.
            status = "DIFFERENT"
        rows.append({"reference": ref_word, "learner": shown, "status": status})
    return rows


def tone_pattern_rows(
    reference: UtteranceAnalysis,
    learner: UtteranceAnalysis,
    result: ComparisonResult | None = None,
) -> list[dict]:
    """The tone-pattern table: one row per prompt word.

    ``expected`` is what the word is *spelled* with, not what a detector made of
    a reference recording. Showing the latter was actively misleading --
    ``àbùbùtán`` is spelled Low-Low-Low-High and the column read ``Low-Mid-?-?``,
    so a learner comparing the two columns was shown the detector's own failure
    as though it were the goal.
    """
    rows = []
    for ref_word, learner_word in _speech_pairs(reference, learner, result):
        rows.append({
            "word": ref_word.word,
            "expected": _expected_pattern(ref_word).split("-"),
            "detected": (
                [_TONE_NAMES[t] for t in learner_word.tone_pattern]
                if learner_word is not None else None
            ),
            "confidences": (
                [round(v.confidence, 3) for v in learner_word.vowels]
                if learner_word is not None else []
            ),
        })
    return rows


def _speech_pairs(
    reference: UtteranceAnalysis,
    learner: UtteranceAnalysis,
    result: ComparisonResult | None = None,
) -> list[tuple[object, object | None]]:
    """Pair reference words with the learner words they aligned to.

    Excludes non-speech annotations: the corpus marks events like ``[snap]``
    inline and the aligner emits them as ``[bracketed]``. The aligner must model
    them, but they were never something a learner was asked to say.

    When ``result`` is given, pairing follows its edit-distance alignment rather
    than zipping the two lists positionally. That matters as soon as a word goes
    undetected: the lists then differ in length and a naive zip silently shifts
    every subsequent row, so the report would attribute one word's tones to
    another.
    """
    ref_words = [w for w in reference.words if not is_noise_token(w.word)]
    learner_words = [w for w in learner.words if not is_noise_token(w.word)]

    if result is None or not result.word_pairs:
        return list(zip(ref_words, learner_words))

    pairs: list[tuple[object, object | None]] = []
    ref_index = learner_index = 0
    for ref_token, learner_token in result.word_pairs:
        reference_word = None
        learner_word = None
        if ref_token is not None and ref_index < len(ref_words):
            reference_word = ref_words[ref_index]
            ref_index += 1
        if learner_token is not None and learner_index < len(learner_words):
            learner_word = learner_words[learner_index]
            learner_index += 1
        if reference_word is not None:
            pairs.append((reference_word, learner_word))
    return pairs


class ReportGenerator:
    """Renders a :class:`ComparisonResult` in several formats."""

    def to_text(
        self,
        result: ComparisonResult,
        reference: UtteranceAnalysis | None = None,
        learner: UtteranceAnalysis | None = None,
    ) -> str:
        """Render a human-readable plain-text report."""
        lines = [
            "YORUBA PRONUNCIATION ASSESSMENT",
            "=" * 62,
            f"Utterance : {result.utterance_id}",
            f"Generated : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "TRANSCRIPTS",
            "-" * 62,
            f"Reference : {result.reference_transcript}",
            (f"Learner   : {result.learner_transcript}"
             if result.text_similarity_applicable
             else f"Prompt    : {result.learner_transcript}  "
                  "(not a recognition of the learner's speech)"),
            "",
        ]

        if not result.verified:
            lines.extend([
                "!! UNVERIFIED ATTEMPT",
                "-" * 62,
                result.verification_note or
                "Could not confirm the learner said the target text.",
                "The scores below describe a forced alignment, not a real match.",
                "",
            ])

        # Which words the acoustic model fitted badly, for the sections below.
        # ``alignment_rows`` derives this itself; the WORDS TO CHECK section
        # further down still needs it here.
        unspoken = {w for w in result.undetected_words}
        flagged_by_gop = [w for w in result.poorly_pronounced if w not in unspoken]

        lines.extend([
            "SCORES",
            "-" * 62,
            f"Text similarity  : {result.text_similarity * 100:5.1f}%  {self._bar(result.text_similarity)}"
            if result.text_similarity_applicable
            else "Text similarity  :   n/a",
            f"Tone similarity  : {result.tone_similarity * 100:5.1f}%  {self._bar(result.tone_similarity)}",
            f"Overall score    : {result.overall_score * 100:5.1f}%  {self._bar(result.overall_score)}",
            "",
            "WORD ALIGNMENT",
            "-" * 62,
            f"{'REFERENCE':<24} {'LEARNER':<24} {'STATUS'}",
        ])

        for row in alignment_rows(result):
            lines.append(
                f"{row['reference'] or '-':<24} {row['learner'] or '-':<24} {row['status']}"
            )

        if reference is not None and learner is not None:
            lines.extend(["", "TONE PATTERNS", "-" * 62])
            lines.append(f"{'WORD':<20} {'EXPECTED':<20} {'DETECTED':<20}")
            for row in tone_pattern_rows(reference, learner, result):
                detected = (
                    "-".join(row["detected"]) if row["detected"] else "(not detected)"
                )
                expected = "-".join(row["expected"])
                lines.append(f"{row['word']:<20} {expected:<20} {detected:<20}")

        # Goodness-of-Pronunciation is a diagnostic signal, not feedback. Two
        # reasons it is not shown as its own section:
        #
        # "The acoustic model fitted these words badly, -3.2 standard deviations
        # below native production" tells a learner nothing they can act on; it
        # describes the engine, not their speech.
        #
        # And it contradicted the rest of the report. A word the duration check
        # says was never produced would still appear here as "poorly produced",
        # which reads as a comment on speech that did not happen.
        #
        # The score stays in the JSON payload for tuning and is folded into the
        # words-to-check list, which is phrased for a learner.

        if result.substitutions:
            lines.extend([
                "WORDS SAID DIFFERENTLY",
                "-" * 62,
                "The vowels measured here match a different real Yoruba word.",
            ])
            for prompted, said in result.substitutions.items():
                lines.append(f"  you said {said!r} where the prompt was {prompted!r}")
            lines.append("")

        to_check = list(dict.fromkeys(list(result.mispronounced_words) + flagged_by_gop))
        if to_check:
            lines.extend([
                "", "WORDS TO CHECK", "-" * 62,
                "These did not sound like the prompt. The check is indicative,",
                "not certain -- a small share of correctly spoken words are",
                "queried by it, so treat them as worth a second try.",
                "  " + ", ".join(to_check),
            ])

        if result.undetected_words:
            lines.extend([
                "", "WORDS NOT DETECTED", "-" * 62,
                "These prompt words could not be found in the recording:",
                "  " + ", ".join(result.undetected_words),
            ])

        lines.extend(["", "TONE ERRORS", "-" * 62])
        if result.tone_errors:
            for error in result.tone_errors:
                lines.append(f"  - {error.describe()}")
        else:
            lines.append("  None detected.")

        lines.extend(["", "FEEDBACK", "-" * 62])
        for item in result.feedback:
            lines.append(f"  - {item}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _bar(value: float, width: int = 20) -> str:
        """Render a score as an ASCII progress bar."""
        filled = int(round(max(0.0, min(1.0, value)) * width))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    def to_dict(
        self,
        result: ComparisonResult,
        reference: UtteranceAnalysis | None = None,
        learner: UtteranceAnalysis | None = None,
    ) -> dict[str, Any]:
        """Build a JSON-ready dict. This is the intended API response shape."""
        payload: dict[str, Any] = {
            "utterance_id": result.utterance_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "transcripts": {
                "reference": result.reference_transcript,
                # In prompted practice this is the prompt echoed back, not a
                # recognition of what the learner actually said.
                "learner": result.learner_transcript,
                "learner_is_prompt_echo": not result.text_similarity_applicable,
            },
            "verified": result.verified,
            "verification": {
                "passed": result.verified,
                "note": result.verification_note,
                "learner_speech_coverage": (
                    round(result.learner_speech_coverage, 4)
                    if result.learner_speech_coverage is not None else None
                ),
                # Which of the two decided the outcome depends on whether a
                # recogniser ran; both are reported so a reader can see why.
                "phrase_resemblance": (
                    round(result.learner_phrase_resemblance, 4)
                    if result.learner_phrase_resemblance is not None else None
                ),
            },
            "scores": {
                "text_similarity": (
                    round(result.text_similarity, 4)
                    if result.text_similarity_applicable else None
                ),
                "tone_similarity": round(result.tone_similarity, 4),
                "overall": round(result.overall_score, 4),
            },
            # Rows of reference, learner and status -- the same table the text
            # report prints. The payload used to emit bare word pairs, leaving
            # every consumer to reimplement the status rules; the app's practice
            # screen did, and got a different answer.
            "word_alignment": alignment_rows(result),
            "tone_errors": [
                {
                    "word": error.word,
                    "word_index": error.word_index,
                    "vowel_index": error.vowel_index,
                    "expected": _TONE_NAMES[error.expected],
                    "detected": _TONE_NAMES[error.detected],
                    "message": error.describe(),
                }
                for error in result.tone_errors
            ],
            "mispronounced_words": result.mispronounced_words,
            "substitutions": result.substitutions,
            "poorly_pronounced": result.poorly_pronounced,
            "undetected_words": result.undetected_words,
            "feedback": result.feedback,
        }

        if reference is not None and learner is not None:
            payload["tone_patterns"] = tone_pattern_rows(reference, learner, result)

        return payload

    def write_json(self, result: ComparisonResult, path: Path, **kwargs: Any) -> Path:
        """Write the JSON report, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(result, **kwargs), ensure_ascii=False, indent=2, default=_serialize),
            encoding="utf-8",
        )
        logger.info("Wrote JSON report to %s", path)
        return path

    def write_text(self, result: ComparisonResult, path: Path, **kwargs: Any) -> Path:
        """Write the plain-text report, creating parent directories."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_text(result, **kwargs), encoding="utf-8")
        logger.info("Wrote text report to %s", path)
        return path
