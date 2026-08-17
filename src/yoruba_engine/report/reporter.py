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

        for ref_word, learner_word in result.word_pairs:
            shown = learner_word
            if ref_word is None:
                status = "INSERTED"
            elif learner_word is None:
                # Only duration evidence says a word was absent. A word the
                # vowel check flagged is present but wrong, and saying it was
                # missing tells the learner to repeat something they did say.
                status = "NOT SAID"
            elif ref_word in result.substitutions:
                shown = result.substitutions[ref_word]
                status = "SAID DIFFERENTLY"
            elif ref_word in result.mispronounced_words:
                # Showing the prompt word here would claim the learner produced
                # it, next to a warning saying they may not have. We know a word
                # was spoken in that slot but not which one.
                shown = "(unclear)"
                status = "CHECK VOWELS"
            elif ref_word == learner_word:
                status = "ok"
            else:
                status = "DIFFERENT"
            lines.append(f"{ref_word or '-':<24} {shown or '-':<24} {status}")

        if reference is not None and learner is not None:
            lines.extend(["", "TONE PATTERNS", "-" * 62])
            lines.append(f"{'WORD':<20} {'EXPECTED':<20} {'DETECTED':<20}")
            for ref_word, learner_word in _speech_pairs(reference, learner, result):
                detected = (
                    learner_word.pattern_string() if learner_word is not None
                    else "(not detected)"
                )
                # The target is what the word is SPELLED with, not what the
                # detector made of the reference recording. Showing the latter
                # was actively misleading: àbùbùtán is spelled Low-Low-Low-High
                # and the column read "Low-Mid-?-?", so a learner comparing the
                # two columns was being shown the detector's own failure as
                # though it were the goal.
                expected = _expected_pattern(ref_word)
                lines.append(f"{ref_word.word:<20} {expected:<20} {detected:<20}")

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
        unspoken = {w for w in result.undetected_words}
        flagged_by_gop = [w for w in result.poorly_pronounced if w not in unspoken]

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
            },
            "scores": {
                "text_similarity": (
                    round(result.text_similarity, 4)
                    if result.text_similarity_applicable else None
                ),
                "tone_similarity": round(result.tone_similarity, 4),
                "overall": round(result.overall_score, 4),
            },
            "word_alignment": [
                {"reference": ref, "learner": learner_word}
                for ref, learner_word in result.word_pairs
            ],
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
            payload["tone_patterns"] = [
                {
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
                }
                for ref_word, learner_word in _speech_pairs(reference, learner, result)
            ]

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
