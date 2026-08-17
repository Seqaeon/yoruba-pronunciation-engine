"""Stage 1: dataset audit.

Reads ``line_index.tsv``, checks that every transcript has audio, measures total
duration, extracts the vocabulary, and reports on diacritic coverage.

The diacritic check is the most consequential output of this stage. If the
corpus lacks tone marks, Stages 2-7 cannot do tone work at all, and that must be
discovered before any modelling effort is spent.
"""

from __future__ import annotations

import csv
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import EngineConfig
from ..exceptions import DatasetError
from ..logging_utils import get_logger
from ..g2p.orthography import has_tone_marks, is_noise_token, normalize, tokenize
from ..datatypes import Utterance

logger = get_logger("audit")


@dataclass
class AuditReport:
    """Aggregated findings from a dataset audit."""

    total_lines: int = 0
    matched: list[Utterance] = field(default_factory=list)
    missing_audio: list[tuple[str, str]] = field(default_factory=list)
    unreadable_audio: list[tuple[str, str]] = field(default_factory=list)
    empty_transcripts: list[str] = field(default_factory=list)
    total_duration_s: float = 0.0
    word_frequency: Counter[str] = field(default_factory=Counter)
    lines_with_tone_marks: int = 0
    speakers: Counter[str] = field(default_factory=Counter)
    #: Count of bracketed non-speech annotations ([snap], [breath], ...).
    noise_tokens: int = 0

    @property
    def vocabulary(self) -> list[str]:
        """Unique words, sorted alphabetically."""
        return sorted(self.word_frequency)

    @property
    def total_duration_hours(self) -> float:
        return self.total_duration_s / 3600.0

    @property
    def tone_mark_coverage(self) -> float:
        """Fraction of non-empty transcripts carrying at least one tone mark."""
        usable = self.total_lines - len(self.empty_transcripts)
        return self.lines_with_tone_marks / usable if usable else 0.0

    @property
    def is_trainable(self) -> bool:
        """Rough go/no-go for training a speaker-independent acoustic model.

        Three hours is a pragmatic floor for MFA's GMM-HMM pipeline. Below that,
        train speaker-adapted models instead and expect noisier boundaries.
        """
        return self.total_duration_hours >= 3.0 and len(self.matched) >= 500


class DatasetAuditor:
    """Validates a transcript/audio corpus and emits the Stage 1 deliverables."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def audit(self) -> AuditReport:
        """Run the full audit.

        Returns:
            The populated report.

        Raises:
            DatasetError: If ``line_index.tsv`` is absent or has no usable rows.
        """
        index_path = self.config.line_index
        if not index_path.exists():
            raise DatasetError(
                f"line_index.tsv not found at {index_path}. "
                "Point EngineConfig.line_index at your corpus index."
            )

        report = AuditReport()
        for utt_id, transcript, speaker in self._read_index(index_path):
            report.total_lines += 1

            if not transcript.strip():
                report.empty_transcripts.append(utt_id)
                continue

            report.speakers[speaker] += 1
            if has_tone_marks(transcript):
                report.lines_with_tone_marks += 1

            for token in tokenize(transcript, keep_noise=True):
                if is_noise_token(token):
                    report.noise_tokens += 1
                else:
                    report.word_frequency[token] += 1

            audio_path = self._find_audio(utt_id)
            if audio_path is None:
                report.missing_audio.append((utt_id, transcript))
                continue

            duration = self._duration_of(audio_path)
            if duration is None:
                report.unreadable_audio.append((utt_id, str(audio_path)))
                continue

            report.total_duration_s += duration
            report.matched.append(
                Utterance(
                    utterance_id=utt_id,
                    audio_path=audio_path,
                    transcript=normalize(transcript),
                    speaker_id=speaker,
                    duration_s=duration,
                )
            )

        if report.total_lines == 0:
            raise DatasetError(f"{index_path} contained no rows")

        self._log_summary(report)
        return report

    def _read_index(self, path: Path) -> list[tuple[str, str, str]]:
        """Parse the index file into (utterance_id, transcript, speaker) rows.

        Accepts the two-column ``id<TAB>transcript`` layout used by the OpenSLR
        Yoruba sets, and a three-column variant with an explicit speaker id. When
        the speaker is absent it is inferred from the id prefix, which is the
        convention in those corpora (``yo_ng_female_00123``).
        """
        rows: list[tuple[str, str, str]] = []
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for line_no, parts in enumerate(reader, start=1):
                parts = [p.strip() for p in parts if p.strip() != ""]
                if not parts:
                    continue
                if len(parts) == 1:
                    logger.warning("%s:%d has no transcript column", path, line_no)
                    continue

                utt_id = parts[0]
                if len(parts) >= 3:
                    speaker, transcript = parts[1], parts[2]
                else:
                    transcript = parts[1]
                    speaker = self._infer_speaker(utt_id)
                rows.append((utt_id, transcript, speaker))
        return rows

    @staticmethod
    def _infer_speaker(utterance_id: str) -> str:
        """Derive a speaker id from an utterance id, falling back to 'unknown'."""
        parts = utterance_id.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else "unknown"

    def _find_audio(self, utterance_id: str) -> Path | None:
        """Locate the audio file for an utterance across supported extensions."""
        for extension in self.config.audio_extensions:
            candidate = self.config.audio_dir / f"{utterance_id}{extension}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _duration_of(path: Path) -> float | None:
        """Return duration in seconds, or ``None`` if the file cannot be read.

        Uses the stdlib ``wave`` module for WAV files to avoid a hard dependency
        on soundfile/librosa during the audit stage, and falls back to soundfile
        for other formats when it is installed.
        """
        if path.suffix.lower() == ".wav":
            try:
                with wave.open(str(path), "rb") as handle:
                    frames = handle.getnframes()
                    rate = handle.getframerate()
                    return frames / rate if rate else None
            except (wave.Error, OSError) as exc:
                logger.debug("wave could not read %s: %s", path, exc)
                # Fall through to soundfile: some "wav" files are actually
                # compressed formats with a misleading extension.

        try:
            import soundfile  # noqa: PLC0415 - optional dependency
        except ImportError:
            logger.debug("soundfile unavailable; cannot measure %s", path)
            return None

        try:
            info = soundfile.info(str(path))
            return float(info.duration)
        except (RuntimeError, OSError) as exc:
            logger.warning("Could not read audio %s: %s", path, exc)
            return None

    def _log_summary(self, report: AuditReport) -> None:
        logger.info(
            "Audited %d lines: %d matched, %d missing audio, %d unreadable",
            report.total_lines,
            len(report.matched),
            len(report.missing_audio),
            len(report.unreadable_audio),
        )
        logger.info(
            "Total audio: %.2f hours across %d speakers; vocabulary %d words",
            report.total_duration_hours,
            len(report.speakers),
            len(report.word_frequency),
        )
        coverage = report.tone_mark_coverage
        if coverage < 0.5:
            logger.error(
                "Only %.1f%% of transcripts carry tone marks. Tone comparison "
                "will not work on this corpus without re-diacritization.",
                coverage * 100,
            )
        elif coverage < 0.95:
            logger.warning(
                "%.1f%% tone-mark coverage; %d lines are undiacritized and "
                "should be excluded from tone training.",
                coverage * 100,
                report.total_lines - report.lines_with_tone_marks,
            )

    # ------------------------------------------------------------------
    # Deliverables
    # ------------------------------------------------------------------

    def write_all(self, report: AuditReport, output_dir: Path | None = None) -> dict[str, Path]:
        """Write the four Stage 1 deliverables.

        Returns:
            Mapping of deliverable name to the path written.
        """
        out = output_dir or self.config.ensure_output_dir()
        out.mkdir(parents=True, exist_ok=True)

        paths = {
            "dataset_report": out / "dataset_report.txt",
            "vocabulary": out / "vocabulary.txt",
            "word_frequency": out / "word_frequency.csv",
            "missing_audio": out / "missing_audio.csv",
        }

        self._write_report(report, paths["dataset_report"])
        self._write_vocabulary(report, paths["vocabulary"])
        self._write_frequency(report, paths["word_frequency"])
        self._write_missing(report, paths["missing_audio"])
        return paths

    def _write_report(self, report: AuditReport, path: Path) -> None:
        singletons = sum(1 for count in report.word_frequency.values() if count == 1)
        lines = [
            "YORUBA CORPUS AUDIT REPORT",
            "=" * 60,
            "",
            f"Index file          : {self.config.line_index}",
            f"Audio directory     : {self.config.audio_dir}",
            "",
            "COVERAGE",
            "-" * 60,
            f"Total index lines   : {report.total_lines}",
            f"Matched utterances  : {len(report.matched)}",
            f"Missing audio       : {len(report.missing_audio)}",
            f"Unreadable audio    : {len(report.unreadable_audio)}",
            f"Empty transcripts   : {len(report.empty_transcripts)}",
            "",
            "DURATION",
            "-" * 60,
            f"Total audio         : {report.total_duration_hours:.2f} hours "
            f"({report.total_duration_s:.0f} s)",
            f"Mean utterance      : "
            f"{report.total_duration_s / len(report.matched):.2f} s"
            if report.matched
            else "Mean utterance      : n/a",
            f"Distinct speakers   : {len(report.speakers)}",
            "",
            "VOCABULARY",
            "-" * 60,
            f"Unique words        : {len(report.word_frequency)}",
            f"Total tokens        : {sum(report.word_frequency.values())}",
            f"Non-speech markers  : {report.noise_tokens} "
            f"([snap]/[breath]/etc, mapped to the spn phone)",
            f"Hapax legomena      : {singletons} "
            f"({singletons / len(report.word_frequency) * 100:.1f}% of vocabulary)"
            if report.word_frequency
            else "Hapax legomena      : 0",
            "",
            "TONE DIACRITICS",
            "-" * 60,
            f"Lines with marks    : {report.lines_with_tone_marks}",
            f"Coverage            : {report.tone_mark_coverage * 100:.1f}%",
            "",
            "READINESS",
            "-" * 60,
        ]

        if report.tone_mark_coverage < 0.5:
            lines.append(
                "BLOCKER: corpus is largely undiacritized. Tone comparison is "
                "impossible without tone marks in the reference transcripts."
            )
        if not report.is_trainable:
            lines.append(
                "WARNING: under 3 hours or under 500 utterances. Train a "
                "speaker-adapted acoustic model rather than a speaker-"
                "independent one, and expect coarser phone boundaries."
            )
        else:
            lines.append("Corpus size is adequate for MFA acoustic model training.")

        if report.missing_audio:
            lines.append(
                f"NOTE: {len(report.missing_audio)} transcripts have no audio; "
                "see missing_audio.csv. Exclude them before training."
            )

        lines.extend(["", "TOP 25 WORDS", "-" * 60])
        for word, count in report.word_frequency.most_common(25):
            lines.append(f"{count:>8}  {word}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote %s", path)

    @staticmethod
    def _write_vocabulary(report: AuditReport, path: Path) -> None:
        path.write_text("\n".join(report.vocabulary) + "\n", encoding="utf-8")
        logger.info("Wrote %d words to %s", len(report.vocabulary), path)

    @staticmethod
    def _write_frequency(report: AuditReport, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["word", "frequency", "has_tone_marks"])
            for word, count in report.word_frequency.most_common():
                writer.writerow([word, count, int(has_tone_marks(word))])
        logger.info("Wrote %s", path)

    @staticmethod
    def _write_missing(report: AuditReport, path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["utterance_id", "transcript"])
            writer.writerows(report.missing_audio)
        logger.info("Wrote %d missing-audio rows to %s", len(report.missing_audio), path)
