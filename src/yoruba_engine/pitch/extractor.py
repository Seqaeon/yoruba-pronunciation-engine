"""Stage 4: pitch extraction over aligned vowel segments.

Uses Parselmouth (Praat's autocorrelation pitch tracker) when available. Praat's
tracker is the right default for tone work: it is robust on short segments and
its octave-jump costs are well tuned, which matters because octave errors are
the single largest source of spurious "tone mismatch" feedback.

Pitch is converted to semitones relative to the speaker's own median F0. This
normalization is what makes a male reference speaker comparable to a female
learner: absolute Hz are not comparable across speakers, but the *relative*
H/M/L pattern is exactly what Yoruba tone encodes.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path

from ..config import PitchConfig
from ..exceptions import PitchExtractionError
from ..logging_utils import get_logger
from ..datatypes import Alignment, Interval, PitchPoint, VowelSegment, WordAnalysis

logger = get_logger("pitch")

#: Phone-label suffixes written by the G2P; used to identify tone-bearing phones.
_VOWEL_SUFFIXES = ("_H", "_M", "_L")

#: Vowel phones without a tone suffix, as produced when the dictionary is built
#: with ``TONE_IN_PHONES = False``. ``E``/``O`` are the dot-below vowels; the
#: ``n``-final entries are the nasal vowels.
_TONELESS_VOWELS = frozenset(
    {"a", "e", "E", "i", "o", "O", "u", "an", "en", "En", "in", "on", "On", "un"}
)


def is_vowel_phone(label: str) -> bool:
    """Whether an MFA phone label denotes a tone-bearing (vowel) segment.

    Handles both dictionary styles. With tone-suffixed phones the label ends in
    ``_H``/``_M``/``_L``; with tone-stripped phones it is the bare vowel.

    Note that the syllabic nasal is only recognized in the tone-suffixed style
    (``n_H``). Once stripped it collapses to ``n``, which is indistinguishable
    from the consonant, so it is excluded rather than risk measuring pitch over
    consonantal nasals.
    """
    if label.endswith(_VOWEL_SUFFIXES):
        return True
    return label in _TONELESS_VOWELS


def hz_to_semitones(f0_hz: float, reference_hz: float) -> float:
    """Convert an F0 value to semitones relative to a reference frequency.

    Args:
        f0_hz: The measured fundamental frequency.
        reference_hz: The speaker's median F0.

    Returns:
        Distance in semitones; positive is higher than the reference.
    """
    if f0_hz <= 0 or reference_hz <= 0:
        raise PitchExtractionError(
            f"Cannot convert non-positive frequency ({f0_hz} Hz, ref {reference_hz} Hz)"
        )
    return 12.0 * math.log2(f0_hz / reference_hz)


class PitchExtractor:
    """Extracts F0 contours for the vowel segments of an aligned utterance."""

    def __init__(self, config: PitchConfig | None = None) -> None:
        self.config = config or PitchConfig()

    def extract(self, audio_path: Path, alignment: Alignment) -> tuple[list[WordAnalysis], float]:
        """Extract per-vowel pitch contours for every word in the alignment.

        Args:
            audio_path: The recording the alignment came from.
            alignment: Word and phone intervals from MFA.

        Returns:
            A tuple of (word analyses, speaker median F0 in Hz). The median is
            computed across all voiced frames in this utterance and is the
            reference for semitone normalization.

        Raises:
            PitchExtractionError: If the audio cannot be read, or no voiced
                frames were found anywhere in the utterance.
        """
        pitch_track = self._track_pitch(audio_path)
        if not pitch_track:
            raise PitchExtractionError(
                f"No voiced frames found in {audio_path}. The recording may be "
                "silent, too noisy, or entirely whispered."
            )

        median_hz = statistics.median(f0 for _, f0 in pitch_track)
        logger.debug("Speaker median F0 for %s: %.1f Hz", audio_path.name, median_hz)

        words: list[WordAnalysis] = []
        for word_index, word_interval in enumerate(alignment.words):
            phones = alignment.phones_within(word_interval)
            vowels = [
                self._analyse_vowel(phone, word_index, pitch_track, median_hz)
                for phone in phones
                if is_vowel_phone(phone.label)
            ]
            words.append(
                WordAnalysis(
                    word=word_interval.label,
                    start_s=word_interval.start_s,
                    end_s=word_interval.end_s,
                    vowels=vowels,
                    phone_count=len(phones),
                )
            )

        total_vowels = sum(len(w.vowels) for w in words)
        logger.info(
            "Extracted pitch for %d vowels across %d words in %s",
            total_vowels, len(words), audio_path.name,
        )
        return words, median_hz

    def _analyse_vowel(
        self,
        phone: Interval,
        word_index: int,
        pitch_track: list[tuple[float, float]],
        median_hz: float,
    ) -> VowelSegment:
        """Build a :class:`VowelSegment` with its trimmed pitch contour."""
        trim = self.config.edge_trim_ratio * phone.duration_s
        start = phone.start_s + trim
        end = phone.end_s - trim

        contour: list[PitchPoint] = []
        for time_s, f0_hz in pitch_track:
            if start <= time_s <= end and f0_hz > 0:
                contour.append(
                    PitchPoint(
                        time_s=time_s,
                        f0_hz=f0_hz,
                        semitones=hz_to_semitones(f0_hz, median_hz),
                    )
                )

        segment = VowelSegment(
            label=phone.label,
            start_s=phone.start_s,
            end_s=phone.end_s,
            word_index=word_index,
            contour=contour,
        )
        self._sample_shape(segment)
        return segment

    @staticmethod
    def _sample_shape(segment: VowelSegment) -> None:
        """Record F0 at 20%, 50% and 80% through the vowel.

        Three samples capture whether the vowel rises, falls, or stays level.
        That shape is informative on its own -- a High following a Low is
        realized with a rise that a single averaged value throws away.

        The 20/80 positions rather than 0/100 avoid the first and last frames,
        which are the least reliable even after edge trimming.
        """
        values = [p.semitones for p in segment.contour if p.semitones is not None]
        if len(values) < 3:
            return

        count = len(values)
        segment.onset_semitones = values[int(count * 0.2)]
        segment.mid_semitones = values[int(count * 0.5)]
        segment.offset_semitones = values[min(count - 1, int(count * 0.8))]

    def _track_pitch(self, audio_path: Path) -> list[tuple[float, float]]:
        """Return ``[(time_s, f0_hz), ...]`` for voiced frames only.

        Falls back to a pure-Python autocorrelation tracker if Parselmouth is
        not installed, so the pipeline remains testable in minimal environments.
        The fallback is markedly less accurate and logs a warning.
        """
        if not audio_path.exists():
            raise PitchExtractionError(f"Audio file not found: {audio_path}")

        try:
            import parselmouth  # noqa: PLC0415 - optional dependency
        except ImportError:
            logger.warning(
                "Parselmouth not installed; using the fallback pitch tracker. "
                "Install praat-parselmouth for production-quality tone detection."
            )
            return self._track_pitch_fallback(audio_path)

        try:
            sound = parselmouth.Sound(str(audio_path))
            pitch = sound.to_pitch(
                time_step=self.config.time_step_s,
                pitch_floor=self.config.floor_hz,
                pitch_ceiling=self.config.ceiling_hz,
            )
        except Exception as exc:  # parselmouth raises bare PraatError subclasses
            raise PitchExtractionError(f"Praat pitch analysis failed for {audio_path}: {exc}") from exc

        times = pitch.xs()
        values = pitch.selected_array["frequency"]
        return [(float(t), float(f)) for t, f in zip(times, values) if f > 0]

    def _track_pitch_fallback(self, audio_path: Path) -> list[tuple[float, float]]:
        """Autocorrelation pitch tracking using only numpy and soundfile.

        Provided so that unit tests and CI can run without Praat. Accuracy is
        sufficient for synthetic test tones, not for real learner audio.
        """
        try:
            import numpy as np  # noqa: PLC0415
            import soundfile  # noqa: PLC0415
        except ImportError as exc:
            raise PitchExtractionError(
                "Neither parselmouth nor (numpy + soundfile) is installed; "
                "cannot extract pitch."
            ) from exc

        try:
            samples, rate = soundfile.read(str(audio_path), dtype="float64", always_2d=False)
        except (RuntimeError, OSError) as exc:
            raise PitchExtractionError(f"Could not read {audio_path}: {exc}") from exc

        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        window = int(0.04 * rate)
        hop = max(1, int(self.config.time_step_s * rate))
        min_lag = int(rate / self.config.ceiling_hz)
        max_lag = int(rate / self.config.floor_hz)

        results: list[tuple[float, float]] = []
        for start in range(0, max(0, len(samples) - window), hop):
            frame = samples[start : start + window]
            frame = frame - frame.mean()
            energy = float(np.sqrt(np.mean(frame**2)))
            if energy < 1e-4:
                continue

            correlation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
            search = correlation[min_lag : min(max_lag, len(correlation))]
            if search.size == 0:
                continue

            peak = int(np.argmax(search)) + min_lag
            if correlation[0] <= 0 or search.max() / correlation[0] < 0.3:
                continue  # too unperiodic to call voiced

            results.append((start / rate + window / (2 * rate), rate / peak))

        return results
