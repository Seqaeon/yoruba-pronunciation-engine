"""MFCC measurement for vowel identity.

Formants alone are not enough. F1 and F2 describe tongue position, which is
nearly identical for a vowel and its nasal counterpart -- so a formant-only
classifier trained on this corpus never predicted the nasal vowels at all
(0-4% recall on ``an``, ``ẹn``, ``in``, ``ọn``, ``un``). Nasality shows up as
antiformants and changes in spectral bandwidth, neither of which a formant
tracker reports.

MFCCs encode the whole spectral envelope, so nasality is present in the
features rather than invisible to them.

Two normalizations matter:

**Cepstral mean subtraction, per speaker.** The low-order coefficients carry
vocal-tract length and channel response. Subtracting the speaker's own mean
removes both, which is the same argument that makes semitones the right unit for
pitch and speaker-relative formants the right unit for vowel space.

**Variance normalization.** Coefficient scales differ by an order of magnitude
between c0 and c12; leaving that in place lets the loudest coefficients dominate
a distance-based model.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from ..exceptions import PitchExtractionError
from ..logging_utils import get_logger

logger = get_logger("mfcc")

#: Number of cepstral coefficients, excluding c0 (energy), which Praat returns
#: as the first row. Twelve is the long-standing default for speech.
DEFAULT_COEFFICIENTS = 12


class MFCCExtractor:
    """Measures cepstral coefficients over vowel segments."""

    def __init__(
        self,
        coefficients: int = DEFAULT_COEFFICIENTS,
        time_step_s: float = 0.005,
        edge_trim_ratio: float = 0.25,
        sample_rate_hz: int = 16000,
    ) -> None:
        """
        Args:
            coefficients: Cepstral coefficients beyond c0.
            time_step_s: Frame step. Shorter than the pitch step because short
                vowels otherwise yield too few frames to average.
            edge_trim_ratio: Fraction of each vowel edge to ignore, where the
                neighbouring consonants dominate the spectrum.
            sample_rate_hz: Rate every recording is resampled to before
                analysis. Unlike formants, which Praat's Burg method makes
                rate-independent by resampling to twice the formant ceiling,
                MFCCs are computed over the whole available band, so the same
                voice yields different coefficients at different rates.
                Measured on one corpus recording, 48 kHz against 16 kHz:

                    coefficient    0      1      2      3
                    difference   109.2   85.9   69.3   40.6
                    training std 778.9   64.9   58.2   66.2

                Coefficients 1 and 2 move by more than a full standard
                deviation of the training data, which is enough on its own to
                put a recording outside the distribution the classifier learned.
                The corpus this model was trained on was written at 16 kHz, so
                that is the rate to match.
        """
        self.coefficients = coefficients
        self.time_step_s = time_step_s
        self.edge_trim_ratio = edge_trim_ratio
        self.sample_rate_hz = sample_rate_hz

    def analyse(self, audio_path: Path):
        """Return ``(frame_times, frames)`` for the whole recording.

        ``frames`` is a list of coefficient vectors, one per analysis frame.

        Raises:
            PitchExtractionError: If Parselmouth is missing or analysis fails.
        """
        try:
            import numpy as np  # noqa: PLC0415
            import parselmouth  # noqa: PLC0415
        except ImportError as exc:
            raise PitchExtractionError(
                "MFCC measurement needs Parselmouth:\n  pip install praat-parselmouth"
            ) from exc

        try:
            sound = parselmouth.Sound(str(audio_path))
            if abs(sound.sampling_frequency - self.sample_rate_hz) > 1:
                sound = sound.resample(self.sample_rate_hz)
            mfcc = sound.to_mfcc(number_of_coefficients=self.coefficients)
        except Exception as exc:
            raise PitchExtractionError(f"MFCC analysis failed for {audio_path}: {exc}") from exc

        # to_array() is (coefficients + 1, frames); transpose to frame-major.
        array = np.asarray(mfcc.to_array()).T
        times = np.asarray(mfcc.xs())
        return times, array

    def measure_segment(self, times, frames, start_s: float, end_s: float):
        """Mean and standard deviation of each coefficient across one vowel.

        The standard deviation is kept because it describes how much the spectrum
        moves during the vowel, which separates a steady monophthong from a
        transition the aligner has mislabelled.

        Returns:
            A flat list of ``2 * (coefficients + 1)`` values, or ``None`` when
            the segment contains too few frames to average.
        """
        import numpy as np  # noqa: PLC0415

        duration = end_s - start_s
        if duration <= 0:
            return None

        trim = self.edge_trim_ratio * duration
        inner_start, inner_end = start_s + trim, end_s - trim
        if inner_end <= inner_start:
            inner_start, inner_end = start_s, end_s

        mask = (times >= inner_start) & (times <= inner_end)
        selected = frames[mask]
        if selected.shape[0] < 2:
            # Fall back to the untrimmed span before giving up: very short
            # vowels can lose every frame to trimming.
            mask = (times >= start_s) & (times <= end_s)
            selected = frames[mask]
        if selected.shape[0] < 1:
            return None

        means = selected.mean(axis=0)
        deviations = selected.std(axis=0) if selected.shape[0] > 1 else np.zeros_like(means)
        return list(means) + list(deviations)


def cepstral_mean_variance(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    """Per-speaker mean and standard deviation of each feature.

    Computed across every vowel of one speaker, then subtracted and divided out.
    This is what removes vocal-tract length and recording-channel effects, and it
    is the difference between a classifier that transfers to a new voice and one
    that memorizes the voices it trained on.
    """
    if not vectors:
        return [], []

    width = len(vectors[0])
    means, deviations = [], []
    for index in range(width):
        column = [v[index] for v in vectors if len(v) > index]
        means.append(statistics.mean(column) if column else 0.0)
        deviations.append(
            statistics.pstdev(column) if len(column) > 1 else 1.0
        )
    # Guard against zero-variance columns, which would divide by nothing.
    deviations = [d if d > 1e-9 else 1.0 for d in deviations]
    return means, deviations


def normalize_vector(vector: list[float], means: list[float], deviations: list[float]) -> list[float]:
    """Apply per-speaker mean subtraction and variance scaling."""
    return [
        (value - means[index]) / deviations[index] if index < len(means) else value
        for index, value in enumerate(vector)
    ]
