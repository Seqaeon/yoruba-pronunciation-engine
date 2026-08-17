"""Formant measurement for vowel identity.

Tone tells you *how* a vowel was pitched; formants tell you *which vowel it was*.
Both are needed. A learner who says ``òkan`` when the prompt was ``òkun`` has
produced the right number of syllables with plausible durations and possibly the
right tones -- alignment and tone analysis alone will call that correct.

The first two formants separate the Yoruba vowel inventory well. Measured on the
training corpus:

======  =======  =======
Vowel   F1 (Hz)  F2 (Hz)
======  =======  =======
i       337      2212
e       365      2148
ẹ       477      1900
a       692      1443
u       395      1299
o       424      1125
ọ       516      1165
======  =======  =======

``u`` and ``a`` differ by roughly 300 Hz in F1, which is why a substitution
between them is recoverable from the audio.

Formants are measured with Praat's Burg algorithm through Parselmouth. Values are
also expressed relative to the speaker's own vowel space, because absolute
formant frequencies vary with vocal tract length -- the same normalization
argument that applies to pitch.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from ..exceptions import PitchExtractionError
from ..logging_utils import get_logger

logger = get_logger("formants")

#: Praat's formant ceiling. 5500 Hz suits female speakers, 5000 Hz male ones;
#: the midpoint works acceptably for both and avoids needing sex labels.
DEFAULT_MAX_FORMANT_HZ = 5500.0


class FormantExtractor:
    """Measures F1-F3 at sample points within vowel segments."""

    def __init__(
        self,
        time_step_s: float = 0.01,
        max_formant_hz: float = DEFAULT_MAX_FORMANT_HZ,
        edge_trim_ratio: float = 0.25,
    ) -> None:
        """
        Args:
            time_step_s: Analysis frame step.
            max_formant_hz: Praat's formant ceiling.
            edge_trim_ratio: Fraction of each end of a vowel to ignore, where
                the neighbouring consonants pull the formants toward their own
                targets.
        """
        self.time_step_s = time_step_s
        self.max_formant_hz = max_formant_hz
        self.edge_trim_ratio = edge_trim_ratio

    def analyse(self, audio_path: Path):
        """Return a Praat formant object for the whole recording.

        Raises:
            PitchExtractionError: If Parselmouth is unavailable or the audio
                cannot be analysed.
        """
        try:
            import parselmouth  # noqa: PLC0415
        except ImportError as exc:
            raise PitchExtractionError(
                "Vowel identity checking needs Parselmouth:\n"
                "  pip install praat-parselmouth"
            ) from exc

        try:
            sound = parselmouth.Sound(str(audio_path))
            return sound.to_formant_burg(
                time_step=self.time_step_s,
                max_number_of_formants=5,
                maximum_formant=self.max_formant_hz,
            )
        except Exception as exc:  # parselmouth raises bare Praat errors
            raise PitchExtractionError(f"Formant analysis failed for {audio_path}: {exc}") from exc

    def measure_segment(self, formants, start_s: float, end_s: float) -> dict[str, float] | None:
        """Measure a single vowel.

        Samples at 30%, 50% and 70% through the trimmed segment and takes the
        median, which resists the occasional frame where Praat's formant tracker
        picks the wrong peak.

        Returns:
            ``{"f1": ..., "f2": ..., "f3": ...}``, or ``None`` if the segment is
            too short or unmeasurable.
        """
        duration = end_s - start_s
        if duration <= 0:
            return None

        trim = self.edge_trim_ratio * duration
        inner_start, inner_end = start_s + trim, end_s - trim
        if inner_end <= inner_start:
            inner_start, inner_end = start_s, end_s

        keys = ("f1", "f2", "f3", "b1", "b2", "b3")
        samples: dict[str, list[float]] = {key: [] for key in keys}
        for position in (0.3, 0.5, 0.7):
            time_s = inner_start + position * (inner_end - inner_start)
            for index in (1, 2, 3):
                try:
                    value = formants.get_value_at_time(index, time_s)
                    bandwidth = formants.get_bandwidth_at_time(index, time_s)
                except Exception:  # noqa: BLE001 - unmeasurable frame
                    continue
                # Praat returns NaN where it cannot track a formant.
                if value and value == value:
                    samples[f"f{index}"].append(float(value))
                if bandwidth and bandwidth == bandwidth:
                    samples[f"b{index}"].append(float(bandwidth))

        if not samples["f1"] or not samples["f2"]:
            return None

        return {
            key: statistics.median(values) if values else 0.0
            for key, values in samples.items()
        }


def normalize_vowel_space(measurements: list[dict[str, float]]) -> tuple[float, float]:
    """Return the speaker's median F1 and F2 across an utterance.

    Absolute formant values scale with vocal tract length, so a child, a woman
    and a man produce the same vowel at very different frequencies. Expressing
    each vowel relative to the speaker's own median makes the measurements
    comparable, exactly as semitone normalization does for pitch.
    """
    f1_values = [m["f1"] for m in measurements if m.get("f1")]
    f2_values = [m["f2"] for m in measurements if m.get("f2")]
    if not f1_values or not f2_values:
        return 500.0, 1500.0  # neutral defaults
    return statistics.median(f1_values), statistics.median(f2_values)
