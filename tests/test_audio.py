"""Resampling must filter, not merely drop samples.

Dropping from 48 kHz to 16 kHz discards two samples in three, and anything above
8 kHz in the original folds back into the speech band unless it is removed
first. ``numpy.interp`` does no filtering, and the difference is not subtle: the
engine transcribed a learner saying ``ọkun`` as ``ń kún`` through linear
interpolation and as ``ọkún`` through a filtered resampler, same audio and same
weights.
"""

from __future__ import annotations

import numpy as np

from yoruba_engine.audio import TARGET_RATE_HZ, resample


def test_length_and_rate() -> None:
    source = np.zeros(48000, dtype=np.float32)
    out = resample(source, 48000, 16000)
    assert abs(len(out) - 16000) <= 1, "one second in, one second out"
    assert out.dtype == np.float32


def test_passthrough_when_rates_match() -> None:
    source = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert np.array_equal(resample(source, TARGET_RATE_HZ, TARGET_RATE_HZ), source)


def test_content_above_nyquist_is_removed_not_folded() -> None:
    """A 12 kHz tone cannot be represented at 16 kHz and must not reappear.

    Without a filter it aliases to |16000 - 12000| = 4 kHz, landing in the
    middle of the vowel formant range, where a classifier reads it as signal.
    """
    duration, rate = 0.5, 48000
    t = np.arange(int(duration * rate)) / rate
    tone = np.sin(2 * np.pi * 12000 * t).astype(np.float32)

    filtered = resample(tone, rate, 16000)
    naive_count = int(len(tone) / rate * 16000)
    naive = np.interp(np.linspace(0, len(tone) - 1, naive_count),
                      np.arange(len(tone)), tone).astype(np.float32)

    def energy_at(signal: np.ndarray, hz: float) -> float:
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / 16000)
        band = (freqs > hz - 300) & (freqs < hz + 300)
        return float(spectrum[band].max())

    # The alias lands at 4 kHz. Filtering must leave far less there.
    assert energy_at(filtered, 4000) < energy_at(naive, 4000) / 10, (
        "the filtered resampler still let a 12 kHz tone alias into 4 kHz"
    )


def test_read_mono_downmixes_and_resamples(tmp_path) -> None:
    import soundfile as sf

    path = tmp_path / "stereo.wav"
    stereo = np.zeros((48000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    sf.write(str(path), stereo, 48000)

    from yoruba_engine.audio import read_mono

    out = read_mono(path)
    assert abs(len(out) - 16000) <= 1
    assert out.ndim == 1
    # The two channels cancel, so the downmix must be near silence.
    assert abs(out).max() < 0.05
