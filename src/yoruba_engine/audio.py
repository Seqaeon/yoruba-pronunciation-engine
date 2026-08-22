"""Reading and resampling audio, correctly.

Everything in this engine expects 16 kHz mono, and learners record at whatever
their device produces -- 48 kHz stereo, usually. The conversion looks trivial
and is not.

**Linear interpolation is not resampling.** Dropping from 48 kHz to 16 kHz
discards two of every three samples, and anything above 8 kHz in the original
has nowhere to go: it folds back into the speech band as alias noise. A proper
resampler low-passes first so that content is removed rather than disguised.
``numpy.interp`` does no filtering at all.

This was not theoretical. The engine transcribed a learner saying ``ọkun`` as
``ń kún`` through ``np.interp`` and as ``ọkún`` through a filtered resampler --
same audio, same model, same weights. The error was read as the recogniser
mishearing, when it was the pipeline handing it corrupted audio.

``scipy.signal.resample_poly`` does polyphase filtering: it low-passes and
decimates in one pass, is exact for rational ratios like 48000/16000, and is
fast enough that it does not register against forced alignment.
"""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: What every model in this project expects.
TARGET_RATE_HZ = 16000


def resample(samples: np.ndarray, source_rate: int,
             target_rate: int = TARGET_RATE_HZ) -> np.ndarray:
    """Resample with an anti-aliasing filter.

    Falls back to linear interpolation only if SciPy is missing, and says so
    loudly: the difference is audible to a model even when it is not to a
    listener.
    """
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "SciPy is not installed, so audio is being resampled by linear "
            "interpolation, which does not filter and lets everything above "
            "%d Hz alias into the speech band. Install scipy.",
            target_rate // 2,
        )
        count = int(len(samples) / source_rate * target_rate)
        return np.interp(
            np.linspace(0, len(samples) - 1, count),
            np.arange(len(samples)), samples,
        ).astype(np.float32)

    divisor = gcd(int(source_rate), int(target_rate))
    up, down = int(target_rate) // divisor, int(source_rate) // divisor
    return resample_poly(samples, up, down).astype(np.float32)


def read_mono(path: Path | str, target_rate: int = TARGET_RATE_HZ) -> np.ndarray:
    """Read any audio file as mono at ``target_rate``.

    Channels are averaged before resampling, which is the cheaper order and
    makes no difference to the result.
    """
    import soundfile as sf  # noqa: PLC0415

    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    return resample(samples, rate, target_rate)


def trim_silence(samples: np.ndarray, rate: int = TARGET_RATE_HZ,
                 margin_s: float = 0.15, headroom_db: float = 10.0) -> np.ndarray:
    """Remove dead air from the ends of a recording.

    A recording made through a browser starts when the learner taps the button
    and ends when they tap it again, so it carries silence at both ends that has
    nothing to do with how they spoke. Removing it matters for two reasons: the
    prompted-mode coverage check divides aligned speech by the *whole file*, so
    dead air counts against a learner who did nothing wrong, and every second
    kept is a second forced alignment and the encoder have to process.

    **The threshold is relative to the recording's own noise floor, not its
    peak.** A peak-relative floor is what broke this: the previous version
    trimmed frames quieter than 2% of the loudest one, which is -34 dB, and
    ordinary room noise sits above that. On a laptop microphone nothing was
    trimmed at all -- a padded 6.08 s recording came back as 6.08 s -- so the
    dead air survived and a word-perfect attempt was reported unverified. The
    same recordings now trim to 1.9-3.6 s across noise floors from -60 to
    -25 dB relative to peak.

    The cost of a noise-relative threshold is that a very noisy recording
    over-trims: at -25 dB the quietest edges of speech fall below the floor and
    about 0.8 s is lost. ``margin_s`` covers the ordinary case, and losing a
    quiet edge is a better failure than keeping four seconds of hiss.

    Only the ends are trimmed. Silence *within* an utterance is real evidence
    about how it was spoken, and removing it would hide hesitation.

    Args:
        samples: Mono audio.
        rate: Its sample rate.
        margin_s: Kept either side of the speech, so an onset is not clipped.
        headroom_db: How far above the noise floor a frame must be to count as
            speech.

    Returns:
        The trimmed audio, or the original if no speech was found -- an empty
        return would turn a quiet recording into a crash.
    """
    window = max(int(0.02 * rate), 1)
    frames = len(samples) // window
    if frames < 2:
        return samples

    energy = np.sqrt(np.mean(
        samples[:frames * window].reshape(frames, window) ** 2, axis=1))
    level_db = 20 * np.log10(np.maximum(energy, 1e-9))

    # The 10th percentile is the noise floor when the recording contains any
    # silence at all. When it does not, that percentile is a quiet speech frame
    # instead, so the threshold is also capped 15 dB below the loudest frame --
    # without the cap, a recording that is speech end to end would be trimmed
    # to its loudest syllable.
    floor_db = float(np.percentile(level_db, 10))
    threshold_db = min(floor_db + headroom_db, float(level_db.max()) - 15.0)

    voiced = np.flatnonzero(level_db > threshold_db)
    if len(voiced) == 0:
        return samples

    margin = int(margin_s * rate)
    start = max(int(voiced[0]) * window - margin, 0)
    end = min((int(voiced[-1]) + 1) * window + margin, len(samples))
    return samples[start:end]
