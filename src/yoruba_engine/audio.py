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
