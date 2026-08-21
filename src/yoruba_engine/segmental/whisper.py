"""Vowel features and transcripts from one Whisper model.

The engine previously needed two large models to do two jobs: a fine-tuned
wav2vec2 for vowel identity, and — once open-vocabulary word identity mattered —
something that could transcribe. Whisper does both from a single checkpoint, and
the encoder pass is shared between them.

**What it costs.** Measured on identical rows with the same classifier and
speaker split, Whisper's encoder reaches 0.889 macro F1 against fine-tuned
wav2vec2's 0.904. The 0.015 sits almost entirely in one class -- ``in`` at 0.779
against 0.834 -- and both encoders are weak there.

**What it buys.** 968 MB rather than 1.8 GB, one encoder resident instead of
two, and roughly 16.4 s per assessment rather than 19.6 s, because transcription
reuses the encoder pass rather than adding one. It also names words the
9,092-entry lexicon does not contain: a learner reading ``òkun`` who says
``ọkun`` was flagged but never named, because no margin can surface a word the
decoder has never heard of.

**Fine-tuning it is not worth doing.** Fine-tuning the encoder on 482,541 vowel
windows moved it from 0.888 to 0.889 locally, against the +0.025 the same
treatment gave wav2vec2. Whisper-small-yoruba is already Yoruba-adapted, so the
headroom wav2vec2 had was already spent. The Kaggle run reported +0.012, but
that scored a linear head trained beside the encoder; a gradient-boosted
classifier was already reading everything the features held.

Two mechanical differences from the wav2vec2 path, both handled here:

Whisper pads every input to 30 seconds, so the encoder always returns 1500
frames and a two-second clip costs what a thirty-second one does. Vowel spans
index into the real region, so the padding is harmless -- but a long recording
must be processed in windows, since anything past 30 seconds is discarded.

Its frame rate is the same 50 per second, one frame per 20 ms, so the
span-to-frame arithmetic is identical and pooling stays comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..audio import resample
from ..exceptions import PitchExtractionError

logger = logging.getLogger(__name__)

#: The checkpoint serving both jobs.
DEFAULT_MODEL = "models/whisper-yoruba"

#: Which hidden state to pool. Index 0 is the convolutional output, so layer 12
#: is ``hidden_states[12]`` -- the last, for whisper-small.
#:
#: Whisper prefers depth where wav2vec2 does not. Swept on identical rows:
#: layer 6 gives 0.851 macro F1, layer 9 gives 0.876, layer 12 gives 0.888,
#: rising monotonically. wav2vec2's layer 16 beat its layer 24.
DEFAULT_LAYER = 12

#: One encoder frame per 20 ms.
FRAME_S = 0.02

#: Whisper's fixed input length. Audio beyond this is not seen at all.
WINDOW_S = 30.0


class WhisperExtractor:
    """Mean-pools a Whisper encoder layer over vowel spans."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        layer: int = DEFAULT_LAYER,
        device: str | None = None,
        sample_rate_hz: int = 16000,
    ) -> None:
        self.model_path = Path(model_path)
        self.layer = layer
        self.sample_rate_hz = sample_rate_hz
        self._device = device
        self._model = None
        self._features = None
        self._torch = None

    @property
    def available(self) -> bool:
        return (self.model_path / "config.json").exists() and any(
            (self.model_path / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin")
        )

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import json  # noqa: PLC0415

            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                WhisperFeatureExtractor, WhisperModel,
            )
            from transformers.models.whisper.modeling_whisper import (  # noqa: PLC0415
                WhisperEncoder,
            )
        except ImportError as exc:
            raise PitchExtractionError(
                "Learned vowel features need torch and transformers:\n"
                "  pip install torch transformers"
            ) from exc
        if not self.available:
            raise PitchExtractionError(
                f"No Whisper checkpoint at {self.model_path}. Fetch it with:\n"
                "  python scripts/fetch_models.py"
            )

        # An encoder-only checkpoint carries no decoder and no feature
        # extractor; the full model carries both. Reading the config tells them
        # apart, rather than guessing and getting either a missing attribute or
        # silently randomised weights.
        architectures = json.loads(
            (self.model_path / "config.json").read_text(encoding="utf-8")
        ).get("architectures") or []
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "WhisperEncoder" in architectures:
            self._model = WhisperEncoder.from_pretrained(self.model_path)
            extractor_source = DEFAULT_MODEL
        else:
            self._model = WhisperModel.from_pretrained(self.model_path).encoder
            extractor_source = self.model_path
        self._model = self._model.eval().to(device)
        self._features = WhisperFeatureExtractor.from_pretrained(extractor_source)
        self._torch = torch
        self._resolved_device = device
        logger.info("Loaded %s encoder layer %d on %s",
                    self.model_path, self.layer, device)
        return self._model

    def analyse(self, audio_path: Path):
        """Return ``(frame_times, frames)`` for the whole recording."""
        model = self._load()
        torch = self._torch
        try:
            import numpy as np  # noqa: PLC0415
            import soundfile as sf  # noqa: PLC0415

            samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise PitchExtractionError(f"Could not read {audio_path}: {exc}") from exc

        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != self.sample_rate_hz:
            samples = resample(samples, rate, self.sample_rate_hz)
        if len(samples) < self.sample_rate_hz // 10:
            raise PitchExtractionError(f"{audio_path.name} is too short to analyse")

        # Anything past 30 seconds is discarded by the feature extractor rather
        # than truncated loudly, so long recordings are processed in windows and
        # their frames concatenated. A learner utterance is seconds long, but a
        # reference reading need not be.
        window = int(WINDOW_S * self.sample_rate_hz)
        collected = []
        for start in range(0, max(len(samples), 1), window):
            chunk = samples[start:start + window]
            if len(chunk) < self.sample_rate_hz // 10:
                break
            mel = self._features(chunk, sampling_rate=self.sample_rate_hz,
                                 return_tensors="pt")
            with torch.inference_mode():
                output = model(mel.input_features.to(self._resolved_device),
                               output_hidden_states=True)
            frames = output.hidden_states[self.layer][0].float().cpu().numpy()
            # Keep only the frames covering real audio; the rest is padding.
            kept = max(int(len(chunk) / self.sample_rate_hz / FRAME_S), 1)
            collected.append(frames[:kept])
        if not collected:
            raise PitchExtractionError(f"{audio_path.name} produced no frames")
        frames = np.concatenate(collected)
        times = np.arange(len(frames)) * FRAME_S
        return times, frames

    def measure_segment(self, times, frames, start_s: float, end_s: float):
        """Mean of every frame inside one vowel, or ``None`` if there are none.

        No edge trimming, matching the wav2vec2 path: a transformer frame
        already carries context from beyond its own 20 ms, so discarding edges
        removes signal rather than consonant contamination.
        """
        if end_s <= start_s or len(frames) == 0:
            return None
        first = max(int(start_s / FRAME_S), 0)
        last = min(max(int(end_s / FRAME_S), first + 1), len(frames))
        if first >= len(frames):
            return None
        return list(frames[first:last].mean(axis=0))
