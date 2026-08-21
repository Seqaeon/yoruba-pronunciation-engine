"""Learned frame representations for vowel identity.

This replaces the 39 hand-designed measurements -- formants, formant bandwidths,
cepstral means and standard deviations -- with a mean-pooled slice of a
self-supervised speech model. Measured on 131,360 vowels from 169 speakers, held
out by speaker on studio recordings only:

    features            dims    accuracy   macro F1    an     in     On     un
    39 hand-designed      39       0.595      0.432  0.249  0.129  0.307  0.256
    wav2vec2 layer 16   1024       0.873      0.811  0.791  0.696  0.813  0.718

The hand-designed set was the binding constraint on the whole engine, not the
quantity of training data. Adding 7.26 h of studio audio had moved the ``ọ``/``o``
contrast by 13.7 points; changing the features moved accuracy by 27.8.

**Layer 16 of 24, not the last.** A CTC model's final layer is optimised to
predict characters, and a transcription objective is rewarded for discarding
detail the transcript does not need -- which is exactly the detail separating
``an`` from ``un``. Swept on held-out speakers:

    layer      4      8     12     16     20     24
    accuracy 0.703  0.763  0.780  0.798  0.785  0.771

**Concatenating the old features adds +0.001 macro F1** -- noise. So Praat is no
longer needed for vowel identity, only for pitch.

The interface deliberately mirrors
:class:`~yoruba_engine.segmental.mfcc.MFCCExtractor`: ``analyse`` returns frame
times and frames, ``measure_segment`` pools one span. That lets
:class:`~yoruba_engine.segmental.verifier.VowelVerifier` hold either backend
without knowing which it has.
"""

from __future__ import annotations

from pathlib import Path

from ..exceptions import PitchExtractionError
from ..logging_utils import get_logger

logger = get_logger("segmental.wav2vec")

#: The model emits one frame per 20 ms at 16 kHz.
FRAME_S = 0.02

#: Default checkpoint. Yoruba-fine-tuned, and its 97-token vocabulary carries
#: every distinction the engine reasons about -- ``ẹ ọ ṣ``, all eight tone-marked
#: vowels, and the syllabic nasals ``ń ǹ``. A romanized model would have folded
#: those away before the audio was ever seen.
#: The encoder the engine reads its features from.
#:
#: LyngualLabs' ``mms-300m-yoruba-english``, fine-tuned here on aligned Yoruba
#: vowels and truncated to the 16 layers actually consumed. Measured against
#: alternatives on identical rows, with the same classifier and speaker split,
#: so only the encoder differs:
#:
#:     encoder                              accuracy   macro F1
#:     frozen upstream                        0.906      0.879
#:     Whisper small yoruba                   0.912      0.888
#:     fine-tuned on 0.4 s vowel windows      0.923      0.904
#:     fine-tuned on whole utterances         0.939      0.927   <- here
#:
#: The last two differ only in what the encoder saw during fine-tuning. Windows
#: were chosen so classes could be balanced by resampling, which whole
#: utterances cannot do; they cost more than that bought. An utterance carries
#: about fifteen labelled vowels against a window's one, so the same three
#: epochs saw nearly twice the training signal, and the encoder saw real
#: acoustic context rather than 400 ms around a vowel -- which is what inference
#: presents.
#:
#: Whisper was measured because one model could have served both vowel identity
#: and transcription, saving 850 MB and 3.2 s. At a 0.015 gap that trade looked
#: worth taking; at 0.038 it does not. Whisper still provides the transcript,
#: where being open-vocabulary lets it name words the 9,092-entry lexicon does
#: not contain.
DEFAULT_MODEL = "models/finetuned_utterance"

#: Named only so classifiers trained before the fine-tuned encoders can resolve
#: what they were fitted to. It is no longer shipped.
FALLBACK_MODEL = "models/lyngual-yoruba"

#: Which hidden state to pool. Index 0 is the convolutional output, so layer 16
#: is ``hidden_states[16]``.
DEFAULT_LAYER = 16


class Wav2VecExtractor:
    """Mean-pools a transformer layer over vowel spans."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        layer: int = DEFAULT_LAYER,
        device: str | None = None,
        sample_rate_hz: int = 16000,
    ) -> None:
        """
        Args:
            model_path: Directory holding the checkpoint.
            layer: Hidden state to pool. See the module docstring for the sweep.
            device: ``"cuda"``, ``"cpu"``, or ``None`` to prefer CUDA when
                present. Inference runs at about 4.7x realtime on CPU, which is
                negligible beside forced alignment's ~35 s, so a GPU is a
                convenience rather than a requirement.
            sample_rate_hz: The rate the model expects. Audio at any other rate
                is resampled, because a mismatch silently shifts every
                coefficient -- the same failure that once made a correctly
                spoken word score 0.002.
        """
        self.model_path = self.resolve(model_path)
        self.layer = layer
        self.sample_rate_hz = sample_rate_hz
        self._device = device
        self._model = None

    @property
    def available(self) -> bool:
        """Whether the checkpoint is present, without loading it."""
        return (self.model_path / "config.json").exists() and any(
            (self.model_path / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin")
        )

    @staticmethod
    def resolve(path: str | Path) -> Path:
        """The requested encoder, or an error naming what is missing.

        This used to fall back to the upstream checkpoint with a warning. That
        was the wrong behaviour: fine-tuning changes what the 1024 dimensions
        mean, so running a classifier against an encoder it was not fitted to
        produces confident wrong answers rather than obvious failure -- the same
        shape of mismatch that once made this engine score 0 out of 23 on real
        audio while every test passed.

        A missing encoder is a broken install, and a broken install should say
        so.
        """
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Encoder not found: {candidate}\n"
            "The vowel classifier records the encoder it was fitted to, and "
            "reads noise from any other.\n"
            "Fetch it with:  python scripts/fetch_models.py"
        )

    def _load(self):
        """Load once, on first use. The model is ~1.26 GB."""
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: PLC0415
            import json  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                Wav2Vec2ForCTC, Wav2Vec2Model,
            )
        except ImportError as exc:
            raise PitchExtractionError(
                "Learned vowel features need torch and transformers:\n"
                "  pip install torch transformers"
            ) from exc
        if not self.available:
            raise PitchExtractionError(
                f"No wav2vec2 checkpoint at {self.model_path}. Fetch it with:\n"
                "  wget -c -O models/lyngual-yoruba/model.safetensors \\\n"
                "    https://huggingface.co/LyngualLabs/mms-300m-yoruba-english"
                "/resolve/main/model.safetensors"
            )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        # The upstream checkpoint carries a CTC head; a checkpoint fine-tuned by
        # this project does not, because fine-tuning replaces that head with a
        # vowel head which is then discarded. Loading a headless checkpoint as
        # ``Wav2Vec2ForCTC`` silently invents an ``lm_head``. Only hidden states
        # are ever read, so either class serves once it is chosen honestly.
        config = Path(self.model_path) / "config.json"
        architectures = []
        if config.exists():
            architectures = json.loads(
                config.read_text(encoding="utf-8")).get("architectures") or []
        loader = Wav2Vec2ForCTC if "Wav2Vec2ForCTC" in architectures else Wav2Vec2Model
        self._model = loader.from_pretrained(self.model_path).eval().to(device)
        self._resolved_device = device
        logger.info("Loaded %s layer %d on %s", self.model_path, self.layer, device)
        return self._model

    def analyse(self, audio_path: Path):
        """Return ``(frame_times, frames)`` for the whole recording.

        Raises:
            PitchExtractionError: If the model or its dependencies are missing,
                or the audio cannot be read.
        """
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
            count = int(len(samples) / rate * self.sample_rate_hz)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, count),
                np.arange(len(samples)), samples,
            ).astype("float32")
        if len(samples) < self.sample_rate_hz // 10:
            raise PitchExtractionError(f"{audio_path.name} is too short to analyse")

        with torch.inference_mode():
            output = model(
                torch.from_numpy(samples)[None].to(self._resolved_device),
                output_hidden_states=True,
            )
        frames = output.hidden_states[self.layer][0].float().cpu().numpy()
        times = np.arange(len(frames)) * FRAME_S
        return times, frames

    def measure_segment(self, times, frames, start_s: float, end_s: float):
        """Mean of every frame inside one vowel, or ``None`` if there are none.

        No edge trimming, unlike the cepstral path. A transformer frame already
        carries context from well beyond its own 20 ms, so discarding the edges
        removes signal rather than consonant contamination.
        """
        if end_s <= start_s or len(frames) == 0:
            return None
        first = max(int(start_s / FRAME_S), 0)
        last = min(max(int(end_s / FRAME_S), first + 1), len(frames))
        if first >= len(frames):
            return None
        return list(frames[first:last].mean(axis=0))
