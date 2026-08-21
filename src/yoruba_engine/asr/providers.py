"""Transcript providers -- the placeholder layer for "what did the learner say?".

The project brief assumes a learner transcript exists but does not say where it
comes from. There are two fundamentally different answers, and the choice drives
whether you need to train or license an ASR model at all:

1. **Prompted practice** (:class:`PromptedTranscriptProvider`) -- the app already
   knows what the learner was asked to say, so the target text *is* the
   transcript. Forced alignment then verifies whether they actually said it.
   No ASR, no model training, and it covers the majority of language-app
   interactions. **This is the default.**

2. **Free speech** (:class:`ASRTranscriptProvider`) -- the learner says something
   unprompted and we must recognise it. This needs a Yoruba ASR model. The
   concrete backend is deliberately left unimplemented: plug in MMS, a
   fine-tuned w2v-BERT, or a hosted API behind this interface.

Note that Yoruba ASR output is usually *undiacritized*, which makes it unusable
for tone comparison on its own. See :class:`ASRTranscriptProvider` for how to
handle that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..exceptions import ASRNotConfiguredError, TranscriptError
from ..g2p.orthography import has_tone_marks, normalize
from ..logging_utils import get_logger

logger = get_logger("asr")


@runtime_checkable
class TranscriptProvider(Protocol):
    """Supplies the text corresponding to a recording."""

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        """Return the transcript for ``audio_path``.

        Args:
            audio_path: The recording.
            prompt: The text the speaker was asked to produce, when known.

        Returns:
            Normalized Yoruba text, ideally with tone marks.
        """
        ...

    @property
    def provides_tone_marks(self) -> bool:
        """Whether output carries tone diacritics, and so supports tone comparison."""
        ...

    @property
    def recognizes_speech(self) -> bool:
        """Whether the transcript reflects what the speaker actually said.

        ``False`` for prompted practice, where the transcript is the prompt
        echoed back and carries no information about the recording. Text
        similarity is meaningless in that mode and the engine excludes it from
        scoring rather than awarding a free 100%.
        """
        ...


class PromptedTranscriptProvider:
    """Uses the known target text as the transcript.

    Appropriate whenever the app displays a word or sentence and asks the learner
    to repeat it. Correctness of the *content* is then established by the forced
    aligner: if the learner said something else, alignment quality collapses and
    the comparison stage scores it accordingly.
    """

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        """Return the prompt text.

        Raises:
            ASRNotConfiguredError: If no prompt was supplied, since this provider
                has no way to recognise unprompted speech.
        """
        if prompt is None:
            raise ASRNotConfiguredError(
                f"PromptedTranscriptProvider needs the target text for {audio_path.name}. "
                "Pass prompt=..., or configure an ASR provider for free speech."
            )
        return normalize(prompt)

    @property
    def provides_tone_marks(self) -> bool:
        """True, because prompts come from the diacritized reference corpus."""
        return True

    @property
    def recognizes_speech(self) -> bool:
        """False: this returns the prompt, not what the learner produced."""
        return False


class ASRTranscriptProvider:
    """Placeholder for a real Yoruba speech recognizer.

    To implement, subclass and override :meth:`_recognize`. Candidate backends,
    roughly in increasing order of effort:

    * **MMS** (``facebook/mms-1b-all``) -- has a Yoruba adapter, zero training,
      but output is undiacritized.
    * **Fine-tuned w2v-BERT / MMS** on your own diacritized corpus -- needs a GPU
      and several hours of labelled audio, but can learn to emit tone marks
      directly, which is what makes tone comparison possible on free speech.
    * **A hosted ASR API** with Yoruba support -- least work, ongoing cost, and
      check whether it returns diacritics before relying on it.

    Args:
        model_name: Identifier passed to whatever backend you wire up.
        emits_tone_marks: Set ``True`` only if your model was trained on
            diacritized text. Leave ``False`` and the pipeline will
            automatically skip tone scoring rather than report garbage.
    """

    def __init__(self, model_name: str = "unset", emits_tone_marks: bool = False) -> None:
        self.model_name = model_name
        self._emits_tone_marks = emits_tone_marks

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        """Recognize speech, warning if the result lacks tone marks."""
        text = normalize(self._recognize(audio_path))
        if text and not has_tone_marks(text):
            logger.warning(
                "ASR output for %s has no tone marks; tone comparison will be "
                "skipped for this utterance.",
                audio_path.name,
            )
        return text

    def _recognize(self, audio_path: Path) -> str:
        """Backend hook. Override this.

        Raises:
            ASRNotConfiguredError: Always, in the base class.
        """
        raise ASRNotConfiguredError(
            f"No ASR backend is wired up (model_name={self.model_name!r}). "
            "Subclass ASRTranscriptProvider and implement _recognize(), or use "
            "PromptedTranscriptProvider for prompted practice."
        )

    @property
    def provides_tone_marks(self) -> bool:
        return self._emits_tone_marks

    @property
    def recognizes_speech(self) -> bool:
        """True: a real recognizer reports what was actually said."""
        return True


class StaticTranscriptProvider:
    """Returns transcripts from an in-memory mapping. Intended for tests."""

    def __init__(self, transcripts: dict[str, str], provides_tone_marks: bool = True) -> None:
        self.transcripts = {k: normalize(v) for k, v in transcripts.items()}
        self._provides_tone_marks = provides_tone_marks

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        key = audio_path.stem
        if key not in self.transcripts:
            raise ASRNotConfiguredError(f"No stored transcript for {key!r}")
        return self.transcripts[key]

    @property
    def provides_tone_marks(self) -> bool:
        return self._provides_tone_marks

    @property
    def recognizes_speech(self) -> bool:
        """True: stored transcripts stand in for real recognition in tests."""
        return True


class MMSTranscriptProvider:
    """Yoruba speech recognition using Meta's MMS wav2vec2 model.

    MMS ships a Yoruba adapter, so this needs no training -- only a ~4 GB
    download on first use. It runs on GPU when one is available and on CPU
    otherwise, a few seconds per utterance either way. The model is loaded once
    per process, not per request.

    **MMS output carries no tone marks.** That is fine for the brief's first
    question, "did both speakers say the same words", because word comparison is
    deliberately tone-insensitive. It is *not* a source of tone information, and
    the engine never treats it as one: tone is always measured acoustically.

    Diacritics are restored only so the forced aligner can look words up in the
    pronunciation dictionary, which is keyed on diacritized forms. Pass a
    :class:`~yoruba_engine.asr.restore.DiacriticRestorer`; without one the
    recognized words are out of vocabulary and the audio cannot be aligned.

    Args:
        model_name: Hugging Face model id.
        language: MMS language code; ``yor`` for Yoruba.
        device: ``"cuda"``, ``"cpu"``, or ``None`` to choose automatically.
        restorer: Diacritic restorer. Strongly recommended.
    """

    def __init__(
        self,
        model_name: str = "facebook/mms-1b-all",
        language: str = "yor",
        device: str | None = None,
        restorer: object | None = None,
    ) -> None:
        self.model_name = model_name
        self.language = language
        self.device = device
        self.restorer = restorer
        self._model = None
        self._processor = None

    def _load(self) -> None:
        """Load the model on first use, so importing this module stays cheap."""
        if self._model is not None:
            return

        try:
            import torch  # noqa: PLC0415
            from transformers import AutoProcessor, Wav2Vec2ForCTC  # noqa: PLC0415
        except ImportError as exc:
            raise ASRNotConfiguredError(
                "MMS recognition needs torch and transformers:\n"
                "  pip install torch transformers\n"
                "Or use PromptedTranscriptProvider, which needs no recognizer."
            ) from exc

        logger.info("Loading %s (%s); the first run downloads ~4 GB",
                    self.model_name, self.language)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, target_lang=self.language
        )
        self._model = Wav2Vec2ForCTC.from_pretrained(
            self.model_name, target_lang=self.language, ignore_mismatched_sizes=True
        )

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device).eval()
        logger.info("MMS ready on %s", self.device)

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        """Recognize the speech in ``audio_path``.

        ``prompt`` is ignored by design: this provider exists to report what was
        actually said, independently of what was asked for.

        Returns:
            Normalized Yoruba text, with diacritics restored when a restorer is
            configured.

        Raises:
            ASRNotConfiguredError: If dependencies are missing.
            TranscriptError: If the audio cannot be read.
        """
        self._load()

        import torch  # noqa: PLC0415

        samples, rate = self._read_audio(audio_path)
        inputs = self._processor(samples, sampling_rate=rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        predicted = torch.argmax(logits, dim=-1)[0]
        text = normalize(self._processor.decode(predicted.cpu().numpy()))

        if self.restorer is not None and text:
            text, unresolved = self.restorer.restore(text)
            if unresolved:
                logger.info("%d recognized words are outside the corpus vocabulary",
                            len(unresolved))

        logger.info("Recognized %s as %r", audio_path.name, text)
        return text

    @staticmethod
    def _read_audio(audio_path: Path):
        """Read audio as mono float32 at 16 kHz, which is what MMS expects."""
        try:
            import numpy as np  # noqa: PLC0415
            import soundfile  # noqa: PLC0415
        except ImportError as exc:
            raise ASRNotConfiguredError("soundfile and numpy are required") from exc

        try:
            samples, rate = soundfile.read(str(audio_path), dtype="float32", always_2d=False)
        except (RuntimeError, OSError) as exc:
            raise TranscriptError(f"Could not read {audio_path}: {exc}") from exc

        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)

        target = 16000
        if rate != target:
            # Linear resample: adequate for 16 kHz features, and avoids a hard
            # dependency on torchaudio or scipy.
            count = int(samples.shape[0] / rate * target)
            positions = np.linspace(0, samples.shape[0] - 1, count)
            samples = np.interp(
                positions, np.arange(samples.shape[0]), samples
            ).astype("float32")
            rate = target

        return samples, rate

    @property
    def provides_tone_marks(self) -> bool:
        """False. MMS emits undiacritized text and restored marks are guesses,
        so the engine never derives tone from this transcript."""
        return False

    @property
    def recognizes_speech(self) -> bool:
        return True


class WhisperTranscriptProvider:
    """Transcribes with a Whisper checkpoint, giving open-vocabulary text.

    Forced alignment cannot report what a learner said, only how badly the
    prompt fits: MFA takes the prompt as given and finds the best placement for
    it. Every substitution the engine names therefore has to be recovered
    indirectly, by matching measured vowels against the 9,092-entry dictionary
    -- which is why a learner reading ``òkun`` who says ``ọkun`` is flagged but
    never named. ``ọkun`` occurs zero times in 209,265 corpus tokens and is not
    in the lexicon, so no decoder margin can surface it.

    Whisper has no lexicon to be limited by. Measured on the project's four test
    recordings it named the substituted word every time, including ``ọkun``, and
    caught both omissions.

    It is weak where the engine is strong, and that is why this supplements
    rather than replaces the rest: its tone marks are unreliable (``àbùbùtàn``
    for ``àbùbùtán``, ``ọkún`` for ``ọkun``), and it produces no phone
    boundaries at all, so tone analysis and vowel spans still come from forced
    alignment and pitch.

    Cost is about 2.8 s per utterance on four CPU threads, roughly flat with
    duration because Whisper pads every input to 30 seconds.
    """

    def __init__(self, model_path: Path | str = "models/whisper-yoruba",
                 device: str | None = None, language: str = "yo") -> None:
        self.model_path = Path(model_path)
        self.language = language
        self._device = device
        self._model = None
        self._processor = None

    @property
    def recognizes_speech(self) -> bool:
        return True

    @property
    def provides_tone_marks(self) -> bool:
        """False -- it emits tone marks, but they cannot be trusted.

        Whisper writes diacritics, and gets the ones that change a vowel's
        identity right: it produced ``ọkàn`` and ``ọkún``, sub-dots intact. The
        tone marks themselves drift -- ``àbùbùtàn`` for ``àbùbùtán``, ``ọkún``
        for ``ọkun`` -- because tone is carried by pitch, which a transcriber
        infers only indirectly.

        This costs the engine nothing, because it does not read tone from the
        transcript. Expected tone comes from the prompt's spelling and produced
        tone from measured pitch, so Whisper's marks are discarded rather than
        corrected. Alignment is unaffected too: the acoustic model has no tone
        in its phones, so ``òkun`` and ``ókun`` yield the same phone sequence.

        What remains true is that a misrecognised *word* shifts which vowel is
        compared against which target, so the report says tone scoring is
        approximate in this mode.
        """
        return False

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                WhisperForConditionalGeneration, WhisperProcessor,
            )
        except ImportError as exc:
            raise ASRNotConfiguredError(
                "Whisper transcription needs torch and transformers:\n"
                "  pip install torch transformers"
            ) from exc
        if not (self.model_path / "config.json").exists():
            raise ASRNotConfiguredError(
                f"No Whisper checkpoint at {self.model_path}. Fetch it with:\n"
                "  python scripts/fetch_models.py"
            )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = WhisperProcessor.from_pretrained(self.model_path)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path).eval().to(device)
        self._torch = torch
        self._resolved_device = device
        logger.info("Loaded Whisper transcriber from %s on %s",
                    self.model_path, device)
        return self._model

    def transcribe(self, audio_path: Path, prompt: str | None = None) -> str:
        """Return what the recording actually contains, ignoring the prompt."""
        model = self._load()
        torch = self._torch
        try:
            import soundfile as sf  # noqa: PLC0415

            from ..audio import resample  # noqa: PLC0415

            samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        except Exception as exc:
            raise ASRNotConfiguredError(f"Could not read {audio_path}: {exc}") from exc
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            samples = resample(samples, rate, 16000)

        features = self._processor(samples, sampling_rate=16000,
                                   return_tensors="pt").input_features
        with torch.inference_mode():
            tokens = model.generate(features.to(self._resolved_device),
                                    max_new_tokens=128)
        text = self._processor.batch_decode(tokens, skip_special_tokens=True)[0]
        return normalize(text.strip())
