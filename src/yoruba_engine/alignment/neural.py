"""Forced alignment without MFA, at a fraction of the cost.

MFA is 13.79 s of an assessment's 14.31 s. None of that is alignment: it is
corpus setup, a SQLite database and feature extraction, charged once per
invocation. Measured, no combination of its flags moves it -- ``--single_speaker``,
``--num_jobs 1`` and ``--no_use_mp`` all land within a second of 13.8 s and
leave boundaries bit-identical. A tool built to align a corpus overnight cannot
be talked into aligning one utterance quickly.

This is the same architecture MFA uses -- an acoustic model scoring frames, then
Viterbi against the phone sequence the prompt requires -- with the GMM replaced
by the wav2vec2 encoder this project already trains and a 2 MB classification
head. Measured against MFA's own boundaries on 2,000 utterances from 26 held-out
speakers:

    within 20 ms    86.2%
    median          10.0 ms
    p90             20.0 ms
    over 100 ms      1.2%

10 ms is the floor rather than a result: wav2vec2 steps 20 ms where MFA works at
10, so half a frame is the best reachable without changing the frame rate.

**MFA is the teacher, so MFA is the ceiling.** The head was trained on MFA's
alignments; nobody hand-labelled them. This reproduces MFA cheaply. It cannot
align better than what it imitates, and MFA is still what trains acoustic models
and what would supervise any retraining of this head. What changes is that it is
no longer called per request.

**GOP is a different quantity here.** MFA reports a GMM log-likelihood over each
phone's span; this reports the mean log-posterior of the aligned phone. Both say
"how well did the audio fit this phone", but they are on different scales, so
``gop_reference.json`` -- which stores per-phone means and standard deviations
measured from MFA -- does not transfer. Until it is rebuilt, ``collect_gop``
returns scores that are internally consistent and not comparable with the
shipped reference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..datatypes import Alignment, Interval
from ..exceptions import AlignmentError
from ..g2p.dictionary import grapheme_to_phonemes
from ..g2p.orthography import is_noise_token, normalize, tokenize

logger = logging.getLogger(__name__)

#: One encoder frame per 20 ms.
FRAME_S = 0.02
SAMPLE_RATE = 16000

#: The label the head predicts for non-speech. Optional between words: a learner
#: may pause or may not, and forcing a silence that is not there would steal
#: frames from the words either side.
SILENCE = "sil"


def _strip_tone(symbol: str) -> str:
    """``a_H`` becomes ``a``.

    The G2P emits tone-suffixed symbols; the acoustic model was trained
    tone-stripped, because MFCCs discard F0 and a GMM cannot see the difference
    anyway. Tone is measured from pitch, separately.
    """
    return symbol.split("_", 1)[0]


def viterbi(log_posteriors: np.ndarray, sequence: np.ndarray,
            optional: np.ndarray, min_frames: int = 1) -> np.ndarray:
    """Align a known phone sequence to frames, monotonically.

    Each phone occupies a contiguous run, runs appear in order, and every frame
    belongs to one -- so this cannot invent, drop or reorder a phone, which is
    what makes forced alignment forced rather than decoding.

    ``optional`` marks states that may be skipped entirely, which is how silence
    between words is handled: MFA models it as optional, and requiring it would
    force a frame of silence into every word gap whether the speaker paused or
    not.

    ``min_frames`` is a floor on how long a phone may last, imposed by expanding
    each phone into that many sub-states of which only the last may repeat.
    Without it a phone can collapse into a single 20 ms frame, where MFA's
    three-state HMM cannot go below 30 ms -- measured, that produced 22% of
    vowels under 40 ms against MFA's 12%, and a vowel pooled from one frame
    gives the vowel classifier almost nothing to read.

    Args:
        log_posteriors: ``(frames, classes)`` log probabilities.
        sequence: Class index per state, in pronunciation order.
        optional: Boolean per state; True means the state may be skipped.
        min_frames: Minimum duration of a non-optional phone, in frames.

    Returns:
        The frame index at which each state starts. A skipped optional state
        takes the start of the state that follows it, giving it zero duration.
    """
    frames = len(log_posteriors)
    states = len(sequence)

    # Expand: a phone needing d frames becomes d sub-states, of which only the
    # last may repeat, so any path through it spends at least d frames there.
    classes: list[int] = []
    owner: list[int] = []
    can_stay: list[bool] = []
    skippable: list[bool] = []
    for position, (label, is_optional) in enumerate(zip(sequence, optional)):
        wanted = 1 if is_optional else max(int(min_frames), 1)
        for step in range(wanted):
            classes.append(int(label))
            owner.append(position)
            can_stay.append(step == wanted - 1)
            skippable.append(bool(is_optional))

    expanded = len(classes)
    if frames < expanded:
        raise AlignmentError(
            f"{frames} frames cannot hold {states} phones at a {min_frames}-frame "
            "minimum; the recording is shorter than the text requires")

    emission = log_posteriors[:, np.array(classes)]
    stay_allowed = np.array(can_stay)
    can_skip = np.array(skippable)

    scores = np.full((frames, expanded), -np.inf, dtype=np.float32)
    # 0 = stayed, 1 = came from the previous state, 2 = skipped one.
    came_from = np.zeros((frames, expanded), dtype=np.int8)

    scores[0, 0] = emission[0, 0]
    if expanded > 1 and can_skip[0]:
        scores[0, 1] = emission[0, 1]
        came_from[0, 1] = 1

    for t in range(1, frames):
        stay = np.where(stay_allowed, scores[t - 1], -np.inf)
        advance = np.concatenate(([-np.inf], scores[t - 1, :-1]))
        skip = np.concatenate(([-np.inf, -np.inf], scores[t - 1, :-2]))
        if expanded > 2:
            # Only legal where the state jumped over is a skippable one.
            jumped = np.concatenate(([False, False], can_skip[:-2]))
            skip = np.where(jumped, skip, -np.inf)
        stacked = np.vstack([stay, advance, skip])
        choice = np.argmax(stacked, axis=0)
        scores[t] = stacked[choice, np.arange(expanded)] + emission[t]
        came_from[t] = choice

    # Walk back through the expanded lattice, then collapse to real phones.
    starts_expanded = np.zeros(expanded, dtype=np.int64)
    position = expanded - 1
    t = frames - 1
    while t > 0 and position > 0:
        step = came_from[t, position]
        if step == 1:
            starts_expanded[position] = t
            position -= 1
        elif step == 2:
            starts_expanded[position] = t
            starts_expanded[position - 1] = t      # skipped: zero duration
            position -= 2
        t -= 1

    starts = np.zeros(states, dtype=np.int64)
    seen = set()
    for expanded_index, phone_index in enumerate(owner):
        if phone_index not in seen:
            starts[phone_index] = starts_expanded[expanded_index]
            seen.add(phone_index)
    return starts


class NeuralAligner:
    """Aligns one recording against one transcript, in about 0.2 s.

    Interface-compatible with the part of :class:`~yoruba_engine.alignment.mfa.MFAAligner`
    the pipeline uses, so swapping them is a constructor change.
    """

    def __init__(
        self,
        model_path: Path | str = "models/frame_aligner.pt",
        encoder_path: Path | str | None = None,
        device: str | None = None,
        min_frames: int = 3,
    ) -> None:
        self.model_path = Path(model_path)
        #: Floor on phone duration, in 20 ms frames. Chosen by measuring tone
        #: accuracy on 200 held-out utterances rather than by matching MFA's
        #: 30 ms HMM floor, which is what suggested two: 1 gives 70.4%, 2 gives
        #: 72.3%, 3 gives 73.1%, against MFA's 74.0%.
        self.min_frames = min_frames
        self._encoder_override = Path(encoder_path) if encoder_path else None
        self._device = device
        self._head = None
        self._encoder = None
        self._phones: list[str] = []
        self._torch = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    def _load(self):
        if self._head is not None:
            return
        try:
            import torch  # noqa: PLC0415
            from transformers import Wav2Vec2Model  # noqa: PLC0415
        except ImportError as exc:
            raise AlignmentError(
                "Neural alignment needs torch and transformers:\n"
                "  pip install torch transformers") from exc
        if not self.available:
            raise AlignmentError(
                f"No frame aligner at {self.model_path}. Train one with:\n"
                "  python scripts/train_frame_aligner.py")

        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self._phones = list(checkpoint["phones"])
        encoder_path = self._encoder_override or Path(checkpoint["encoder"])
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        # The head must be the one trained on this encoder's output. The
        # checkpoint names its encoder for the same reason the vowel classifier
        # does: a mismatched pair loads without complaint and produces confident
        # nonsense.
        self._encoder = Wav2Vec2Model.from_pretrained(
            encoder_path, torch_dtype=dtype).to(device).eval()
        head = torch.nn.Sequential(
            torch.nn.Linear(checkpoint["width"], checkpoint["hidden"]),
            torch.nn.GELU(), torch.nn.Dropout(0.0),
            torch.nn.Linear(checkpoint["hidden"], len(self._phones)),
        )
        head.load_state_dict(checkpoint["state_dict"])
        self._head = head.to(device).eval()
        self._torch = torch
        self._resolved_device = device
        self._dtype = dtype
        logger.info("Loaded frame aligner %s with encoder %s on %s",
                    self.model_path, encoder_path, device)

    def _states(self, transcript: str):
        """The phone sequence to align, with optional silence around each word.

        Returns the class indices, whether each may be skipped, and which word
        each belongs to -- the last is what makes word boundaries recoverable
        from a phone alignment.
        """
        index_of = {phone: position for position, phone in enumerate(self._phones)}
        if SILENCE not in index_of:
            raise AlignmentError(f"The head has no {SILENCE!r} class")

        sequence: list[int] = []
        optional: list[bool] = []
        owner: list[int] = []          # word index, or -1 for silence

        def add_silence():
            sequence.append(index_of[SILENCE])
            optional.append(True)
            owner.append(-1)

        add_silence()
        words = [w for w in tokenize(normalize(transcript), keep_noise=False)
                 if not is_noise_token(w)]
        if not words:
            raise AlignmentError(f"Nothing to align in {transcript!r}")

        for position, word in enumerate(words):
            phonemes = grapheme_to_phonemes(word)
            for phoneme in phonemes:
                label = _strip_tone(phoneme.symbol)
                if label not in index_of:
                    raise AlignmentError(
                        f"Phone {label!r} from {word!r} is not in the aligner's "
                        f"inventory of {len(self._phones)}")
                sequence.append(index_of[label])
                optional.append(False)
                owner.append(position)
            add_silence()

        return (np.array(sequence, dtype=np.int64), np.array(optional, dtype=bool),
                owner, words)

    def align_single(
        self,
        audio_path: Path,
        transcript: str,
        utterance_id: str | None = None,
        keep_workdir: bool = False,       # accepted for interface parity
        collect_gop: bool = False,
    ) -> Alignment:
        """Align one recording against a known transcript."""
        from ..audio import read_mono  # noqa: PLC0415

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise AlignmentError(f"Audio file not found: {audio_path}")
        self._load()
        torch = self._torch
        utt_id = utterance_id or audio_path.stem

        # read_mono resamples with a filter. Reading raw samples and assuming
        # 16 kHz put one speaker's boundaries 360 ms out during training,
        # because 2.2% of this project's corpus is 48 kHz.
        samples = read_mono(audio_path, SAMPLE_RATE)
        if len(samples) < SAMPLE_RATE // 10:
            raise AlignmentError(f"{audio_path.name} is too short to align")

        with torch.inference_mode():
            batch = torch.tensor(samples, dtype=self._dtype).unsqueeze(0)
            hidden = self._encoder(batch.to(self._resolved_device)).last_hidden_state
            logits = self._head(hidden[0].to(torch.float32))
            log_posteriors = torch.log_softmax(logits, -1).cpu().numpy()

        sequence, optional, owner, words = self._states(transcript)
        starts = viterbi(log_posteriors, sequence, optional, self.min_frames)
        ends = np.append(starts[1:], len(log_posteriors))

        phones: list[Interval] = []
        scores: list = []
        for position, (state, start, end) in enumerate(zip(sequence, starts, ends)):
            if end <= start:
                continue          # a skipped optional silence
            label = self._phones[state]
            interval = Interval(label=label, start_s=float(start) * FRAME_S,
                                end_s=float(end) * FRAME_S)
            phones.append(interval)
            if collect_gop and label != SILENCE:
                from .gop import PhoneScore  # noqa: PLC0415

                # The mean log-posterior of the aligned phone. Not on MFA's
                # scale -- see the module docstring.
                scores.append(PhoneScore(
                    phone=label, start_s=interval.start_s, end_s=interval.end_s,
                    goodness=float(log_posteriors[start:end, state].mean())))

        word_intervals: list[Interval] = []
        for position, word in enumerate(words):
            spans = [(start, end) for own, start, end in zip(owner, starts, ends)
                     if own == position and end > start]
            if not spans:
                continue
            word_intervals.append(Interval(
                label=word,
                start_s=float(min(s for s, _ in spans)) * FRAME_S,
                end_s=float(max(e for _, e in spans)) * FRAME_S))

        logger.debug("%s: %d words, %d phones from %d frames",
                     utt_id, len(word_intervals), len(phones), len(log_posteriors))
        return Alignment(utterance_id=utt_id, words=word_intervals,
                         phones=phones, phone_scores=scores)

    def check_available(self, require_acoustic_model: bool = True) -> None:
        """Interface parity with MFAAligner; raises if the head is missing."""
        if not self.available:
            raise AlignmentError(
                f"No frame aligner at {self.model_path}. Train one with:\n"
                "  python scripts/train_frame_aligner.py")
