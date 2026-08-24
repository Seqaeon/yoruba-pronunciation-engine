"""End-to-end pipeline: audio in, pronunciation report out.

This module is the single seam a FastAPI layer should sit on. A request handler
constructs :class:`AssessmentPipeline` once at startup and calls
:meth:`AssessmentPipeline.assess` per request.

Typical use::

    pipeline = AssessmentPipeline(DEFAULT_CONFIG)
    result = pipeline.assess(
        reference_audio=Path("data/corpus/wavs/ref_0001.wav"),
        learner_audio=Path("uploads/learner.wav"),
        target_text="òrí",
    )
"""

from __future__ import annotations

from pathlib import Path

from .alignment.durations import WordDurationModel
from .alignment.mfa import MFAAligner
from .asr.providers import PromptedTranscriptProvider, TranscriptProvider
from .compare.engine import ComparisonEngine
from .g2p.orthography import normalize
from .config import DEFAULT_CONFIG, EngineConfig
from .exceptions import AlignmentError, PitchExtractionError
from .logging_utils import get_logger
from .pitch.extractor import PitchExtractor
from .report.reporter import ReportGenerator
from .tone.analyzer import ToneAnalyzer
from .datatypes import ComparisonResult, UtteranceAnalysis


def _audio_duration(path: Path) -> float | None:
    """Duration of an audio file in seconds, or ``None`` if unreadable."""
    import wave  # noqa: PLC0415

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                return handle.getnframes() / rate if rate else None
        except Exception:  # noqa: BLE001 - fall through to soundfile
            pass

    try:
        import soundfile  # noqa: PLC0415

        return float(soundfile.info(str(path)).duration)
    except Exception:  # noqa: BLE001 - duration is optional
        return None

logger = get_logger("pipeline")


def _default_aligner(config):
    """The neural aligner when its head is present, MFA otherwise.

    MFA costs 13.79 s of a 14.31 s assessment, all of it per-invocation setup
    that no flag reduces. The neural aligner costs 0.56 s and produces the same
    word statuses, the same vowel flags, and boundaries agreeing with MFA's on
    86.2% of edges within 20 ms.

    It is not free. Measured on 200 held-out utterances against the tone the
    corpus spelling specifies, MFA's spans yield 74.0% tone accuracy and the
    neural aligner's 73.1%. That 0.9 points buys a 29x speedup, and it is small
    against the ceiling: the tone classifier's own held-out accuracy is 75.9%,
    so both aligners already run within three points of what the classifier can
    do. The tone headroom is in the classifier -- 49,531 vowels from 36
    speakers, against the vowel classifier's 575,125 from 2,786 -- not in the
    alignment.

    MFA remains what trains acoustic models and what supervises this head. What
    changed is that it is no longer called per request.
    """
    from .alignment.neural import NeuralAligner  # noqa: PLC0415

    neural = NeuralAligner()
    if neural.available:
        logger.info("Aligning with %s (MFA is used for training, not requests)",
                    neural.model_path)
        return neural
    logger.info("No frame aligner at %s; falling back to MFA, which costs about "
                "13.8 s per utterance", neural.model_path)
    return MFAAligner(config)


def _tone_only_difference(prompted: str, decoded: str) -> bool:
    """Whether two spellings differ in nothing but tone marks.

    Sub-dots are kept: ``ọ`` and ``o`` are different vowels, and the decoder can
    hear that. Tone marks are not, and it cannot.
    """
    from .g2p.orthography import normalize, strip_tones  # noqa: PLC0415

    return strip_tones(normalize(prompted)) == strip_tones(normalize(decoded))


class AssessmentPipeline:
    """Orchestrates Stages 3 through 7 for a single assessment request."""

    def __init__(
        self,
        config: EngineConfig | None = None,
        transcript_provider: TranscriptProvider | None = None,
        use_tone_classifier: bool | None = None,
        use_vowel_verifier: bool | None = None,
        aligner=None,
    ) -> None:
        """
        Args:
            config: Engine configuration; defaults to :data:`DEFAULT_CONFIG`.
            transcript_provider: How learner transcripts are obtained. Defaults
                to :class:`PromptedTranscriptProvider`, which requires that the
                target text be passed to :meth:`assess`.
            use_tone_classifier: Whether to use the trained tone classifier.
                ``None`` (the default) uses it when a model file exists and
                falls back to the threshold rules otherwise. ``True`` requires
                it and raises if it is missing; ``False`` forces the rules.
            use_vowel_verifier: Whether to check vowel identity, which catches
                substitutions such as ``òkan`` for ``òkun``. Same policy:
                ``None`` uses it when a model exists.
            aligner: Where phone and word boundaries come from. Defaults to
                :class:`~yoruba_engine.alignment.mfa.MFAAligner`. Pass a
                :class:`~yoruba_engine.alignment.neural.NeuralAligner` to spend
                0.04 s per utterance instead of 13.79 s; anything implementing
                ``align_single`` will do.
        """
        self.config = config or DEFAULT_CONFIG
        self.aligner = aligner if aligner is not None else _default_aligner(self.config)
        self.pitch_extractor = PitchExtractor(self.config.pitch)
        self.tone_analyzer = ToneAnalyzer(
            self.config.tone, classifier=self._load_tone_classifier(use_tone_classifier)
        )
        self.comparison_engine = ComparisonEngine(self.config.scoring, self.config.tone)
        self.reporter = ReportGenerator()
        self.duration_model = WordDurationModel.load(
            fraction=self.config.scoring.min_duration_fraction
        )
        self.vowel_verifier = self._load_vowel_verifier(use_vowel_verifier)
        self.transcript_provider = transcript_provider or PromptedTranscriptProvider()
        self.word_decoder = self._load_word_decoder(self.vowel_verifier)
        self.gop_reference = self._load_gop_reference()
        self._reference_store = self._build_reference_cache()


    def _build_reference_cache(self):
        """A cache for reference analyses, or ``None`` if disabled."""
        directory = getattr(self.config, "reference_cache_dir", None)
        if directory is None:
            directory = Path("output") / "reference_cache"
        if not directory:
            return None
        from .cache import ReferenceCache  # noqa: PLC0415

        pitch = self.config.pitch
        return ReferenceCache.build(
            directory=Path(directory),
            acoustic_model=self.config.mfa.acoustic_model,
            dictionary=self.config.mfa.dictionary,
            # Pitch settings decide where a vowel starts and how its tone reads,
            # so a cached analysis is only valid for the settings that made it.
            pitch_settings=(f"{pitch.time_step_s}:{pitch.edge_trim_ratio}:"
                            f"{pitch.min_voiced_frames}:{pitch.floor_hz}:"
                            f"{pitch.ceiling_hz}"),
        )

    @staticmethod
    def _load_tone_classifier(use_tone_classifier: bool | None):
        """Load the trained tone classifier according to the requested policy.

        Three implementations, best first, each falling back to the next when its
        model or its dependencies are missing. None of them may be fatal by
        default: tone is the engine's whole point, and a missing file should
        degrade the reading rather than refuse to give one.

        The sequence model reads a whole utterance at once and reaches 79.0% on
        held-out speakers where the per-unit tree reaches 69.6%. That gap is
        representational rather than informational -- eight times more data moved
        the tree 69.6% to 70.2%, because Yoruba tone is relative and a model
        seeing one unit plus three lag features cannot represent declination.

        The threshold rules remain below both, and remain the floor that works
        with no model file and no torch at all.
        """
        if use_tone_classifier is False:
            return None

        from .tone.classifier import DEFAULT_MODEL_PATH, ToneClassifier  # noqa: PLC0415
        from .tone.sequence import ToneSequenceClassifier  # noqa: PLC0415

        sequence = ToneSequenceClassifier.load()
        if sequence is not None:
            return sequence

        if use_tone_classifier is None and not DEFAULT_MODEL_PATH.exists():
            logger.info(
                "No tone classifier at %s; using threshold rules. Train one with "
                "scripts/train_tone_classifier.py for better accuracy.",
                DEFAULT_MODEL_PATH,
            )
            return None

        try:
            return ToneClassifier.load()
        except Exception as exc:
            if use_tone_classifier is True:
                raise
            logger.warning("Could not load tone classifier (%s); using threshold rules", exc)
            return None

    def analyse_utterance(
        self,
        audio_path: Path,
        transcript: str,
        utterance_id: str | None = None,
    ) -> UtteranceAnalysis:
        """Run alignment, pitch extraction, and tone detection on one recording.

        Args:
            audio_path: The recording to analyse.
            transcript: Its (diacritized) transcript.
            utterance_id: Identifier; defaults to the audio file stem.

        Returns:
            The full per-word, per-vowel analysis.

        Raises:
            AlignmentError: If forced alignment fails.
            PitchExtractionError: If no pitch could be recovered.
        """
        utt_id = utterance_id or audio_path.stem
        logger.info("Analysing %s", utt_id)

        alignment = self.aligner.align_single(
            audio_path, transcript, utterance_id=utt_id,
            collect_gop=self.config.gop.enabled and self.gop_reference is not None,
        )
        words, median_hz = self.pitch_extractor.extract(audio_path, alignment)
        words = self.tone_analyzer.analyse(words)

        duration = _audio_duration(audio_path)
        speech = sum(w.end_s - w.start_s for w in words)
        coverage = speech / duration if duration else None

        return UtteranceAnalysis(
            utterance_id=utt_id,
            audio_path=audio_path,
            transcript=transcript,
            words=words,
            speaker_median_hz=median_hz,
            audio_duration_s=duration,
            speech_coverage=coverage,
            phone_scores=alignment.phone_scores,
        )

    def _analyse_reference(
        self, audio_path: Path, transcript: str, utterance_id: str,
    ) -> UtteranceAnalysis:
        """Analyse the reference, reusing the previous result when nothing moved.

        The reference for a prompt is the same file with the same transcript on
        every assessment, but it was being force-aligned each time -- 17.6
        seconds of a 43-second request, 41% of the work, recomputed to the same
        answer. The learner's recording still gets the full treatment; only the
        half that cannot have changed is reused.
        """
        cache = self._reference_store
        if cache is None:
            return self.analyse_utterance(audio_path, transcript,
                                          utterance_id=utterance_id)

        key = cache.key(audio_path, transcript)
        cached = cache.get(key)
        if cached is not None:
            logger.info("Reusing cached reference analysis for %s", utterance_id)
            return cached

        analysis = self.analyse_utterance(audio_path, transcript,
                                          utterance_id=utterance_id)
        cache.put(key, analysis)
        return analysis

    def assess(
        self,
        reference_audio: Path | None,
        learner_audio: Path,
        target_text: str,
        learner_text: str | None = None,
        utterance_id: str | None = None,
    ) -> ComparisonResult:
        """Compare a learner recording against a native reference.

        Args:
            reference_audio: Native-speaker recording, or ``None`` to build the
                reference from ``target_text`` alone. An application offering
                arbitrary phrases cannot record a native speaker for each, and
                Yoruba orthography specifies the tones, so the spelling is a
                sufficient target.
            learner_audio: Learner recording.
            target_text: The diacritized text both speakers were producing.
            learner_text: What the learner actually said, if known. When omitted
                it is obtained from the configured transcript provider, which by
                default returns ``target_text`` (prompted-practice mode).
            utterance_id: Identifier for the report.

        Returns:
            The scored comparison with feedback.
        """
        # With no reference recording the learner's file is the only thing that
        # varies between assessments, so it names the report. A constant here
        # meant every text-only assessment wrote prompt_report.txt over the
        # last one.
        utt_id = utterance_id or (
            reference_audio.stem if reference_audio is not None
            else learner_audio.stem)

        # Does this recording contain speech at all? Asked before anything else,
        # because everything downstream assumes it does. The recogniser
        # hallucinates a word from noise, the aligner lays the prompt's phones
        # across it, and pitch is measured from a hum -- producing a score for a
        # learner who said nothing. It also fails outright on long prompts: two
        # seconds of hum against `Èyí ni Igba` raised "99 frames cannot hold 216
        # phones", a hallucinated transcript shown to the learner as an error.
        silent = self._reject_if_silent(learner_audio, target_text, utt_id)
        if silent is not None:
            return silent

        if reference_audio is None:
            # No native recording for this prompt. The comparison never needed
            # one: it reads the reference's words, their vowel counts and the
            # transcript, and takes tone targets from the spelling rather than
            # from a detector run over the reference. All of that is in the
            # text. See yoruba_engine.textref.
            from .textref import reference_from_text  # noqa: PLC0415

            reference = reference_from_text(target_text, f"{utt_id}_ref")
        else:
            reference = self._analyse_reference(
                reference_audio, target_text, utterance_id=f"{utt_id}_ref"
            )

        # A recogniser earns its place by naming the word a learner actually
        # said, which is the only way the engine can report a substitution. That
        # is worth most on a one-word drill -- the whole point of the igba set is
        # that five different words are spelled almost alike -- so recognition
        # stays on however short the prompt is.
        #
        # It was briefly gated off below three words, on the evidence that a
        # one-second `ìgbà` came back as `bí ìgbà tẹ bá`. That evidence was
        # wrong: the recording genuinely contained all four words, because the
        # clip had been cut badly. The decoder was right and the gate was
        # removed. What a short clip does cause is the milder failure handled
        # just below.
        recognise = (self.transcript_provider.recognizes_speech
                     and len(normalize(target_text).split())
                     >= self.config.scoring.min_words_to_recognise)
        if learner_text is None and not recognise:
            learner_text = target_text
        elif learner_text is None:
            learner_text = self.transcript_provider.transcribe(learner_audio, prompt=target_text)
            if self.transcript_provider.recognizes_speech:
                # Yoruba is written with spaces the speech does not have, so a
                # recogniser's word boundaries are guesswork -- 'Ẹ káàbọ̀' comes
                # back as 'ekaabo' or 'e ka bo'. Everything downstream pairs
                # prompt words with heard ones, so a mis-split makes a correct
                # attempt read as words not said. The prompt already fixes the
                # word count; only the sounds are in question.
                from .asr.segment import resegment_to_prompt  # noqa: PLC0415

                respaced = resegment_to_prompt(target_text, learner_text)
                if respaced != learner_text:
                    logger.info("Re-spaced transcript %r as %r",
                                learner_text, respaced)
                    learner_text = respaced

                # Half a second of audio does not constrain a decoder trained on
                # thirty-second windows, and it fills the silence with common
                # function words: the igba reference clips came back as `tí gbà`
                # and `mo ní ìgbà` for recordings a native speaker confirms
                # contain one word. Scored literally, those extra tokens are
                # insertions, and a correct attempt lands at zero.
                #
                # On a one-word prompt the fix is available without weakening
                # anything: exactly one of the heard tokens is the attempt at
                # that word, so keep the closest one and drop the padding. The
                # substitution check survives, because the token kept is
                # whatever was actually said -- `ìgbá` for `ìgbà` is still a
                # different word and still reported. Only the decoder's filler
                # is discarded, and only when the prompt cannot contain filler.
                heard_words = normalize(learner_text).split()
                if len(normalize(target_text).split()) == 1 and len(heard_words) > 1:
                    from .compare.engine import phrase_resemblance  # noqa: PLC0415

                    # Two different failures look alike here and must not be
                    # treated alike. `ìgbà` heard as `ti gbà` is one word split
                    # in two, and both halves are the learner's audio. `ìgbà`
                    # heard as `mo ní ìgbà` is one word plus decoder filler, and
                    # only the last token is the learner's audio.
                    #
                    # Resemblance separates them cleanly, because a split still
                    # resembles the prompt as a whole while filler does not:
                    #
                    #     ti gbà      whole 0.89   best token 0.86   -> keep both
                    #     mo ní ìgbà  whole 0.67   best token 1.00   -> keep one
                    #
                    # Keeping the whole of a split also keeps its vowel count,
                    # which the tone check needs: dropping `ti` leaves one vowel
                    # against a two-vowel prompt and the tone score collapses.
                    closest = max(heard_words,
                                  key=lambda w: phrase_resemblance(target_text, w))
                    if phrase_resemblance(target_text, closest) > phrase_resemblance(
                            target_text, learner_text):
                        logger.info("Kept %r from %r for the one-word prompt %r",
                                    closest, learner_text, target_text)
                        learner_text = closest

        # What gets aligned is not always what gets reported. Alignment needs the
        # phone sequence the audio actually contains; the transcript is what the
        # learner is told they were heard to say.
        #
        # They come apart on a one-word prompt the recogniser mis-spells. `ìgbà`
        # heard as `ti gbà` is two words to the aligner, so it lays two words of
        # phones across one word of audio, every vowel span lands in the wrong
        # place, and tone accuracy goes to zero on a recording that is correct.
        # When the recognition resembles the prompt closely enough to be an
        # attempt at it, the prompt is the better phone sequence -- it is what a
        # native speaker confirms is there -- so align against that.
        #
        # Nothing is hidden by this. The heard text still sets the text score and
        # still names a substitution; tone comes from pitch and vowel identity
        # from the classifier, neither of which reads the spelling. And a learner
        # who said a different word does not reach here: their recognition would
        # not resemble the prompt.
        align_text = learner_text
        if (recognise and learner_text
                and len(normalize(target_text).split()) == 1
                and normalize(learner_text) != normalize(target_text)):
            from .compare.engine import phrase_resemblance  # noqa: PLC0415

            if (phrase_resemblance(target_text, learner_text)
                    >= self.config.scoring.min_phrase_resemblance):
                logger.info("Aligning %s against the prompt %r rather than the "
                            "heard %r", utt_id, target_text, learner_text)
                align_text = target_text

        try:
            learner = self.analyse_utterance(
                learner_audio, align_text, utterance_id=f"{utt_id}_learner"
            )
        except PitchExtractionError:
            # No voiced frames anywhere in the recording. That is the same
            # finding as the silence check above, arrived at by a better
            # detector -- pitch tracking answers "is anyone speaking" more
            # reliably than the energy range does, which matters because a
            # browser's automatic gain control lifts room noise to nearly full
            # scale: a recording of nothing measured 16.9 dB of range and peaked
            # at -0.9 dBFS, well clear of the energy threshold.
            #
            # It used to escape as an exception naming a temporary file, which
            # the app showed to the learner.
            logger.warning("Utterance %s has no voiced frames; treating it as "
                           "no speech", utt_id)
            return self._no_speech_result(
                utt_id, target_text,
                "No voice was detected in the recording -- it sounds silent, "
                "too noisy, or whispered.")
        if align_text != learner_text:
            learner.transcript = learner_text

        dropped: list[str] = []
        mispronounced: list[str] = []
        substitutions: dict[str, str] = {}
        # GOP scores the acoustic model's fit to whatever text was aligned, so
        # it is meaningful in both modes. It was previously computed only in
        # prompted mode, which left it unbound the first time a recognising
        # provider ran -- the branch had never been exercised, because until
        # Whisper arrived every provider echoed the prompt back.
        poorly = self._check_pronunciation(learner)
        if not recognise:
            # Decode before dropping anything: a word the learner replaced is
            # still fully present in the audio, and calling it missing when we
            # can name what was said instead is the wrong feedback.
            substitutions = self._decode_words(learner)
            dropped = self._drop_unspoken_words(learner)
            mispronounced = self._check_vowel_identity(learner)
        else:
            # In recognised mode the transcript already says which words were
            # said, so omissions and substitutions come from comparing it with
            # the prompt rather than from alignment quality. The vowel check
            # still runs: it reports how a word that *was* said was produced.
            mispronounced = self._check_vowel_identity(learner)
            self._defer_to_vowels(learner, target_text, mispronounced, poorly)

            # A word that was not produced cannot have been said differently.
            #
            # Decoding runs first on purpose: a substituted word IS present in
            # the audio, so dropping before decoding would report it as an
            # omission. But the same ordering lets the decoder name a word in a
            # slot where nothing was spoken -- forced alignment still assigns
            # that slot some sliver of audio, and the decoder dutifully scores
            # it. Observed after the margin was lowered to 2.0: a recording that
            # omitted 'bíi' entirely was told "you said 'bòó'", which is a
            # confident claim about silence.
            #
            # The duration check is the authority on presence, so its verdict
            # overrides the decoder's here.
            #
            # Match on the decoded *value*, not the prompt word: _decode_words
            # has already renamed the word in place, so the duration check sees
            # and reports the decoded name. A first attempt at this matched on
            # the key and silently did nothing.
            absent = {normalize(word) for word in dropped} | set(dropped)
            for prompted, decoded in list(substitutions.items()):
                if decoded in absent or normalize(decoded) in absent:
                    substitutions.pop(prompted)
                    dropped = [w for w in dropped if w not in (decoded, prompted)]
                    dropped.append(prompted)
                    logger.info("%s: withdrew decoded substitution %r -> %r; the "
                                "duration check says nothing was produced there",
                                learner.utterance_id, prompted, decoded)

        # A recogniser's spelling of a correctly spoken word need not match the
        # prompt's: Whisper wrote ``bí`` for ``bíi``, which is not an error the
        # learner made. The prompted path keeps exact matching, where both
        # sequences come from the same text and any difference is real.
        vowel_accuracy, vowel_errors = self._score_vowel_identity(learner, target_text)
        result = self.comparison_engine.compare(
            reference, learner,
            tolerate_spelling=recognise,
            vowel_accuracy=vowel_accuracy)
        # Words carrying a confident vowel error join the ones the per-word check
        # flagged, so the report marks them CHECK VOWELS as well as scoring them.
        mispronounced = list(dict.fromkeys(list(mispronounced) + vowel_errors))
        result.utterance_id = utt_id
        result.undetected_words = dropped
        result.mispronounced_words = mispronounced
        result.substitutions = substitutions
        result.poorly_pronounced = poorly
        # Report what was actually determined, never the prompt echoed back.
        # A word we could not find is shown as [?]; one we decoded to a
        # different real word is shown as that word.
        if not self.transcript_provider.recognizes_speech:
            derived = []
            for token in result.learner_transcript.split():
                bare = normalize(token.strip(".,;:!?"))
                if bare in {normalize(w) for w in dropped}:
                    derived.append("[?]")
                elif bare in substitutions:
                    derived.append(substitutions[bare])
                elif bare in {normalize(w) for w in mispronounced} or \
                        bare in {normalize(w) for w in poorly}:
                    # GOP earns a marker of its own. At 30 ms a vowel
                    # measurement cannot distinguish "said u" from "said an in a
                    # span the aligner mangled" -- both read as u with high
                    # confidence -- so the vowel check is silent on both. GOP
                    # scores the whole word against the acoustic model and
                    # separates them: -4.8 for the genuinely wrong word, nothing
                    # for the one whose only error was tonal.
                    derived.append(f"{token}?")
                else:
                    derived.append(token)
            result.learner_transcript = " ".join(derived)
        if substitutions:
            # The alignment was driven by the prompt, so learner.transcript is
            # the prompt. Rewrite it from what the decoder actually concluded --
            # that is the whole point of deriving a transcript rather than
            # echoing one back.
            # Match on the normalized form: the transcript carries punctuation
            # ("òkun.") while the substitution keys are bare words.
            rewritten = []
            for token in result.learner_transcript.split():
                replacement = substitutions.get(normalize(token.strip(".,;:!?")))
                rewritten.append(replacement if replacement else token)
            result.learner_transcript = " ".join(rewritten)
        self._verify_attempt(result, learner)

        if not self.transcript_provider.provides_tone_marks and learner_text != target_text:
            result.feedback.append(
                "Note: what you said was recognised automatically. Tone is "
                "measured from your pitch, not read from that transcript, but "
                "a misheard word shifts which sounds are compared, so tone "
                "scoring is approximate here."
            )

        self._reference_cache = reference  # exposed for report rendering
        self._learner_cache = learner
        return result

    def _load_gop_reference(self):
        """Load per-phone reference statistics for the GOP check.

        Absent, the check is simply skipped: a raw likelihood with nothing to
        compare it against cannot say whether a phone was well produced.
        """
        if not self.config.gop.enabled:
            return None
        from .alignment.gop import GOPReference  # noqa: PLC0415

        # Which reference depends on which aligner produced the scores. MFA
        # reports a GMM log-likelihood, typically -4 to -5; the neural aligner
        # reports a mean log-posterior, bounded above by zero and around -0.2.
        # Measured on the same phones: a -4.78 against a -0.11. Using one
        # reference for the other's scores would flag or excuse words for
        # arithmetic reasons, so the pairing is chosen rather than assumed.
        from .alignment.neural import NeuralAligner  # noqa: PLC0415

        name = ("gop_reference_neural.json"
                if isinstance(self.aligner, NeuralAligner) else "gop_reference.json")
        path = self.config.mfa.acoustic_model.parent / name
        if not path.exists():
            logger.info("No GOP reference at %s; the check is skipped", path)
            return None
        reference = GOPReference.load(path)
        if not reference.statistics:
            return None
        logger.info("Loaded GOP reference for %d phones", len(reference.statistics))
        return reference

    def _load_word_decoder(self, vowel_verifier):
        """Build the minimal-pair decoder, or None if its inputs are missing.

        It needs both the vowel classifier and the pronunciation dictionary. Its
        absence is never fatal -- without it the engine still reports omissions
        and tone errors, it simply cannot name what was said instead.
        """
        if vowel_verifier is None or not self.config.mfa.dictionary.exists():
            return None
        try:
            from .segmental.decoder import WordDecoder  # noqa: PLC0415

            return WordDecoder.load(self.config.mfa.dictionary,
                                    classifier=vowel_verifier.classifier)
        except Exception as exc:
            logger.info("Word decoder unavailable: %s", exc)
            return None

    @staticmethod
    def _load_vowel_verifier(use_vowel_verifier: bool | None):
        """Load the vowel-identity verifier according to the requested policy.

        Its absence must never be fatal: without it the engine still detects
        omissions and tone errors, it simply cannot catch substitutions.
        """
        if use_vowel_verifier is False:
            return None

        from .segmental.classifier import DEFAULT_MODEL_PATH, VowelClassifier  # noqa: PLC0415
        from .segmental.verifier import VowelVerifier  # noqa: PLC0415

        if use_vowel_verifier is None and not DEFAULT_MODEL_PATH.exists():
            logger.info(
                "No vowel classifier at %s; substitutions will not be detected. "
                "Train one with scripts/train_vowel_classifier.py",
                DEFAULT_MODEL_PATH,
            )
            return None

        try:
            return VowelVerifier(VowelClassifier.load())
        except Exception as exc:
            if use_vowel_verifier is True:
                raise
            logger.warning("Could not load vowel classifier (%s); skipping "
                           "substitution detection", exc)
            return None

    def _check_pronunciation(self, learner: UtteranceAnalysis) -> dict[str, float]:
        """Score each word against the acoustic model's own phone likelihoods.

        This is the only check that can flag a word the learner invented. A
        recording of ``àbùbùtu`` against a prompt of ``àbùbùtán`` scores -4.98
        here, with one vowel at -10.6; the decoder cannot name it because
        ``àbùbùtu`` is in no dictionary, and the vowel classifier only reports
        per-vowel probability without a word-level verdict.

        It is deliberately *not* trusted on minimal pairs. ``ọkàn`` read against
        ``òkun`` scores +0.25, better than average, because forced alignment
        already moved the boundaries to fit. Those belong to the classifier.
        """
        import statistics as stats  # noqa: PLC0415

        if self.gop_reference is None or not learner.phone_scores:
            return {}

        scores = sorted(learner.phone_scores, key=lambda p: p.start_s)
        flagged: dict[str, float] = {}
        for word in learner.words:
            inside = [p for p in scores
                      if word.start_s - 0.02 <= p.start_s and p.end_s <= word.end_s + 0.02]
            values = [z for z in (self.gop_reference.z_score(p) for p in inside)
                      if z is not None]
            if len(values) < self.config.gop.min_scored_phones:
                continue
            mean = stats.mean(values)
            if mean < self.config.gop.threshold:
                flagged[word.word] = round(mean, 2)

        if flagged:
            logger.info("%s: %d words scored poorly by the acoustic model: %s",
                        learner.utterance_id, len(flagged), flagged)
        return flagged

    def _decode_words(self, learner: UtteranceAnalysis) -> dict[str, str]:
        """Name the words the learner said instead of the ones prompted.

        Forced alignment writes the prompt onto the audio whatever was spoken,
        so ``learner.words`` carries the prompt's spelling by construction. Where
        the measured vowels support a different *real* Yoruba word by a wide
        enough margin, the word is rewritten here, which makes the comparison
        score a substitution as a substitution and lets the report say "you said
        ọkàn" instead of "òkun missing".

        Silence is the default. Measured against 2,801 words of correctly-read
        corpus speech, a decoder that always named its best guess contradicted
        32.5% of them; at the calibrated margin that falls to 2.7%. Anything
        below the margin leaves the prompt's spelling in place.
        """
        if self.word_decoder is None or self.vowel_verifier is None or not learner.words:
            return {}

        segmental = self.vowel_verifier.verify(learner.audio_path, learner.words)
        if not segmental.distributions:
            return {}

        substitutions: dict[str, str] = {}
        for index, word in enumerate(learner.words):
            distributions = segmental.distributions.get(index)
            if not distributions:
                continue
            verdict = self.word_decoder.decode(normalize(word.word), distributions)
            # The decoder reads vowel identity, so it cannot hear tone -- and a
            # word differing from the prompt only in tone marks has, by
            # definition, the same vowels. Naming one is not a measurement, it is
            # the decoder picking between candidates it cannot distinguish.
            #
            # `ìgbá` (garden egg) and `ìgbà` (time) differ in one accent. The
            # decoder reported SAID DIFFERENTLY on a correct reading of `ìgbá`,
            # which is precisely backwards in an exercise built to teach that
            # tone carries the meaning. A tonal error belongs to the tone
            # report, which reads pitch and can actually see it.
            if verdict.substituted and _tone_only_difference(word.word, verdict.decoded):
                logger.info("%s: not naming %r for %r; they differ only in tone, "
                            "which the vowel decoder cannot read",
                            learner.utterance_id, verdict.decoded, word.word)
                continue
            if verdict.substituted:
                substitutions[word.word] = verdict.decoded
                logger.info("%s: %r decoded as %r (margin %+.2f)",
                            learner.utterance_id, word.word, verdict.decoded,
                            verdict.margin)
                word.word = verdict.decoded
        return substitutions

    def _score_vowel_identity(self, learner: UtteranceAnalysis,
                              target_text: str) -> tuple[float | None, list[str]]:
        """How many of the prompt's vowels the learner actually produced.

        The existing vowel check verifies each word against *its own* spelling,
        which with a recogniser is the word that was heard. That answers "is this
        transcript self-consistent" and cannot answer "did you say the vowel the
        prompt asked for" -- a learner reading ``dàbọ̀`` whose ``ọ`` came out as
        ``o`` was transcribed ``dàbò``, and the check happily confirmed the
        vowels matched ``dàbò``.

        So this compares against the prompt's spelling instead, which is where
        the required vowels are written.

        **Only confident disagreements count.** The classifier is 93.1% accurate,
        so roughly one vowel in fourteen is wrong; deducting on every objection
        would tax correct speech more than it corrects wrong speech, which is why
        vowel flags have always advised rather than scored. A vowel counts as an
        error only when the classifier puts the required vowel below
        ``max_confident_vowel_error`` while backing something else above
        ``min_confident_vowel``. On the ``dàbọ̀`` recording that is o=0.91
        against ọ=0.01 -- not a marginal call.

        Returns:
            The share of judged vowels that were produced as written, or ``None``
            when nothing could be judged, and the words carrying a confident
            error.
        """
        if self.vowel_verifier is None or not learner.words:
            return None, []

        from .g2p.dictionary import grapheme_to_phonemes  # noqa: PLC0415
        from .g2p.orthography import is_noise_token, normalize, tokenize  # noqa: PLC0415

        segmental = self.vowel_verifier.verify(learner.audio_path, learner.words)
        if not segmental.distributions:
            return None, []
        index_of = getattr(self.vowel_verifier, "_label_index", None)
        if not index_of:
            return None, []

        prompt_words = [w for w in tokenize(normalize(target_text), keep_noise=False)
                        if not is_noise_token(w)]
        scoring = self.config.scoring
        judged = correct = 0
        wrong: list[str] = []

        for position, word in enumerate(learner.words):
            distributions = segmental.distributions.get(position)
            if not distributions or position >= len(prompt_words):
                continue
            try:
                phonemes = grapheme_to_phonemes(prompt_words[position])
            except Exception:
                continue
            expected = [p.symbol.split("_", 1)[0] for p in phonemes if p.is_vowel]
            if len(expected) != len(distributions):
                # The prompt and the aligned word disagree about how many vowels
                # there are, so nothing can be compared position by position.
                continue

            for label, distribution in zip(expected, distributions):
                slot = index_of.get(label)
                if slot is None:
                    continue
                judged += 1
                required = float(distribution[slot])
                best = max(range(len(distribution)), key=distribution.__getitem__)
                if (required < scoring.max_confident_vowel_error
                        and float(distribution[best]) >= scoring.min_confident_vowel
                        and best != slot):
                    if word.word not in wrong:
                        wrong.append(word.word)
                else:
                    correct += 1

        if not judged:
            return None, []
        if wrong:
            logger.info("%s: %d of %d prompt vowels confidently not produced (%s)",
                        learner.utterance_id, judged - correct, judged, wrong)
        return correct / judged, wrong

    def _defer_to_vowels(self, learner: UtteranceAnalysis, target_text: str,
                         mispronounced: list[str],
                         poorly: dict[str, float] | None = None) -> None:
        """Restore a prompt word the recogniser altered and the acoustics reject.

        Two components disagree about one slot, and the engine was believing the
        wrong one. A learner reading ``ó dàbọ̀`` was transcribed ``ó dàbò`` --
        the sub-dot lost, which changes the vowel from /ɔ/ to /o/ -- and scored
        50% on a two-word phrase. But the vowel check *also* rejected ``dàbò``,
        which means the audio does not support what the recogniser heard either.

        When that happens the honest reading is that the transcript is wrong,
        not the learner. The vowel classifier is measured at 93.1% on vowel
        identity, and the recogniser's diacritics are documented as drifting, so
        on this specific question the classifier is the better witness.

        Only near-misses are restored: the two spellings must agree once tone
        marks and sub-dots are removed, so ``dàbò`` reverts to ``dàbọ̀`` while
        ``ọkàn`` never reverts to ``òkun``. A genuine substitution differs by
        more than diacritics and keeps being reported as one.

        The word still carries CHECK VOWELS. What changes is that it stops
        counting as a wrong *word* when the only evidence it was wrong is a
        transcript the acoustics contradict.
        """
        import unicodedata  # noqa: PLC0415

        from .compare.engine import _collapse_repeats  # noqa: PLC0415
        from .g2p.orthography import normalize, strip_tones, tokenize  # noqa: PLC0415

        def bare(word: str) -> str:
            """Spelling with tone marks and sub-dots removed."""
            decomposed = unicodedata.normalize("NFD", strip_tones(normalize(word)))
            return _collapse_repeats(
                "".join(c for c in decomposed if not unicodedata.combining(c)).lower())

        # Either check counts. The report marks a word CHECK VOWELS when the
        # vowel classifier objects *or* the acoustic model fitted it badly, and
        # consulting only the first meant a word flagged by GOP alone kept the
        # recogniser's spelling and kept costing the learner marks -- which is
        # exactly what `dàbọ̀` heard as `dàbò` did.
        flagged = set(mispronounced) | set(poorly or {})
        if not flagged:
            return
        prompt_words = tokenize(normalize(target_text), keep_noise=False)
        restored = []
        for position, word in enumerate(learner.words):
            if word.word not in flagged or position >= len(prompt_words):
                continue
            expected = prompt_words[position]
            if word.word != expected and bare(word.word) == bare(expected):
                restored.append((word.word, expected))
                word.word = expected
        if not restored:
            return

        # The transcript has to agree. Restoring the word only in the alignment
        # left the screen contradicting itself: a table row reading `dàbọ̀ ok`
        # above a line reading "We heard: ó dàbò". If the engine has concluded
        # the recogniser dropped a sub-dot, it should not keep quoting the
        # spelling it just rejected.
        swap = {heard: want for heard, want in restored}
        learner.transcript = " ".join(
            swap.get(token, token) for token in learner.transcript.split())

        logger.info(
            "%s: the vowel check rejected the recogniser's spelling, so "
            "%s reverted to the prompt's", learner.utterance_id,
            ", ".join(f"{heard!r}->{want!r}" for heard, want in restored))

    def _check_vowel_identity(self, learner: UtteranceAnalysis) -> list[str]:
        """Remove words the learner appears to have said as a different word.

        Alignment and duration only reveal *omissions*: a substituted word of
        similar length occupies plausible time and passes both. Comparing the
        vowels actually produced against the ones the prompt requires is what
        catches it.

        Flagged words are dropped from the learner side so the comparison scores
        them as errors, and returned so the report can name them as
        substitutions rather than omissions.
        """
        if self.vowel_verifier is None or not learner.words:
            return []

        result = self.vowel_verifier.verify(learner.audio_path, learner.words)
        if not result.mispronounced:
            return []

        flagged = set(result.mispronounced)
        names = [w.word for i, w in enumerate(learner.words) if i in flagged]
        # Neither delete nor rename. Deleting made the report say "you did not
        # say this", which is the wrong correction for a learner who said
        # something -- just not the right thing. Renaming broke the word
        # alignment, because the comparison matches on word text and a altered
        # label no longer matches its own reference word.
        #
        # So the word stays exactly as it is and the flag travels separately,
        # in the returned list. The score is only penalised where a
        # substitution could actually be NAMED -- _decode_words rewrites those,
        # and the comparison scores them as differences on its own. A vowel
        # that merely looks wrong earns a "check this" note, not a mark-down:
        # at 6.1% false alarms, penalising every flag would tax correct speech
        # more than it corrects wrong speech.

        logger.info("%s: %d words pronounced differently: %s",
                    learner.utterance_id, len(names), names)
        return names

    def _drop_unspoken_words(self, learner: UtteranceAnalysis) -> list[str]:
        """Remove prompt words the learner did not actually produce.

        Without a recognizer the learner's "transcript" is the prompt, so text
        similarity would be 100% by construction and would measure nothing. This
        recovers a real measurement from the alignment instead.

        Forced alignment must place every word of the prompt somewhere in the
        audio. A word the learner skipped therefore does not vanish -- it gets
        compressed into a sliver between its neighbours.

        The threshold scales with word length, because duration does: a one-phone
        word is legitimately 120 ms while an eight-phone word runs to 780 ms. A
        flat threshold would either reject common short words like ``ti`` and the
        syllabic nasal ``ń``, or let a long word crushed to a tenth of its normal
        length pass as spoken. See
        :class:`~yoruba_engine.alignment.durations.WordDurationModel`.

        This catches **omissions, not substitutions**: if the learner said a
        different word of similar length in the same place, it still occupies
        plausible time and passes. Open-vocabulary substitution detection needs a
        real recognizer -- see
        :class:`~yoruba_engine.asr.providers.MMSTranscriptProvider`.

        Args:
            learner: The analysed learner utterance. Modified in place.

        Returns:
            The words judged not to have been spoken.
        """
        kept = []
        dropped = []

        for word in learner.words:
            duration = word.end_s - word.start_s
            phone_count = word.phone_count or max(len(word.vowels) * 2, 1)
            if self.duration_model.was_spoken(duration, phone_count):
                kept.append(word)
            else:
                dropped.append(word.word)

        if dropped:
            logger.info(
                "%s: %d prompt words were not detected in the audio: %s",
                learner.utterance_id, len(dropped), dropped[:6],
            )
            learner.words = kept

        return dropped

    def _verify_attempt(self, result: ComparisonResult, learner: UtteranceAnalysis) -> None:
        """Check the learner actually produced the target text, and flag it if not.

        Forced alignment cannot fail: given any audio and any text it returns an
        alignment. In prompted mode the learner's transcript *is* the target
        text, so without this check a learner who said something entirely
        different is compared against the target and scores 100%.

        **How that is checked depends on what evidence exists.**

        When a recogniser supplied the transcript, the question is answered
        directly -- word overlap between what was heard and what was prompted.
        That is what recognition is for.

        Otherwise the only evidence is circumstantial: how much of the recording
        the aligned words occupy. When the text matches the audio the words
        account for most of it; when it does not, the aligner crams them into a
        fraction and leaves the rest as silence. Measured on 120 utterances
        aligned both ways, that proxy reaches 72% balanced accuracy at best and
        catches 12% of mismatches at the shipped threshold, so a passing result
        means "not obviously wrong", not "correct".

        Coverage is recorded either way, because it is worth reporting even when
        it is not what decided the outcome.
        """
        result.learner_speech_coverage = learner.speech_coverage

        if self.transcript_provider.recognizes_speech:
            self._verify_by_recognition(result)
        else:
            self._verify_by_coverage(result)

        # Whatever the text said, the audio still has to fit the prompt's
        # phones. These run last and can only tighten the verdict, never loosen
        # one already refused.
        if result.verified:
            self._verify_by_goodness(result, learner)
        if result.verified:
            self._verify_one_word_fills_recording(result)

    def _verify_one_word_fills_recording(self, result: ComparisonResult) -> None:
        """Reject a one-word prompt that occupies little of the recording.

        The text checks are blind here: `igbà` against room noise transcribed as
        `gbàgbá` resembles the prompt 0.60, over the 0.55 bar, because short
        strings share characters by chance. Goodness does not catch it either,
        because the aligner places the word in whichever half-second fits best
        and marks the rest silence -- so the phones it scores really do fit.

        What gives it away is how little of the recording that half-second is.
        See :attr:`~yoruba_engine.config.ScoringConfig.min_one_word_coverage`.
        """
        if len(normalize(result.reference_transcript).split()) != 1:
            return

        # Not when the recogniser heard the prompt word itself. Low coverage
        # then means the learner left silence around a word they said correctly,
        # which is the reason the recognising path ignores coverage in the first
        # place. Only an inexact recognition is asked to also fill the
        # recording, and every false accept measured had one: `gbélá`, `gba`,
        # `bá`, `agbóná` against the prompt `igbà`.
        heard = normalize(result.learner_transcript or "")
        if heard and heard == normalize(result.reference_transcript):
            return

        coverage = result.learner_speech_coverage
        if coverage is None or coverage >= self.config.scoring.min_one_word_coverage:
            return

        result.verified = False
        result.verification_note = (
            f"The prompt is one word, but it accounts for only {coverage:.0%} of "
            "the recording. That usually means something else was said, or "
            "nothing was.")
        self._withhold_score(result)
        logger.warning("Utterance %s failed verification: one-word coverage %.2f",
                       result.utterance_id, coverage)

    def _verify_by_goodness(self, result: ComparisonResult,
                            learner: UtteranceAnalysis) -> None:
        """Reject an attempt whose audio does not fit the prompt's phones.

        Forced alignment places the prompt on any recording it is given; it
        cannot make it fit. GOP measures that fit against how native speakers
        produce the same phones, which is the question the text checks cannot
        answer on a short prompt -- see
        :attr:`~yoruba_engine.config.ScoringConfig.min_mean_gop_z`.
        """
        reference = getattr(self, "gop_reference", None)
        if reference is None or not learner.phone_scores:
            return

        # Against the *prompt*, not against whatever was aligned.
        #
        # Wiring this to learner.phone_scores first was useless: when the
        # recogniser's text is what got aligned -- which is exactly the case
        # where the learner said something else -- GOP measures how well the
        # audio fits the words they actually said, and that fits well. It
        # rejected 2 of 100 wrong-prompt recordings.
        #
        # The question is whether the audio fits *this prompt*, so the prompt is
        # what has to be aligned to ask it. The extra alignment costs about
        # 40 ms with the neural aligner.
        scores = learner.phone_scores
        prompt = result.reference_transcript
        if normalize(learner.transcript or "") != normalize(prompt):
            try:
                against_prompt = self.aligner.align_single(
                    learner.audio_path, prompt,
                    utterance_id=f"{result.utterance_id}_fit", collect_gop=True)
            except Exception:
                return          # cannot ask the question; do not answer it
            if not against_prompt.phone_scores:
                return
            scores = against_prompt.phone_scores

        values = [z for z in (reference.z_score(score) for score in scores)
                  if z is not None]
        if len(values) < self.config.gop.min_scored_phones:
            return

        import statistics as stats  # noqa: PLC0415

        mean = stats.mean(values)
        result.mean_gop_z = round(mean, 2)
        if mean >= self.config.scoring.min_mean_gop_z:
            return

        result.verified = False
        result.verification_note = (
            f"The recording does not sound like this phrase: the acoustic model "
            f"rates its fit at {mean:.2f} against native production, where "
            f"{self.config.scoring.min_mean_gop_z:.2f} is the least it accepts.")
        self._withhold_score(result)
        logger.warning("Utterance %s failed verification: mean GOP z %.2f",
                       result.utterance_id, mean)

    def _verify_by_recognition(self, result: ComparisonResult) -> None:
        """Reject an attempt that does not sound like the prompt at all.

        Deliberately not gated on word agreement. A learner whose every word
        came back wrong is the learner this app exists for, and telling them
        "we could not confirm you said the prompt" is both false and useless
        feedback. What disqualifies an attempt is sounding like a *different*
        phrase, which character-level resemblance separates cleanly and word
        agreement does not.
        """
        from .compare.engine import phrase_resemblance  # noqa: PLC0415

        resemblance = phrase_resemblance(
            result.reference_transcript, result.learner_transcript)
        result.learner_phrase_resemblance = resemblance
        if resemblance >= self.config.scoring.min_phrase_resemblance:
            return

        result.verified = False
        result.verification_note = (
            f"What we heard does not sound like this phrase: it came back as "
            f"{result.learner_transcript!r}, which resembles the prompt "
            f"{resemblance:.0%} (expected at least "
            f"{self.config.scoring.min_phrase_resemblance:.0%}). The scores "
            "below describe a forced fit rather than a real match."
        )
        self._withhold_score(result)
        logger.warning(
            "Utterance %s failed verification: phrase resemblance %.2f (%r vs %r)",
            result.utterance_id, resemblance,
            result.reference_transcript, result.learner_transcript,
        )

    def _verify_by_coverage(self, result: ComparisonResult) -> None:
        """Reject an attempt whose aligned words occupy too little of the audio."""
        coverage = result.learner_speech_coverage
        if coverage is None or coverage >= self.config.scoring.min_speech_coverage:
            return

        result.verified = False
        result.verification_note = (
            f"The aligned words account for only {coverage:.0%} of the recording "
            f"(expected at least {self.config.scoring.min_speech_coverage:.0%}). "
            "The learner probably said something other than the prompt, so the "
            "scores below describe a forced fit rather than a real match."
        )
        self._withhold_score(result)
        logger.warning(
            "Utterance %s failed verification: coverage %.2f",
            result.utterance_id, coverage,
        )

    def _reject_if_silent(self, learner_audio: Path, target_text: str,
                          utt_id: str) -> ComparisonResult | None:
        """A withheld result when the recording holds no speech, else ``None``.

        Speech is loud in syllables and quiet between them; steady noise is not.
        The measure is the 95th minus the 5th percentile of frame energy in
        decibels, which is invariant to how loud the recording was made. See
        :attr:`~yoruba_engine.config.ScoringConfig.min_dynamic_range_db` for the
        measured distributions behind the threshold.

        Returns a result rather than raising, so that saying nothing is scored
        zero with an explanation like any other unverified attempt, instead of
        reaching the app as an error.
        """
        import numpy as np  # noqa: PLC0415

        from .audio import read_mono  # noqa: PLC0415

        try:
            samples = read_mono(learner_audio, 16000)
        except Exception:
            return None                  # let the normal path report the problem

        window = 320                     # 20 ms at 16 kHz
        frames = len(samples) // window
        if frames < 10:
            # Under 200 ms. This used to return None and let the recording
            # through, on the grounds that it was too short to characterise --
            # which handed the shortest recordings a free pass through the one
            # check meant to catch an empty one. A fifth of a second cannot
            # hold a Yoruba word: the engine's own single-word reference clips
            # run 380 to 600 ms.
            logger.warning("Utterance %s is %.2fs, too short to contain the "
                           "prompt", utt_id, len(samples) / 16000)
            return self._no_speech_result(
                utt_id, target_text,
                f"The recording is only {len(samples) / 16000:.1f} seconds long, "
                "which is too short to hold the phrase.")

        energy = np.sqrt(np.mean(
            samples[:frames * window].reshape(frames, window) ** 2, axis=1)) + 1e-12
        level = 20 * np.log10(energy)
        dynamic_range = float(np.percentile(level, 95) - np.percentile(level, 5))
        if dynamic_range >= self.config.scoring.min_dynamic_range_db:
            return None

        logger.warning("Utterance %s has %.1f dB of dynamic range, below %.1f: "
                       "treating it as no speech", utt_id, dynamic_range,
                       self.config.scoring.min_dynamic_range_db)
        return self._no_speech_result(
            utt_id, target_text,
            f"The recording has {dynamic_range:.1f} dB between its loudest and "
            "quietest moments, where speech has far more. It sounds like "
            "nothing was said.")

    def _no_speech_result(self, utt_id: str, target_text: str,
                          note: str) -> ComparisonResult:
        """A zero-scored result for a recording with nothing in it."""
        result = ComparisonResult(
            utterance_id=utt_id,
            reference_transcript=normalize(target_text),
            learner_transcript="",
            text_similarity=0.0,
            tone_similarity=0.0,
            overall_score=0.0,
        )
        result.verified = False
        result.verification_note = note
        self._withhold_score(result)
        result.feedback = [
            "We could not hear any speech in that recording.",
            "Check your microphone is working and that you are recording in a "
            "reasonably quiet place, then say the phrase aloud.",
        ]
        return result

    @staticmethod
    def _withhold_score(result: ComparisonResult) -> None:
        """Zero the score and say what to do, without presenting a forced fit.

        This used to zero ``overall_score`` alone and leave the components as
        they were, which meant an unverified recording still reported a tone
        percentage -- and the aligner will always place the prompt's vowels
        somewhere, so there is always a percentage to report. Recording two
        seconds of appliance hum came back as 50% tone, and a conversation in
        another room as 67%: partial credit for saying nothing.

        A forced fit is not a measurement of the learner, so none of it is
        published. The components are zeroed *and* marked inapplicable, because
        a bare 0% would say the learner got every tone wrong, which is a
        different and equally false claim.
        """
        result.overall_score = 0.0
        result.text_similarity = 0.0
        result.text_similarity_applicable = False
        result.tone_similarity = 0.0
        result.tone_similarity_applicable = False
        result.vowel_accuracy = None
        # Tone errors from a forced fit name vowels the learner never produced.
        result.tone_errors = []
        result.feedback = [
            "We could not confirm you said the prompt. Listen to the reference "
            "again and repeat just that phrase.",
            "If you did say it, try recording in a quieter place, and avoid long "
            "pauses or extra words before and after.",
        ]

    def assess_and_report(
        self,
        reference_audio: Path | None,
        learner_audio: Path,
        target_text: str,
        output_dir: Path | None = None,
        **kwargs,
    ) -> tuple[ComparisonResult, dict[str, Path]]:
        """Run :meth:`assess` and write text and JSON reports.

        Returns:
            The result, and a mapping of format name to written path.
        """
        result = self.assess(reference_audio, learner_audio, target_text, **kwargs)
        out = output_dir or self.config.ensure_output_dir()

        render_kwargs = {
            "reference": getattr(self, "_reference_cache", None),
            "learner": getattr(self, "_learner_cache", None),
        }
        paths = {
            "text": self.reporter.write_text(
                result, out / f"{result.utterance_id}_report.txt", **render_kwargs
            ),
            "json": self.reporter.write_json(
                result, out / f"{result.utterance_id}_report.json", **render_kwargs
            ),
        }
        return result, paths

    def health_check(self) -> dict[str, object]:
        """Report which dependencies are installed and which assets exist.

        Intended as the backing implementation of a ``/health`` endpoint.
        """
        status: dict[str, object] = {}

        try:
            self.aligner.check_available()
            status["mfa"] = {"available": True, "version": self.aligner.version()}
        except Exception as exc:
            status["mfa"] = {"available": False, "reason": str(exc)}

        try:
            import parselmouth  # noqa: PLC0415, F401
            status["parselmouth"] = {"available": True}
        except ImportError:
            status["parselmouth"] = {
                "available": False,
                "reason": "pip install praat-parselmouth (fallback tracker in use)",
            }

        status["dictionary"] = {"exists": self.config.mfa.dictionary.exists()}
        status["acoustic_model"] = {"exists": self.config.mfa.acoustic_model.exists()}
        classifier = self.tone_analyzer.classifier
        status["tone_classifier"] = (
            {"loaded": True, **{k: v for k, v in classifier.metadata.items()
                                if k != "holdout_speakers"}}
            if classifier is not None
            else {"loaded": False, "reason": "using threshold rules "
                  "(train with scripts/train_tone_classifier.py)"}
        )
        verifier = self.vowel_verifier
        status["vowel_verifier"] = (
            {"loaded": True, **{k: v for k, v in verifier.classifier.metadata.items()
                                if k not in ("features",)}}
            if verifier is not None
            else {"loaded": False,
                  "reason": "substitutions will not be detected "
                            "(train with scripts/train_vowel_classifier.py)"}
        )
        status["transcript_provider"] = {
            "class": type(self.transcript_provider).__name__,
            "provides_tone_marks": self.transcript_provider.provides_tone_marks,
        }
        return status
