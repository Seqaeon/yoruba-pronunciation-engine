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
        self.aligner = aligner if aligner is not None else MFAAligner(self.config)
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

        The classifier is roughly 11 points more accurate than the threshold
        rules on held-out speakers, but it needs a model file and scikit-learn,
        so its absence must never be fatal in the default configuration.
        """
        if use_tone_classifier is False:
            return None

        from .tone.classifier import DEFAULT_MODEL_PATH, ToneClassifier  # noqa: PLC0415

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

        if learner_text is None:
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

        learner = self.analyse_utterance(
            learner_audio, learner_text, utterance_id=f"{utt_id}_learner"
        )

        dropped: list[str] = []
        mispronounced: list[str] = []
        substitutions: dict[str, str] = {}
        # GOP scores the acoustic model's fit to whatever text was aligned, so
        # it is meaningful in both modes. It was previously computed only in
        # prompted mode, which left it unbound the first time a recognising
        # provider ran -- the branch had never been exercised, because until
        # Whisper arrived every provider echoed the prompt back.
        poorly = self._check_pronunciation(learner)
        if not self.transcript_provider.recognizes_speech:
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
        result = self.comparison_engine.compare(
            reference, learner,
            tolerate_spelling=self.transcript_provider.recognizes_speech)
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
            if verdict.substituted:
                substitutions[word.word] = verdict.decoded
                logger.info("%s: %r decoded as %r (margin %+.2f)",
                            learner.utterance_id, word.word, verdict.decoded,
                            verdict.margin)
                word.word = verdict.decoded
        return substitutions

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

    @staticmethod
    def _withhold_score(result: ComparisonResult) -> None:
        """Zero the score and say what to do, without presenting a forced fit."""
        result.overall_score = 0.0
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
