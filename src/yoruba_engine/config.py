"""Central configuration.

All tunable thresholds live here rather than being scattered as literals, so a
future API layer can override them per-request (e.g. a more forgiving tone
threshold for absolute beginners).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PitchConfig:
    """Parameters for Praat pitch extraction.

    The defaults span the range actually measured in the training corpus:
    male speakers median ~133 Hz (90-175 observed), female ~237 Hz (111-311).
    Narrow further if you serve one demographic; it reduces octave errors.
    """

    floor_hz: float = 75.0
    ceiling_hz: float = 350.0
    #: Pitch analysis step. 5 ms rather than 10 because a vowel whose tone
    #: cannot be read is not a neutral outcome -- the comparison skips it, so a
    #: wrong tone there is never reported and the learner is silently excused.
    #:
    #: A learner recording of ``àbùbùtán`` aligned at 190, 30, 30 and 30 ms.
    #: After a 20% edge trim, a 30 ms vowel leaves 18 ms, which at a 10 ms step
    #: is one or two frames -- below ``min_voiced_frames`` -- so three quarters
    #: of the word was unjudgeable. Short vowels are ordinary in fluent speech.
    #:
    #: Swept against SLR86, where the diacritized spelling is ground truth:
    #:
    #:     step   trim  coverage  accuracy  coverage x accuracy
    #:     0.010  0.20    80.0%     75.9%        60.7%
    #:     0.005  0.20    92.9%     70.6%        65.6%   <- here
    #:     0.005  0.10    93.4%     70.6%        65.9%
    #:
    #: Coverage rises 13 points and accuracy falls 5, so more vowels are judged
    #: and a larger share of the judgements are wrong. That trade is taken
    #: because 20% of vowels going unjudged is how a wrong tone escapes
    #: entirely, and because ``ToneConfig.min_confidence`` already withholds
    #: shaky readings from the learner while still counting them in the score.
    #:
    #: ``min_voiced_frames`` was swept too and made no difference at 3 or 2: the
    #: binding constraint is how many frames the step and trim leave, not the
    #: threshold applied afterwards.
    time_step_s: float = 0.005
    #: Ignore the first and last N% of a vowel, where coarticulation with
    #: neighbouring consonants distorts F0.
    edge_trim_ratio: float = 0.20
    #: A vowel needs at least this many voiced frames to be analysable.
    min_voiced_frames: int = 3


@dataclass
class ToneConfig:
    """Thresholds for classifying a normalized pitch value as H, M, or L.

    Values are in semitones relative to the speaker's own median F0, which makes
    them speaker- and sex-independent. A Yoruba H/M/L system typically spans
    roughly 6-8 semitones end to end.

    These defaults were tuned by grid search against the orthographic tone labels
    of the training corpus (see section 9 of the training notebook). They are
    asymmetric because the per-utterance median sits above the centre of the
    speaker's Mid register: Mid is the most frequent tone, but High tones pull
    the median up. Re-run the sweep if you change corpus or microphone setup.
    """

    high_threshold_st: float = 0.6
    low_threshold_st: float = -2.0

    #: Yoruba tone is relative, so a vowel is judged partly against its
    #: neighbours rather than on its absolute level alone. These weight how much
    #: the difference from the previous and next vowel shifts the score. The
    #: previous vowel matters more than the next: tone targets are approached
    #: from the preceding syllable.
    prev_weight: float = 0.2
    next_weight: float = 0.1

    #: Weight on the rise/fall across the vowel itself. A High following a Low
    #: is realized as a rise, which a single averaged level discards.
    slope_weight: float = 0.4
    #: A falling/rising slope steeper than this suggests a contour tone or a
    #: mis-segmented vowel; used to lower confidence, not to reclassify.
    max_level_slope_st_per_s: float = 12.0
    #: Below this confidence the tone is reported but flagged as uncertain.
    min_confidence: float = 0.35

    #: How credible the prompt's tone has to be, in the classifier's own
    #: distribution, for the learner to be given it.
    #:
    #: Marking on the classifier's top choice alone is far stricter than its
    #: certainty warrants. Every corpus utterance is a native speaker reading
    #: correctly, so every tone error the engine reports on corpus audio is a
    #: false alarm -- and on 306,449 held-out units that rule reported one on
    #: 15.3% of vowels. One vowel in six and a half was told it was wrong when
    #: it was right, and 22.2% of vowels in two-vowel utterances.
    #:
    #: **The two thresholds are not the same because the two mistakes are not.**
    #: High said as Low changes the word; the igba set exists to demonstrate
    #: exactly that. High said as Mid is a near-miss, and it is also where the
    #: classifier is least sure, because adjacent levels are separated by a
    #: register a short recording cannot establish. Holding both to one bar
    #: means either nagging about near-misses or letting real errors through.
    #:
    #: Measured on the same held-out units, with detection tested by asking for
    #: a tone the speaker did not produce and checking the engine objects:
    #:
    #:     rule                       false alarms   H<->L caught   adjacent
    #:     top choice                       15.3%            96%         91%
    #:     single 0.25                       8.8%            92%         84%
    #:     single 0.15                       5.8%            89%         77%
    #:     gross 0.35 / adjacent 0.10        5.6%            92%         72%
    #:
    #: The split rule holds High-versus-Low detection exactly where a single
    #: 0.25 bar put it while cutting false alarms by a further 36%, and it beats
    #: a single 0.15 bar on both counts at once. What it gives up is near-miss
    #: detection, which is the right thing to give up: a learner does not need
    #: to be native, and being wrongly corrected teaches nothing.
    min_tone_credibility: float = 0.35

    #: The same, for a confusion between neighbouring levels -- High with Mid,
    #: or Mid with Low. Deliberately far looser. See above.
    min_tone_credibility_adjacent: float = 0.10


@dataclass
class SegmentalConfig:
    """Thresholds for checking *which* vowel a learner produced.

    Catches substitutions -- ``òkan`` for ``òkun`` -- which alignment and
    duration checks cannot see, because the wrong word still occupies plausible
    time.

    The classifier behind this is 73.9% accurate, so its top prediction is not
    trustworthy enough to contradict a learner. These thresholds instead ask how
    much probability it gives the *expected* vowel, and require corroboration
    before objecting. Measured on correctly produced corpus vowels, 5.2% fall
    below 0.05 by chance, so two in one word is rare and one is not evidence.
    """

    #: A word is queried when *any* of its vowels gives the expected vowel less
    #: probability than this.
    #:
    #: Requiring two suspect vowels was wrong: minimal pairs differ in exactly
    #: one segment by definition, so ``òkun``/``òkan`` could never trigger it.
    #:
    #: Measured on 934 correctly-spoken corpus words with the class-balanced
    #: classifier:
    #:
    #:     threshold 0.01 -> 1.6% of correct words queried
    #:     threshold 0.02 -> 3.4%
    #:     threshold 0.04 -> 7.5%
    #:
    #: Recalibrated for the wav2vec2 classifier (accuracy 0.873, macro F1 0.811).
    #: The learned features changed what these thresholds can buy. At the old
    #: 0.02/0.15 the 39-feature model gave 6.1% false alarms for 66.3% of
    #: substitutions caught; the same setting now gives **0.5% for 78.9%** --
    #: better on both axes without touching a threshold. That headroom is spent
    #: on recall instead:
    #:
    #:     suspect  weak  false alarms  caught
    #:       0.02   0.15      0.5%       78.9%
    #:       0.10   0.30      2.2%       90.5%
    #:       0.15   0.30      3.1%       91.5%
    #:       0.20   0.30      3.8%       94.0%   <- here
    #:       0.30   0.30      5.8%       94.0%
    #:
    #: 0.20/0.30 is the measured optimum under recall-minus-twice-false-alarms.
    #: Spending 3.8% is affordable now in a way it was not before: a flagged word
    #: is no longer dropped from the learner side, so a false flag costs a
    #: "check this word" note rather than marks. Only a substitution the decoder
    #: can NAME rewrites the word and penalises the score.
    suspect_probability: float = 0.20

    #: Extra corroboration for multi-vowel words: two independently suspect
    #: vowels is much rarer than one, so such words are queried even when no
    #: single vowel falls below the strict threshold.
    #:
    #: Calibrated from both sides by ``scripts/calibrate_suspect.py``, which
    #: aligns each utterance twice -- against its true transcript, and against
    #: one with a single word swapped for a different real word of the same
    #: vowel count. Over 1,600 known-correct and 199 known-wrong words:
    #:
    #:     suspect  weak  false alarms  substitutions caught
    #:       0.02   0.05      3.6%            50.8%   <- was here
    #:       0.02   0.15      6.1%            66.3%   <- now here
    #:       0.02   0.30     11.2%            80.4%
    #:       0.05   0.30     14.4%            86.4%
    #:       0.30   0.30     42.9%            95.0%
    #:
    #: The old 0.05 missed half of real substitutions -- the failure this check
    #: exists to prevent -- while spending only 3.6% on false alarms.
    #:
    #: 0.30 with the learned features. The reason for the old, cautious 0.15 was
    #: that a flagged word was deleted from the learner side and thus scored as
    #: an error, so false alarms cost marks. Flags no longer do that, and the
    #: classifier is far sharper, so the conservative setting was leaving
    #: 15 points of recall unclaimed for no benefit.
    weak_probability: float = 0.30

    #: A vowel shorter than this is measured but not judged.
    #:
    #: Forced alignment must place every prompt word somewhere, so an omitted
    #: word steals time from its neighbours, and a squeezed span used to be read
    #: confidently and wrongly. Under the 37.9 h acoustic model a 30 ms final
    #: vowel returned P(expected) = 0.000 with entropy 0.10 -- confidently wrong,
    #: so no uncertainty measure could catch it -- and 15.1% of correctly-read
    #: SLR86 vowels landed under 40 ms.
    #:
    #: The 118 h model changed what a short span means. Only 5.3% of SLR86
    #: vowels now fall under 40 ms, and where they do the classifier is right:
    #:
    #:     duration      n   median P(expected)   below 0.20
    #:      0-30 ms    113        0.969              7.1%
    #:     30-50 ms    850        0.976              3.5%
    #:      50+ ms   4,638        0.987              1.8%
    #:
    #: A short vowel is now usually a genuinely short vowel rather than a
    #: mis-segmented one, so the floor exists to skip far less:
    #:
    #:     floor    vowels skipped   false flags removed   ratio
    #:     20 ms          0.4%              0.9%            2.3
    #:     30 ms          2.0%              7.1%            3.6   <- here
    #:     40 ms          6.8%             22.3%            3.3
    #:     50 ms         17.2%             39.3%            2.3
    #:
    #: Re-measured against the fine-tuned encoder, which is confident more often
    #: and correctly so: P(expected) in the 0-30 ms bucket rose from 0.969 to
    #: 0.988 and median entropy fell from 0.17 to 0.06. 30 ms remains the best
    #: trade.
    #:
    #: 30 ms both earns its cost best and skips a third as many vowels as 40 ms
    #: did. That matters because the guard was silencing real errors: a learner
    #: who said ``àbùbùtu`` for ``àbùbùtán`` produced a 30 ms final vowel that
    #: the classifier read correctly as ``u``, and the 40 ms floor threw that
    #: verdict away, leaving GOP to catch the word by a different route.
    min_vowel_duration_s: float = 0.030

    #: Entropy of the vowel distribution above which the vowel is not judged.
    #:
    #: Catches the opposite failure: the model spreading probability because the
    #: audio is genuinely ambiguous. A learner's ``ọkun`` read as
    #: 0.46/0.15/0.15/0.12 is the classifier saying it cannot tell, and a
    #: probability floor cannot see that -- 0.46 clears any sane threshold.
    #:
    #:     ceiling  vowels skipped   false flags removed   ratio
    #:       2.0         0.0%               0.9%             --
    #:       1.5         0.2%               4.5%           22.5   <- here
    #:       1.0         1.4%              14.3%           10.2
    #:
    #: 1.5 removes fourteen times as large a share of false alarms as of vowels
    #: overall, which makes it the cheapest guard in the engine.
    max_vowel_entropy: float = 1.5
    min_weak_vowels: int = 2


@dataclass
class ScoringConfig:
    """Weights for the composite pronunciation score."""

    text_weight: float = 0.4
    tone_weight: float = 0.4

    #: Weight on producing the prompt's vowels. Tone stays the largest single
    #: concern -- it is what changes meaning in Yoruba -- but a learner saying
    #: every word with the wrong vowels was previously scoring full marks with
    #: only an advisory note beside them.
    vowel_weight: float = 0.2
    #: Scores below this trigger "needs practice" feedback in the report.
    pass_threshold: float = 0.70

    #: Minimum fraction of the learner's recording that the aligned words must
    #: occupy for the attempt to count as verified. Below this, the learner
    #: probably said something other than the prompt, and the scores describe a
    #: forced fit rather than a real match.
    #:
    #: Measured on 120 corpus utterances aligned twice -- against their own
    #: transcript and against another utterance's -- correct alignments cover a
    #: median 64% of the audio and mismatched ones 54%. The distributions
    #: overlap heavily: at 0.35 this catches only 12% of deliberately
    #: mismatched utterances, and no threshold does better than 72% balanced
    #: accuracy. Raising it trades false confidence for false accusations --
    #: 0.50 catches 43% but wrongly fails 16% of genuine attempts.
    #:
    #: **This is a weak check and it is now the fallback, not the primary one.**
    #: It is a proxy for a question a recogniser answers directly, so when the
    #: transcript provider recognises speech the engine gates on
    #: ``min_recognised_similarity`` instead. This value applies only in
    #: prompted mode, where the transcript is the prompt echoed back and there
    #: is no other evidence available.
    #:
    #: It is also sensitive to dead air, because the denominator is the whole
    #: file: silence a learner leaves at either end counts against them. That is
    #: why browser recordings are trimmed before assessment.
    #: A prompt vowel counts as not produced when the classifier puts it below
    #: this while backing something else above ``min_confident_vowel``.
    #:
    #: Both bars are deliberately far apart. The classifier is 93.1% accurate,
    #: so about one vowel in fourteen is wrong, and deducting on every objection
    #: would tax correct speech more than it corrects wrong speech -- which is
    #: why vowel flags advised rather than scored until now. The case this is
    #: built for is not marginal: a learner producing /o/ where ``dàbọ̀`` needs
    #: /ɔ/ scores o=0.91 against ọ=0.01.
    max_confident_vowel_error: float = 0.15
    min_confident_vowel: float = 0.70

    #: Below this many words in the prompt, the recogniser is not consulted.
    #:
    #: Default 1, which never skips. It was briefly set to 3 on a mistaken
    #: diagnosis: a one-second clip of `ìgbà` transcribed as `bí ìgbà tẹ bá`
    #: and was read as the decoder inventing words. It was not -- the clip had
    #: been cut with neighbouring words still in it, and the recogniser was
    #: right. Transcription is the only thing that can say what a learner
    #: actually said, and a short prompt is where a substitution is most likely
    #: to be the whole answer, so it is the last thing to switch off.
    min_words_to_recognise: int = 1

    min_speech_coverage: float = 0.35

    #: How much a recognised transcript must sound like the prompt before the
    #: attempt counts as an attempt at that phrase at all.
    #:
    #: This replaces the coverage proxy whenever a real recogniser supplied the
    #: transcript. "Did the learner try to say this phrase?" is what recognition
    #: answers directly, where alignment coverage only ever circled it.
    #:
    #: Compared as characters with tone marks stripped, not as words -- see
    #: :func:`~yoruba_engine.compare.engine.phrase_resemblance` for why. A
    #: learner who mangles every word is still attempting the prompt; word-level
    #: agreement puts them in the same range as someone reading a different
    #: phrase, and character-level agreement separates them.
    #:
    #: Set in the middle of a measured gap: genuine attempts scored 0.67-0.97
    #: and different phrases 0.13-0.43. That is four recordings by one speaker
    #: against eight unrelated phrases, so it is a provisional threshold on a
    #: wide gap, not a tuned one on a narrow one.
    #:
    #: This is not a grade. It asks only whether the learner attempted the
    #: prompt, so a poor but genuine attempt must clear it comfortably.
    min_phrase_resemblance: float = 0.55

    #: How short a word may be, relative to the 5th-percentile duration for its
    #: length, and still count as actually spoken.
    #:
    #: In prompted practice there is no recognizer, so "did the learner say this
    #: word?" is answered from the alignment: a skipped word gets squeezed to
    #: almost nothing because the aligner must still place it somewhere.
    #:
    #: The comparison is against a per-length expectation rather than a fixed
    #: number of milliseconds, because duration scales with word length -- one
    #: phone runs about 120 ms and eight phones about 780 ms on this corpus. A
    #: flat threshold would either reject legitimately short words like ``ti``
    #: or wave through a long word crushed to a fraction of its normal length.
    #:
    #: Calibrated from both sides by ``scripts/calibrate_omissions.py``, which
    #: aligns corpus utterances against their own transcript with one extra word
    #: spliced in -- so every inserted word is a known omission and every
    #: original word is known-spoken. Over 1,982 spoken and 243 omitted words:
    #:
    #:     fraction      0.3    0.4    0.5    0.6    0.7    0.8    0.9    1.0
    #:     false alarms 0.6%   1.1%   1.4%   2.3%   3.0%   4.6%   6.3%   9.4%
    #:     omissions   27.6%  61.7%  78.2%  88.1%  92.2%  95.9%  97.1%  98.4%
    #:
    #: That table is measured on CORPUS audio, and 0.8 was briefly set from it.
    #: It was wrong: re-measured on real learner recordings -- a different
    #: speaker, mic and room, which the aligner fits less tightly -- words the
    #: learner did say land closer to the floor:
    #:
    #:     fraction      0.5   0.6   0.7   0.8   0.9   1.0
    #:     false MISSING   0     0     0     2     3     4   (of 14 spoken)
    #:     omissions       2     2     2     2     2     2   (of 2)
    #:
    #: 0.8 bought no extra recall on learner audio and cost two false "you did
    #: not say this" reports. Genuinely omitted words sat at ratio 0.50 while
    #: the shortest genuinely spoken one was 0.72, so 0.6 sits mid-gap with
    #: room on both sides -- and still lifts corpus omission recall from 78.2%
    #: at 0.5 to 88.1%.
    #:
    #: The general lesson, learned twice: thresholds for judging learners must
    #: be calibrated on learner audio, not on the corpus the model trained on.
    #: A word aligned to less than this share of its expected duration is
    #: reported as unspoken.
    #:
    #: Recalibrated against the 118 h acoustic model by splicing one unspoken
    #: word into 250 utterances, giving 2,034 known-spoken and 249 known-omitted
    #: words:
    #:
    #:     fraction   false alarms   omissions caught
    #:        0.5         1.9%            57.8%
    #:        0.6         5.8%            73.1%
    #:        0.7         7.2%            83.9%   <- here
    #:        0.8         9.2%            88.0%
    #:        0.9        11.5%            92.8%
    #:
    #: The calibration script recommends 0.9, maximising recall minus false
    #: alarms with the two weighted equally. They are not equal here. A missed
    #: omission means a learner is not told they skipped a word; a false alarm
    #: means being told they skipped a word they actually said, which is the
    #: error that makes a report untrustworthy rather than merely incomplete.
    #:
    #: 0.7 takes the steepest part of the curve -- 10.8 points of recall for 1.4
    #: points of false alarms over 0.6 -- and stops before the trade flattens.
    min_duration_fraction: float = 0.7

    def __post_init__(self) -> None:
        # All three, because vowel identity is now scored rather than merely
        # noted. A recording where vowels could not be judged drops that weight
        # and renormalises the other two, so the sum still has to be 1.0 here.
        total = self.text_weight + self.tone_weight + self.vowel_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total} "
                f"(text {self.text_weight}, tone {self.tone_weight}, "
                f"vowel {self.vowel_weight})")


def _default_mfa_executable() -> str:
    """Locate the ``mfa`` binary.

    Preference order: the ``MFA_BIN`` environment variable, the project-local
    conda environment at ``.mfa-env/`` (created by the install steps in
    INSTALL.md), then whatever is on ``PATH``. The project-local lookup means
    the engine works without activating a conda environment first.
    """
    override = os.environ.get("MFA_BIN")
    if override:
        return override

    local = PROJECT_ROOT / ".mfa-env" / "bin" / "mfa"
    if local.exists():
        return str(local)

    return "mfa"


@dataclass
class GOPConfig:
    """Thresholds for the Goodness-of-Pronunciation check.

    GOP reads the acoustic model's own per-phone log-likelihood -- the score it
    must compute to align at all -- rather than re-deriving a judgement from
    measured formants. It is a second, independent opinion alongside the vowel
    classifier, and the two fail in opposite directions:

    * GOP catches gross mismatch, including words the learner invented. A
      recording of ``àbùbùtu`` against a prompt of ``àbùbùtán`` scored -4.98,
      with one vowel at -10.6. Nothing else in the engine can flag that,
      because ``àbùbùtu`` is not a word and cannot be looked up.
    * GOP is blind to minimal pairs. ``ọkàn`` read against ``òkun`` scored
      **+0.25** -- better than average. Forced alignment has already slid the
      boundaries to maximise fit, and ``ọ`` fits ``o`` well, so the measure and
      the aligner optimise the same quantity.

    Calibrated by ``scripts/calibrate_gop.py`` on 1,627 known-correct and 198
    known-wrong words, scoring the mean z-score across a word's phones:

        threshold     -1.0   -1.5   -2.0   -2.5   -3.0
        false alarms  4.9%   2.0%   1.1%   0.6%   0.4%
        caught       97.0%  87.9%  75.3%  63.1%  51.5%

    -1.5 is chosen over the -1.0 that maximises recall-minus-false-alarms:
    correct words have a 5th percentile of -0.99, so -1.0 sits right on top of
    the correct distribution and would query one word in twenty.

    Note the recall figure is measured against *random* word substitutions,
    which are usually acoustically distant. It does not transfer to minimal
    pairs -- those are the vowel classifier's job.
    """

    #: Mean z-score below which a word is queried.
    threshold: float = -1.5

    #: The word is only judged when at least this many of its phones have
    #: reference statistics; one scored phone is not evidence.
    min_scored_phones: int = 2

    enabled: bool = True


@dataclass
class MFAConfig:
    """Locations of Montreal Forced Aligner assets.

    ``acoustic_model`` points at the model produced by
    ``notebooks/train_mfa_yoruba.ipynb``. There is no reliable off-the-shelf
    Yoruba acoustic model, so this path will not exist until you have trained
    one.
    """

    executable: str = field(default_factory=_default_mfa_executable)
    acoustic_model: Path = PROJECT_ROOT / "models" / "yoruba_acoustic.zip"
    dictionary: Path = PROJECT_ROOT / "data" / "dictionary" / "yoruba.dict"
    num_jobs: int = field(default_factory=lambda: int(os.environ.get("MFA_JOBS", "4")))
    beam: int = 100
    retry_beam: int = 400
    #: Passed to ``mfa align --single_speaker`` when the corpus is one speaker.
    single_speaker: bool = False


@dataclass
class EngineConfig:
    """Top-level configuration object threaded through the pipeline."""

    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    line_index: Path = PROJECT_ROOT / "data" / "corpus" / "line_index.tsv"
    audio_dir: Path = PROJECT_ROOT / "data" / "corpus" / "wavs"
    output_dir: Path = PROJECT_ROOT / "output"
    audio_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a")

    gop: GOPConfig = field(default_factory=GOPConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    tone: ToneConfig = field(default_factory=ToneConfig)
    segmental: SegmentalConfig = field(default_factory=SegmentalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    mfa: MFAConfig = field(default_factory=MFAConfig)

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


DEFAULT_CONFIG = EngineConfig()
