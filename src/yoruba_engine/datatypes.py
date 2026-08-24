"""Core datatypes shared across pipeline stages.

These are deliberately plain dataclasses with no framework dependencies so they
can be converted to Pydantic models at the API boundary without restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Tone(str, Enum):
    """The three contrastive tones of Standard Yoruba.

    ``UNKNOWN`` is used when pitch extraction failed for a segment (for example
    on a devoiced or creaky vowel) and must never be silently treated as MID.
    """

    HIGH = "H"
    MID = "M"
    LOW = "L"
    UNKNOWN = "?"

    @property
    def diacritic(self) -> str:
        """The combining mark that represents this tone in Yoruba orthography."""
        return {Tone.HIGH: "́", Tone.LOW: "̀", Tone.MID: ""}.get(self, "")

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Phoneme:
    """A single phone in the pronunciation dictionary.

    Attributes:
        symbol: ASCII-safe phone label used in the MFA dictionary (e.g. ``gb``,
            ``a_H``). Tone-bearing vowels carry a ``_H``/``_M``/``_L`` suffix.
        is_vowel: Whether this phone is a tone-bearing unit. Only vowels carry
            tone in Yoruba, so only these are measured during pitch analysis.
        tone: The lexical tone, or ``None`` for consonants.
        is_nasal: Whether this is a nasalized vowel (orthographic ``Vn``).
    """

    symbol: str
    is_vowel: bool = False
    tone: Tone | None = None
    is_nasal: bool = False


@dataclass(frozen=True)
class Utterance:
    """One audio file plus its transcript, as listed in ``line_index.tsv``."""

    utterance_id: str
    audio_path: Path
    transcript: str
    speaker_id: str = "unknown"
    duration_s: float | None = None


@dataclass
class Interval:
    """A labelled time span from an MFA TextGrid."""

    label: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def midpoint_s(self) -> float:
        return (self.start_s + self.end_s) / 2.0


@dataclass
class Alignment:
    """Word- and phone-level alignment for a single utterance."""

    utterance_id: str
    words: list[Interval] = field(default_factory=list)
    phones: list[Interval] = field(default_factory=list)

    #: Per-phone acoustic scores from the aligner, when they were collected.
    #: Populated by :meth:`MFAAligner.align_single` with ``collect_gop=True``;
    #: empty otherwise, which callers must treat as "not measured" rather than
    #: as a pass.
    phone_scores: list = field(default_factory=list)

    def phones_within(self, word: Interval) -> list[Interval]:
        """Return the phone intervals falling inside ``word``.

        Uses midpoint containment rather than strict nesting, because MFA
        boundary times for words and phones can differ by a frame.
        """
        return [
            p for p in self.phones
            if word.start_s <= p.midpoint_s <= word.end_s and p.label not in SILENCE_LABELS
        ]


#: Labels MFA uses for silence and out-of-vocabulary padding.
SILENCE_LABELS: frozenset[str] = frozenset({"", "sil", "sp", "spn", "<eps>"})


@dataclass
class PitchPoint:
    """A single voiced pitch measurement."""

    time_s: float
    f0_hz: float
    semitones: float | None = None


@dataclass
class VowelSegment:
    """A vowel interval with its extracted pitch contour and detected tone."""

    label: str
    start_s: float
    end_s: float
    word_index: int
    contour: list[PitchPoint] = field(default_factory=list)
    detected_tone: Tone = Tone.UNKNOWN
    mean_semitones: float | None = None
    slope_st_per_s: float | None = None
    confidence: float = 0.0

    #: F0 sampled at 20%, 50% and 80% through the trimmed contour. A single
    #: average hides the shape of the vowel; the rise or fall across it carries
    #: real tonal information, especially where a High follows a Low.
    onset_semitones: float | None = None
    mid_semitones: float | None = None
    offset_semitones: float | None = None

    #: The contextual score the classifier actually thresholds on: the vowel's
    #: own level adjusted by its neighbours and its internal slope. Retained for
    #: debugging and for tuning runs.
    tone_score: float | None = None

    @property
    def excursion_st(self) -> float | None:
        """Rise (positive) or fall (negative) across the vowel, in semitones."""
        if self.onset_semitones is None or self.offset_semitones is None:
            return None
        return self.offset_semitones - self.onset_semitones

    @property
    def voiced_ratio(self) -> float:
        """Fraction of the segment for which a pitch value was recovered."""
        if self.end_s <= self.start_s:
            return 0.0
        return min(1.0, len(self.contour) * 0.01 / (self.end_s - self.start_s))


@dataclass
class WordAnalysis:
    """Per-word tonal analysis, used as the unit of learner feedback."""

    word: str
    start_s: float
    end_s: float
    vowels: list[VowelSegment] = field(default_factory=list)

    #: How many phones the aligner placed inside this word. Needed to judge
    #: whether its duration is plausible: expected duration scales with length,
    #: and the vowel count alone underestimates consonant-heavy words.
    phone_count: int = 0

    @property
    def tone_pattern(self) -> list[Tone]:
        return [v.detected_tone for v in self.vowels]

    def pattern_string(self) -> str:
        """Human-readable pattern, e.g. ``Low-High``."""
        names = {Tone.HIGH: "High", Tone.MID: "Mid", Tone.LOW: "Low", Tone.UNKNOWN: "?"}
        return "-".join(names[t] for t in self.tone_pattern) or "-"


@dataclass
class UtteranceAnalysis:
    """Everything the engine knows about one rendition of one utterance."""

    utterance_id: str
    audio_path: Path
    transcript: str
    words: list[WordAnalysis] = field(default_factory=list)
    speaker_median_hz: float | None = None

    #: Total audio duration, and the fraction of it the aligned words occupy.
    #: Forced alignment always produces *an* alignment, even against words the
    #: speaker never said; when the text does not match the audio, the aligner
    #: is forced to cram the words into a fraction of the recording. Coverage is
    #: therefore the signal that the speaker actually said the target text.
    audio_duration_s: float | None = None
    speech_coverage: float | None = None

    #: Per-phone acoustic scores from the aligner, when collected. Kept on the
    #: analysis so the GOP check can run after pitch and tone, without a second
    #: alignment pass.
    phone_scores: list = field(default_factory=list)

    @property
    def word_tokens(self) -> list[str]:
        return [w.word for w in self.words]


@dataclass
class ToneError:
    """A single tonal mismatch between learner and reference."""

    word: str
    word_index: int
    vowel_index: int
    expected: Tone
    detected: Tone

    #: How far this vowel sat from the one before it in the same word, in
    #: semitones, and which way the prompt says it should have gone. Both are
    #: ``None`` on a word's first vowel, which has nothing to move from.
    #:
    #: These carry the part of the measurement that survives on a short prompt.
    #: A tone label is an absolute judgement and needs a register to be judged
    #: against; pitch is normalised against the recording's own median, so a
    #: two-vowel word is normalised against itself and there is no register.
    #: The *movement* between two vowels needs no register at all -- it is a
    #: difference, and both terms shift together. So on exactly the utterances
    #: where the label is least trustworthy, the interval is still sound, and
    #: telling a learner "your pitch fell where it should rise" is supported by
    #: the measurement in a way that "that vowel was Low, not Mid" is not.
    measured_step_st: float | None = None
    expected_step: str | None = None      # "rise", "fall" or "level"

    #: Whether the interval runs forward to the next vowel rather than back to
    #: the previous one. A word's first vowel has nothing behind it, and on a
    #: two-vowel word that is half the errors -- and both of the ones the igba
    #: set actually produces. The interval is the same measurement either way.
    step_to_next: bool = False

    #: Whether the recording was long enough to establish where this speaker's
    #: Low, Mid and High actually sit. Under four tone-bearing units it is not:
    #: pitch is normalised against the recording's own median, so a short
    #: utterance is normalised against itself.
    #:
    #: This does not change the score. It changes what the learner is told, and
    #: only in the case where the engine can see that the *movement* was right
    #: and cannot honestly say the starting level was wrong.
    register_measurable: bool = True

    def describe(self) -> str:
        names = {Tone.HIGH: "High", Tone.MID: "Mid", Tone.LOW: "Low", Tone.UNKNOWN: "unclear"}
        ordinal = ["first", "second", "third", "fourth", "fifth"]
        which = ordinal[self.vowel_index] if self.vowel_index < len(ordinal) else f"{self.vowel_index + 1}th"
        line = (
            f"Tone mismatch on {which} vowel of '{self.word}': "
            f"expected {names[self.expected]}, detected {names[self.detected]}."
        )
        movement = self.describe_movement()
        if movement:
            return f"{line} {movement}"
        if not self.register_measurable and self.measured_step_st is not None:
            # The pitch moved the way the prompt asks; what is in doubt is the
            # level it moved from, and one word is not enough to establish that.
            # Saying so is more useful than a flat assertion the measurement
            # does not support.
            return (
                f"{line} The movement is right -- what could not be judged is "
                "the level it starts from, which needs more than a single word. "
                "Try the same word in a sentence."
            )
        return line

    def describe_movement(self) -> str | None:
        """What the learner's pitch did between the last vowel and this one.

        Returns ``None`` when there is no previous vowel to move from, or when
        the movement agrees with the prompt -- in which case the label is the
        only thing wrong and saying it twice would not help.
        """
        if self.measured_step_st is None or self.expected_step is None:
            return None

        size = abs(self.measured_step_st)
        if size < 1.0:
            went = "stayed level"
        elif self.measured_step_st > 0:
            went = f"rose {size:.1f} semitones"
        else:
            went = f"fell {size:.1f} semitones"

        wanted = {"rise": "rise", "fall": "fall", "level": "stay level"}[self.expected_step]
        agrees = (
            (self.expected_step == "rise" and self.measured_step_st >= 1.0)
            or (self.expected_step == "fall" and self.measured_step_st <= -1.0)
            or (self.expected_step == "level" and size < 1.0)
        )
        if agrees:
            return None
        where = ("into the next vowel" if self.step_to_next
                 else "from the vowel before it")
        return f"Your pitch {went} {where}, where it should {wanted}."


@dataclass
class ComparisonResult:
    """The final scored comparison between a reference and a learner rendition."""

    utterance_id: str
    reference_transcript: str
    learner_transcript: str
    text_similarity: float
    tone_similarity: float
    overall_score: float
    word_pairs: list[tuple[str | None, str | None]] = field(default_factory=list)
    tone_errors: list[ToneError] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)

    #: Whether the learner plausibly said the target text at all. ``False`` means
    #: the alignment did not account for the recording, so the scores above
    #: describe a forced fit and must not be presented as an achievement.
    #: Words the learner appears to have pronounced as a different word, caught
    #: by checking vowel identity rather than timing.
    mispronounced_words: list[str] = field(default_factory=list)

    #: Words the acoustic model scored poorly, prompt word -> mean z-score.
    #: An independent opinion from the vowel classifier's: this one catches
    #: gross mismatch and invented words, and is blind to minimal pairs.
    poorly_pronounced: dict[str, float] = field(default_factory=dict)

    #: Prompt word -> the real word the decoder concluded was said instead.
    #: Only filled when the margin cleared the calibrated threshold, so an
    #: entry here is a claim the engine is willing to defend to a learner.
    substitutions: dict[str, str] = field(default_factory=dict)

    #: Prompt words that the aligner could not find in the learner's audio.
    #: Populated in prompted practice, where these are the evidence behind the
    #: text-similarity score.
    undetected_words: list[str] = field(default_factory=list)

    #: Whether ``text_similarity`` measures anything. In prompted practice the
    #: learner transcript is the prompt echoed back, so the value is 100% by
    #: construction and is excluded from the overall score.
    text_similarity_applicable: bool = True

    verified: bool = True
    verification_note: str | None = None
    learner_speech_coverage: float | None = None

    #: How much the recognised transcript sounds like the prompt, 0 to 1, when
    #: a recogniser produced one. This is what decides ``verified`` in that
    #: mode; ``learner_speech_coverage`` is still recorded but is not the gate.
    learner_phrase_resemblance: float | None = None

    #: Share of the prompt's vowels the learner confidently produced, or ``None``
    #: when none could be judged. Distinct from ``mispronounced_words``, which
    #: checks each word against its own spelling and so cannot see a vowel the
    #: recogniser spelled to match what was said.
    vowel_accuracy: float | None = None
