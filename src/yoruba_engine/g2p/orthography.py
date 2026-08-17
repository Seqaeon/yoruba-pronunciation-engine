"""Yoruba orthography handling: normalization, tone marks, and syllable structure.

Yoruba text arrives in wildly inconsistent Unicode. The same word may appear as
precomposed characters (U+1EB9 LATIN SMALL LETTER E WITH DOT BELOW), as a base
letter plus a combining dot below (U+0323), with the dot rendered as a cedilla
or an underscore, or with tone marks stacked in either order relative to the
dot. Everything in this package assumes text has passed through
:func:`normalize` first.

Tone marking convention:
    High  -> acute accent   (U+0301), e.g. á
    Mid   -> unmarked       (canonically), e.g. a
    Low   -> grave accent   (U+0300), e.g. à

Note that some corpora mark Mid explicitly with a macron. That is handled, but
an *unmarked* vowel is always read as Mid, which is why undiacritized corpora
are unusable for tone work -- see :func:`has_tone_marks`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..datatypes import Tone

#: Combining marks that carry meaning in Yoruba orthography.
COMBINING_ACUTE = "́"
COMBINING_GRAVE = "̀"
COMBINING_MACRON = "̄"
COMBINING_DOT_BELOW = "̣"

#: Marks sometimes substituted for the canonical dot below by naive input
#: methods or by transliteration from print sources.
_DOT_BELOW_VARIANTS = {
    "̧",  # combining cedilla
    "̱",  # combining macron below
    "̭",  # combining circumflex below
}

#: Quote characters used interchangeably for the Yoruba elision apostrophe.
_ELISION_VARIANTS = ("’", "‘", "ʼ", "`", "´")

#: A combining mark at the start of the string or directly after whitespace,
#: i.e. one with no base character to attach to.
_ORPHAN_MARK = re.compile(r"(?:(?<=^)|(?<=\s))[̀-ͯ]+")

#: Only these three letters take a dot below in Yoruba orthography. A dot on any
#: other letter is a typing error and is discarded rather than failing the word.
DOT_BELOW_BASES = frozenset("eos")

_TONE_BY_MARK = {
    COMBINING_ACUTE: Tone.HIGH,
    COMBINING_GRAVE: Tone.LOW,
    COMBINING_MACRON: Tone.MID,
}

#: The seven oral vowels of Standard Yoruba, as base letters after the dot below
#: has been folded into the letter identity.
ORAL_VOWELS = ("a", "e", "ẹ", "i", "o", "ọ", "u")

#: The five nasal vowels of Standard Yoruba (fáwẹ́lì aránmúpè), written
#: orthographically as vowel + ``n``. Note that ``e`` and ``o`` do **not**
#: nasalize: there is no ``*en`` or ``*on`` vowel. Sequences that look like one
#: are either a missing sub-dot (``ẹn``/``ọn``) or a syllabic nasal.
NASAL_VOWELS = ("an", "ẹn", "in", "ọn", "un")

#: Vowels that can take nasalization, as base letters.
NASALIZABLE = frozenset({"a", "ẹ", "i", "ọ", "u"})

#: ``e`` and ``o`` cannot nasalize, so a nasalizing ``n`` after one of them
#: means the sub-dot was dropped when the text was typed -- a very common
#: omission. Mapping them onto the legal nasal vowel recovers the intended word
#: instead of inventing a phoneme Yoruba does not have.
MISSING_DOT_NASAL = {"e": "ẹ", "o": "ọ"}

#: Consonant graphemes, longest first so that digraphs win during segmentation.
CONSONANTS = (
    "gb",  # labial-velar stop, a single phoneme -- must precede "g" and "b"
    "ṣ",   # postalveolar fricative
    "b", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "w", "y",
)

#: The syllabic nasal, which bears tone on its own (e.g. ǹ in "ǹ jẹun").
SYLLABIC_NASAL = "n"


def normalize(text: str) -> str:
    """Normalize Yoruba text to a canonical composed form.

    Folds dot-below variants onto the canonical U+0323, orders combining marks
    consistently, collapses whitespace, and lowercases. The result is NFC so
    that ``len()`` and slicing behave intuitively for callers, while
    :func:`decompose_char` re-splits when marks are needed.

    Args:
        text: Raw transcript text.

    Returns:
        Normalized text. Never ``None``; empty input yields an empty string.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFD", text)
    for variant in _DOT_BELOW_VARIANTS:
        text = text.replace(variant, COMBINING_DOT_BELOW)

    # Yoruba marks elision with an apostrophe (l'órí = ní + orí). Corpora use
    # whichever quote character the input method produced; fold them all onto
    # the ASCII apostrophe so the G2P sees one form.
    for variant in _ELISION_VARIANTS:
        text = text.replace(variant, "'")

    # Drop combining marks with no base character to attach to. These occur in
    # the corpora where a diacritic was typed before its letter, or where a
    # word was split mid-cluster; left in place they produce phantom tokens.
    text = _ORPHAN_MARK.sub("", text)

    # Canonical ordering places the below-mark before the above-mark. NFD does
    # this by combining class already, but explicit re-normalization guards
    # against text that was assembled by string concatenation.
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.lower().split())


def grapheme_clusters(text: str) -> list[str]:
    """Split text into grapheme clusters: each base character with its marks.

    This must be used instead of ``list(text)`` anywhere tone marks matter.
    Yoruba routinely produces combinations with no precomposed Unicode form --
    ``ẹ̀`` is U+1EB9 followed by a combining grave, because no single code point
    encodes "e with dot below and grave". Naive character iteration therefore
    splits the tone mark away from its vowel and silently reads the vowel as Mid.

    Args:
        text: Any string.

    Returns:
        Clusters in order; joining them reproduces the input exactly.
    """
    clusters: list[str] = []
    for char in text:
        if clusters and unicodedata.combining(char):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


@dataclass(frozen=True)
class DecomposedChar:
    """A base letter with its Yoruba-relevant combining marks separated out."""

    base: str
    tone: Tone
    has_dot_below: bool

    @property
    def letter(self) -> str:
        """The base letter including the dot below, e.g. ``ẹ``.

        The dot is only honoured on ``e``, ``o``, and ``s`` -- the three letters
        that take one in Yoruba. The corpora contain typos where the dot landed
        on a neighbouring letter (``j̣``, ``ṛ``, ``ạ``); discarding it there
        recovers the intended letter instead of failing the whole word.
        """
        if self.has_dot_below and self.base in DOT_BELOW_BASES:
            return unicodedata.normalize("NFC", self.base + COMBINING_DOT_BELOW)
        return self.base


def decompose_char(char: str) -> DecomposedChar:
    """Split one character into base letter, tone, and dot-below presence.

    Args:
        char: A single (possibly precomposed) character.

    Returns:
        Its decomposition. Characters with no tone mark are reported as
        :attr:`Tone.MID`, matching Yoruba convention.
    """
    decomposed = unicodedata.normalize("NFD", char)
    base = ""
    tone = Tone.MID
    dot = False

    for component in decomposed:
        if component in _TONE_BY_MARK:
            tone = _TONE_BY_MARK[component]
        elif component == COMBINING_DOT_BELOW:
            dot = True
        elif not unicodedata.combining(component):
            base += component

    return DecomposedChar(base=base, tone=tone, has_dot_below=dot)


def strip_tones(text: str) -> str:
    """Remove tone marks while preserving dots below.

    Useful for building a tone-insensitive lookup key, and for comparing a
    learner's segmental accuracy independently of their tonal accuracy.
    """
    decomposed = unicodedata.normalize("NFD", normalize(text))
    kept = "".join(c for c in decomposed if c not in _TONE_BY_MARK)
    return unicodedata.normalize("NFC", kept)


def has_tone_marks(text: str) -> bool:
    """Whether the text carries any explicit tone diacritic.

    A corpus where this is ``False`` for nearly every line is undiacritized.
    Tone comparison against such a corpus is meaningless: every vowel would be
    read as Mid. Stage 1 audits for exactly this condition.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return any(c in (COMBINING_ACUTE, COMBINING_GRAVE) for c in decomposed)


def tone_pattern(word: str) -> list[Tone]:
    """Extract the lexical tone sequence from a written word.

    Each tone-bearing unit (vowel, or syllabic nasal) contributes one tone.

    Args:
        word: A single normalized Yoruba word.

    Returns:
        Tones in order of appearance. ``["L", "H"]`` for ``òrí``.
    """
    tones: list[Tone] = []
    chars = grapheme_clusters(normalize(word))
    index = 0

    while index < len(chars):
        decomposed = decompose_char(chars[index])
        letter = decomposed.letter

        if letter in ORAL_VOWELS:
            tones.append(decomposed.tone)
            # A following unmarked "n" nasalizes this vowel rather than forming
            # its own tone-bearing unit. A tone-marked one is a syllabic nasal.
            if index + 1 < len(chars):
                nxt = decompose_char(chars[index + 1])
                if nxt.letter == SYLLABIC_NASAL and _is_nasalizing(chars, index + 1):
                    index += 1
        elif letter == SYLLABIC_NASAL and _is_syllabic_nasal(chars, index):
            tones.append(decomposed.tone)

        index += 1

    return tones


def _is_nasalizing(chars: list[str], n_index: int) -> bool:
    """Whether the ``n`` at ``n_index`` nasalizes the preceding vowel.

    Three conditions must hold:

    * It carries no tone mark of its own. A marked ``ǹ``/``ń`` is a *syllabic
      nasal* -- its own tone-bearing unit, as in ``oǹdó`` or ``alóńgẹ`` -- and
      nasalizing the preceding vowel would both invent a nasal vowel and lose a
      syllable.
    * It is word-final or followed by a consonant. An ``n`` before a vowel is
      the onset of the next syllable.
    * The preceding vowel is one that can nasalize.
    """
    marked = decompose_char(chars[n_index]).tone is not Tone.MID
    if marked:
        return False

    if n_index + 1 < len(chars):
        following = decompose_char(chars[n_index + 1]).letter
        if following in ORAL_VOWELS:
            return False

    return True


def _is_syllabic_nasal(chars: list[str], index: int) -> bool:
    """Whether the ``n`` at ``index`` is a tone-bearing syllabic nasal.

    Two cases: it carries its own tone mark (``ǹ``/``ń``), as in ``oǹdó``; or it
    begins the word and is not followed by a vowel, as in the progressive marker
    ``ń``.
    """
    if decompose_char(chars[index]).tone is not Tone.MID:
        return True
    if index != 0:
        return False
    if index + 1 >= len(chars):
        return True
    return decompose_char(chars[index + 1]).letter not in ORAL_VOWELS


#: Punctuation stripped from token edges. Dashes are included because the
#: corpora use a bare em dash as a standalone token.
_PUNCTUATION = ".,!?;:\"'()[]{}«»…“”‘’—–"

#: Canonical token for any non-speech event in a transcript.
NOISE_TOKEN = "[noise]"

#: Bracketed annotation markers used by the source corpora to flag non-speech
#: events: ``[snap]`` (a clapper used for session sync), ``[breath]``,
#: ``[external]`` (background noise), ``[abrupt]``, ``[hesitation]``.
_ANNOTATION = re.compile(r"\[[^\]]*\]")


def tokenize(text: str, keep_noise: bool = True) -> list[str]:
    """Split a transcript into word tokens, dropping punctuation.

    Hyphens inside words are preserved because Yoruba compounds use them, but
    surrounding punctuation is stripped.

    Bracketed annotation markers are collapsed to :data:`NOISE_TOKEN` rather
    than being deleted. Keeping them matters: the aligner needs to know a
    non-speech event occupies that stretch of audio, or it will stretch the
    neighbouring vowel across it and corrupt the F0 measurement that tone
    detection depends on. The dictionary maps the token to MFA's ``spn``
    (spoken noise) phone.

    Args:
        text: Raw transcript text.
        keep_noise: When ``False``, annotation markers are dropped entirely.
            Use this when building a vocabulary of real words, or when
            comparing a learner's words against a reference.

    Returns:
        Token list in order.
    """
    normalized = normalize(text)
    tokens: list[str] = []

    for raw in normalized.split():
        if _ANNOTATION.fullmatch(raw):
            if keep_noise:
                tokens.append(NOISE_TOKEN)
            continue

        # An annotation fused to adjacent punctuation, e.g. "word.[breath]".
        stripped_markers = _ANNOTATION.sub(" ", raw)
        if stripped_markers != raw:
            if keep_noise and _ANNOTATION.search(raw):
                for piece in stripped_markers.split():
                    cleaned = piece.strip(_PUNCTUATION)
                    if cleaned:
                        tokens.append(cleaned)
                tokens.append(NOISE_TOKEN)
                continue
            raw = stripped_markers

        for piece in raw.split():
            cleaned = piece.strip(_PUNCTUATION)
            if cleaned:
                tokens.append(cleaned)

    return tokens


def is_noise_token(token: str) -> bool:
    """Whether a token marks a non-speech event rather than a spoken word.

    Matches any fully bracketed token, not just :data:`NOISE_TOKEN`. MFA
    rewrites bracketed dictionary entries to its own generic ``[bracketed]``
    label in the TextGrid, so alignment output never contains the token that
    went in. Checking for the brackets themselves catches both directions, plus
    any residual ``[snap]``/``[breath]`` markers from an unnormalized source.
    """
    stripped = token.strip()
    return len(stripped) > 1 and stripped.startswith("[") and stripped.endswith("]")
