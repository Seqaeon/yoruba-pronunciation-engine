"""Stage 2: Yoruba grapheme-to-phoneme conversion and MFA dictionary generation.

Yoruba orthography is close to phonemic, so a rule-based converter handles the
overwhelming majority of the vocabulary with no training data. The rules here
cover:

* digraph consonants (``gb``, and ``ṣ`` when typed as ``sh``)
* the dot-below vowels ``ẹ`` and ``ọ`` as distinct phonemes
* nasal vowels written ``Vn``
* syllabic nasals, which are tone-bearing
* tone, encoded as a suffix on the vowel phone (``a_H``, ``a_M``, ``a_L``)

Encoding tone into the phone label is a deliberate choice: it lets the MFA
acoustic model learn tone-conditioned vowel realizations, and it makes the
alignment output directly usable for tone comparison without a second lookup.
The cost is a roughly 3x larger vowel inventory, which needs correspondingly
more training data -- see the notebook for the tradeoff and how to switch to
tone-stripped phones if your corpus is small.

Manual overrides always win over the rules; see :class:`PronunciationDictionary`.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from ..exceptions import G2PError
from ..logging_utils import get_logger
from ..datatypes import Phoneme, Tone
from .orthography import (
    CONSONANTS,
    MISSING_DOT_NASAL,
    NASAL_VOWELS,
    ORAL_VOWELS,
    SYLLABIC_NASAL,
    decompose_char,
    grapheme_clusters,
    NOISE_TOKEN,
    is_noise_token,
    normalize,
)

logger = get_logger("g2p")

#: MFA's conventional phone label for spoken noise / non-speech events.
NOISE_PHONE = "spn"

#: Phones that may carry tone, before the tone suffix is appended.
TONE_BEARING = frozenset(ORAL_VOWELS) | frozenset(NASAL_VOWELS) | {"n"}

_TONE_SUFFIX = {Tone.HIGH: "_H", Tone.MID: "_M", Tone.LOW: "_L", Tone.UNKNOWN: "_M"}


def _phone_label(base: str, tone: Tone | None) -> str:
    """Build the ASCII-safe phone label written into the MFA dictionary.

    Dot-below vowels become ``E``/``O`` (uppercase) so the dictionary stays
    ASCII, which avoids a class of Kaldi encoding bugs.
    """
    ascii_base = base.replace("ẹ", "E").replace("ọ", "O").replace("ṣ", "S")
    if tone is None:
        return ascii_base
    return f"{ascii_base}{_TONE_SUFFIX[tone]}"


def grapheme_to_phonemes(word: str) -> list[Phoneme]:
    """Convert one Yoruba word to its phoneme sequence.

    Args:
        word: A single word, already tokenized. Normalized internally.

    Returns:
        The phoneme sequence in pronunciation order.

    Raises:
        G2PError: If the word is empty, or contains characters that are not part
            of the Yoruba alphabet (digits, Latin loanword letters like ``c``,
            ``q``, ``v``, ``x``, ``z``). Such words need a manual override.
    """
    normalized = normalize(word)
    if not normalized:
        raise G2PError("Cannot convert an empty word")

    if is_noise_token(normalized):
        # Non-speech events map to MFA's spoken-noise phone, so the aligner
        # accounts for the time they occupy instead of absorbing it into a
        # neighbouring vowel.
        return [Phoneme(symbol=NOISE_PHONE)]

    chars = grapheme_clusters(normalized)
    phonemes: list[Phoneme] = []
    index = 0

    while index < len(chars):
        decomposed = decompose_char(chars[index])
        letter = decomposed.letter

        # --- Consonant digraphs (must be tested before single letters) ---
        if index + 1 < len(chars):
            pair = letter + decompose_char(chars[index + 1]).letter
            if pair == "gb":
                phonemes.append(Phoneme(symbol="gb"))
                index += 2
                continue
            if pair == "sh":  # common ASCII substitute for ṣ
                phonemes.append(Phoneme(symbol="S"))
                index += 2
                continue

        # --- Vowels, possibly nasalized by a following n ---
        if letter in ORAL_VOWELS:
            is_nasal = False
            consumed = 1
            base = letter

            if index + 1 < len(chars):
                nxt_letter = decompose_char(chars[index + 1]).letter
                if nxt_letter == SYLLABIC_NASAL and _nasalizes(chars, index + 1):
                    is_nasal = True
                    consumed = 2
                    # Only five vowels nasalize. An "n" after e or o means the
                    # sub-dot was dropped in typing, so recover the intended
                    # nasal vowel rather than invent *en / *on.
                    base = MISSING_DOT_NASAL.get(letter, letter)
                    if base != letter:
                        logger.debug(
                            "Read %r as nasal %r in %r (e and o do not nasalize)",
                            letter, base, word,
                        )
                    base = base + "n"
            phonemes.append(
                Phoneme(
                    symbol=_phone_label(base, decomposed.tone),
                    is_vowel=True,
                    tone=decomposed.tone,
                    is_nasal=is_nasal,
                )
            )
            index += consumed
            continue

        # --- Syllabic nasal: tone-bearing, treated as a vowel for pitch ---
        if letter == SYLLABIC_NASAL and _is_syllabic(chars, index):
            phonemes.append(
                Phoneme(
                    symbol=_phone_label("n", decomposed.tone),
                    is_vowel=True,
                    tone=decomposed.tone,
                    is_nasal=True,
                )
            )
            index += 1
            continue

        # --- Plain consonants ---
        if letter in CONSONANTS:
            phonemes.append(Phoneme(symbol=_phone_label(letter, None)))
            index += 1
            continue

        # Compound joiner, elision apostrophe, and stray punctuation carry no
        # sound. Elision is already reflected in the surrounding vowels.
        if letter in ("-", "'", "—", "–", "."):
            index += 1
            continue

        raise G2PError(
            f"Unmappable character {letter!r} in word {word!r}. "
            "Add a manual override to the dictionary for this word."
        )

    if not phonemes:
        raise G2PError(f"Word {word!r} produced no phonemes")

    return phonemes


def _nasalizes(chars: list[str], n_index: int) -> bool:
    """Whether ``n`` at this index nasalizes the preceding vowel.

    A tone-marked ``ǹ``/``ń`` does not: it is a syllabic nasal carrying its own
    tone, as in ``oǹdó``. Nor does an ``n`` before a vowel, which is the onset of
    the next syllable.
    """
    if decompose_char(chars[n_index]).tone is not Tone.MID:
        return False
    if n_index + 1 >= len(chars):
        return True
    return decompose_char(chars[n_index + 1]).letter not in ORAL_VOWELS


def _is_syllabic(chars: list[str], index: int) -> bool:
    """Whether an ``n`` is a tone-bearing syllabic nasal.

    True when it carries its own tone mark anywhere in the word (``oǹdó``), or
    when it begins the word and is not followed by a vowel (``ń lọ``).
    """
    if decompose_char(chars[index]).tone is not Tone.MID:
        return True
    if index != 0:
        return False
    if index + 1 >= len(chars):
        return True
    return decompose_char(chars[index + 1]).letter not in ORAL_VOWELS


class PronunciationDictionary:
    """An MFA-format pronunciation dictionary with manual override support.

    The generated file is plain text, one entry per line::

        ọmọ    O_M m O_M
        òrí    o_L r i_H

    Words may appear more than once to give pronunciation variants; MFA picks
    the best-fitting one during alignment.

    Overrides are stored in a separate TSV so that regenerating the dictionary
    from an updated vocabulary never destroys hand-corrections.
    """

    def __init__(self, overrides: dict[str, list[str]] | None = None) -> None:
        """
        Args:
            overrides: Mapping of word to a list of phone-string variants.
        """
        self.overrides: dict[str, list[str]] = overrides or {}
        self.entries: dict[str, list[str]] = {}
        self.failures: dict[str, str] = {}

    @classmethod
    def load_overrides(cls, path: Path) -> PronunciationDictionary:
        """Build a dictionary pre-loaded with manual overrides from a TSV file.

        The file has two columns: word, and a space-separated phone string.
        Repeated words become pronunciation variants. Lines starting with ``#``
        are comments.
        """
        overrides: dict[str, list[str]] = {}
        if not path.exists():
            logger.info("No override file at %s; using rules only", path)
            return cls()

        with path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                parts = raw.rstrip("\n").split("\t")
                if len(parts) < 2:
                    logger.warning("Skipping malformed override at %s:%d", path, line_no)
                    continue
                word = normalize(parts[0])
                overrides.setdefault(word, []).append(parts[1].strip())

        logger.info("Loaded %d manual overrides from %s", len(overrides), path)
        return cls(overrides=overrides)

    def build(self, vocabulary: Iterable[str]) -> None:
        """Generate entries for every word in ``vocabulary``.

        Words that fail G2P are collected in :attr:`failures` rather than
        raising, so that one bad token does not abort a 20,000-word build. The
        caller is expected to inspect and resolve them.
        """
        self.entries.clear()
        self.failures.clear()

        for word in vocabulary:
            normalized = normalize(word)
            if not normalized:
                continue

            if normalized in self.overrides:
                self.entries[normalized] = list(self.overrides[normalized])
                continue

            try:
                phonemes = grapheme_to_phonemes(normalized)
            except G2PError as exc:
                self.failures[normalized] = str(exc)
                logger.debug("G2P failed for %r: %s", normalized, exc)
                continue

            self.entries[normalized] = [" ".join(p.symbol for p in phonemes)]

        # The noise token never appears in the extracted vocabulary (the audit
        # counts it separately), but transcripts contain it, so the dictionary
        # must define it or MFA will treat it as out-of-vocabulary.
        self.entries.setdefault(NOISE_TOKEN, [NOISE_PHONE])

        logger.info(
            "Built dictionary: %d entries, %d failures",
            len(self.entries),
            len(self.failures),
        )

    def strip_tone_suffixes(self) -> None:
        """Remove ``_H``/``_M``/``_L`` from every phone, de-duplicating variants.

        The acoustic model and the dictionary must agree on the phone set. A
        model trained on tone-stripped phones cannot align against a
        tone-suffixed dictionary -- and MFA does not report this as an error, it
        simply produces meaningless alignments.
        """
        import re  # noqa: PLC0415

        stripped: dict[str, list[str]] = {}
        for word, variants in self.entries.items():
            seen: list[str] = []
            for variant in variants:
                phones = " ".join(re.sub(r"_[HML]$", "", p) for p in variant.split())
                if phones not in seen:
                    seen.append(phones)
            stripped[word] = seen
        self.entries = stripped

    def phone_inventory(self) -> set[str]:
        """Every distinct phone symbol used across all entries."""
        inventory: set[str] = set()
        for variants in self.entries.values():
            for variant in variants:
                inventory.update(variant.split())
        return inventory

    def write(self, path: Path) -> None:
        """Write the dictionary in MFA format, sorted for reproducible diffs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for word in sorted(self.entries):
                for variant in self.entries[word]:
                    handle.write(f"{word}\t{variant}\n")
        logger.info("Wrote %d dictionary entries to %s", len(self.entries), path)

    def write_failures(self, path: Path) -> None:
        """Write unmappable words to a CSV for manual review.

        This file is the worklist for building up the override table.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["word", "reason"])
            for word in sorted(self.failures):
                writer.writerow([word, self.failures[word]])
        logger.info("Wrote %d G2P failures to %s", len(self.failures), path)
