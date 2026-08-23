"""Minimal Praat TextGrid reader.

Only the long text format that MFA emits is supported, and only IntervalTiers.
This avoids a dependency on ``textgrid``/``praatio`` for what is a small,
stable format. If you already depend on praatio elsewhere, swapping this out is
a drop-in change behind :func:`parse_textgrid`.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..exceptions import AlignmentError
from ..logging_utils import get_logger
from ..datatypes import Alignment, Interval, SILENCE_LABELS

logger = get_logger("textgrid")

_NUMBER = re.compile(r"=\s*([-\d.eE+]+)")
_TEXT = re.compile(r'text\s*=\s*"(.*)"\s*$', re.DOTALL)
_TIER_NAME = re.compile(r'name\s*=\s*"(.*?)"')


def _extract_number(line: str) -> float:
    match = _NUMBER.search(line)
    if match is None:
        raise AlignmentError(f"Expected a number in TextGrid line: {line!r}")
    return float(match.group(1))


def parse_textgrid(path: Path) -> dict[str, list[Interval]]:
    """Parse a long-format TextGrid into ``{tier_name: [Interval, ...]}``.

    Args:
        path: Path to the ``.TextGrid`` file.

    Returns:
        Intervals per tier, in file order. Empty and silence-labelled intervals
        are retained here; filtering is the caller's decision.

    Raises:
        AlignmentError: If the file is missing or not a parseable long-format
            TextGrid.
    """
    if not path.exists():
        raise AlignmentError(f"TextGrid not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Praat sometimes writes UTF-16 with a BOM.
        content = path.read_text(encoding="utf-16")

    if "interval" not in content:
        raise AlignmentError(
            f"{path} contains no intervals; short-format TextGrids are not supported"
        )

    tiers: dict[str, list[Interval]] = {}
    current_tier: str | None = None
    pending: dict[str, float | str] = {}

    for raw_line in content.splitlines():
        line = raw_line.strip()

        name_match = _TIER_NAME.search(line)
        if name_match and line.startswith("name"):
            current_tier = name_match.group(1)
            tiers.setdefault(current_tier, [])
            pending = {}
            continue

        if current_tier is None:
            continue

        if line.startswith("xmin"):
            pending["xmin"] = _extract_number(line)
        elif line.startswith("xmax"):
            pending["xmax"] = _extract_number(line)
        elif line.startswith("text"):
            text_match = _TEXT.search(line)
            label = text_match.group(1) if text_match else ""
            if "xmin" in pending and "xmax" in pending:
                tiers[current_tier].append(
                    Interval(
                        label=label.strip(),
                        start_s=float(pending["xmin"]),
                        end_s=float(pending["xmax"]),
                    )
                )
            pending = {}

    if not tiers:
        raise AlignmentError(f"No tiers parsed from {path}")

    logger.debug("Parsed %s: tiers %s", path.name, list(tiers))
    return tiers


def alignment_from_textgrid(path: Path, utterance_id: str | None = None) -> Alignment:
    """Build an :class:`Alignment` from an MFA TextGrid.

    MFA names its tiers ``words`` and ``phones``, optionally prefixed with the
    speaker id (``speaker1 - words``). Both layouts are handled.

    Args:
        path: The TextGrid produced by ``mfa align``.
        utterance_id: Identifier to attach; defaults to the file stem.

    Returns:
        The alignment, with silence intervals removed.

    Raises:
        AlignmentError: If neither a word nor a phone tier can be identified.
    """
    tiers = parse_textgrid(path)

    def find_tier(suffix: str) -> list[Interval]:
        for name, intervals in tiers.items():
            if name.strip().lower().endswith(suffix):
                return intervals
        return []

    words = find_tier("words")
    phones = find_tier("phones")

    if not words and not phones:
        raise AlignmentError(
            f"{path} has tiers {list(tiers)}; expected tiers named 'words' and 'phones'"
        )

    def drop_silence(intervals: list[Interval]) -> list[Interval]:
        return [i for i in intervals if i.label.lower() not in SILENCE_LABELS]

    return Alignment(
        utterance_id=utterance_id or path.stem,
        words=drop_silence(words),
        phones=drop_silence(phones),
    )


def mark_syllabic_nasals(alignment: Alignment, transcript: str) -> Alignment:
    """Relabel syllabic nasals in an alignment, in place.

    Tone in Yoruba sits on tone-bearing units, which are vowels *and* syllabic
    nasals: ``ń`` in ``Mo ń bọ̀`` is a syllable carrying High. The acoustic model
    is tone-stripped, so ``n_H`` reaches a TextGrid as a bare ``n`` that nothing
    can tell from the consonant in ``ìnú``, and everything downstream treated it
    as a consonant.

    The cost was larger than one missing unit per nasal. ``tone_pattern`` counts
    a syllabic nasal as tone-bearing while the alignment did not, so the two
    disagreed on how many units a word had -- and a word whose written pattern
    and vowel count disagree is dropped from tone training entirely. Every word
    containing ``ń`` was being discarded, not merely under-measured.

    The TextGrid cannot recover the distinction on its own; the transcript can,
    because the G2P marks a syllabic nasal as tone-bearing. This zips the two,
    and refuses when their phone counts disagree -- a divergence would mislabel
    every phone after it, which is worse than leaving them all as consonants.

    Args:
        alignment: Phone intervals from forced alignment.
        transcript: What was said, in Yoruba orthography.

    Returns:
        The same alignment, with syllabic nasals labelled ``n~`` or ``m~``.
    """
    from ..g2p.dictionary import grapheme_to_phonemes  # noqa: PLC0415
    from ..g2p.orthography import is_noise_token, normalize, tokenize  # noqa: PLC0415

    flags: list[bool] = []
    for token in tokenize(normalize(transcript or ""), keep_noise=False):
        if is_noise_token(token):
            continue
        try:
            phonemes = grapheme_to_phonemes(token)
        except Exception:
            return alignment          # one unmappable word makes the zip unsafe
        for phoneme in phonemes:
            base = phoneme.symbol.split("_", 1)[0]
            flags.append(bool(phoneme.is_vowel) and base in {"n", "m"})

    spoken = [p for p in alignment.phones if p.label not in SILENCE_LABELS]
    if len(flags) != len(spoken):
        return alignment

    for phone, syllabic in zip(spoken, flags):
        if syllabic and not phone.label.endswith("~"):
            phone.label = phone.label + "~"
    return alignment
