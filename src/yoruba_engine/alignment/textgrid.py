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
