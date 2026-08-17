"""Goodness of Pronunciation, read from the aligner's own acoustic scores.

The engine spent a long time judging vowels with a bolt-on classifier over 39
hand-designed Praat features, which tops out near 60% on studio speech. That was
never necessary: the acoustic model has to score every phone against every frame
in order to align at all, and MFA keeps the result. Running an alignment with a
persistent temporary directory leaves::

    alignment/phone_intervals.csv    begin, end, phone_goodness, phone_id
    <corpus>.db                      phone id -> label, utterance id -> file

``phone_goodness`` is the log-likelihood the model assigns to that phone over
that span, from a model trained on 38 hours -- far more evidence than 39
numbers measured from a spectrum.

Two normalisations make it usable:

**Per frame.** The raw value scales with duration: a 0.82 s span scores about
-58 while a 0.04 s span scores -83. Dividing by the frame count makes long and
short phones comparable.

**Per phone type.** Phones differ in how well the model fits them regardless of
who is speaking -- a stop is inherently less predictable than a long vowel. A
raw likelihood cannot say whether -4.0 is good, but -4.0 against a reference
mean of -2.4 with a standard deviation of 0.5 is clearly poor. Reference
statistics are accumulated from correctly-read corpus speech by
``scripts/build_gop_reference.py``, and the score reported here is a z-score
against them: negative means worse than a native speaker typically produces.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..logging_utils import get_logger

logger = get_logger("gop")

#: Kaldi's frame shift. Used to convert a span in seconds to a frame count.
FRAME_SHIFT_S = 0.01

#: Phones that carry no pronunciation judgement.
NON_SPEECH = frozenset({"sil", "spn", "<eps>", ""})


@dataclass
class PhoneScore:
    """One phone's acoustic score within an utterance."""

    phone: str
    start_s: float
    end_s: float
    #: Total log-likelihood over the span, as MFA reports it.
    goodness: float

    @property
    def frames(self) -> int:
        return max(int(round((self.end_s - self.start_s) / FRAME_SHIFT_S)), 1)

    @property
    def per_frame(self) -> float:
        """Log-likelihood per frame, comparable across phone durations."""
        return self.goodness / self.frames


def read_scores(temporary_directory: Path, corpus_name: str) -> dict[str, list[PhoneScore]]:
    """Read per-phone scores from an MFA run left on disk.

    Args:
        temporary_directory: What was passed to ``--temporary_directory``.
        corpus_name: The corpus directory's name, which MFA uses to name both
            its working subdirectory and its database.

    Returns:
        Mapping of file stem to its phones in time order. Empty if the run was
        cleaned up or never wrote intervals.
    """
    root = temporary_directory / corpus_name
    intervals = root / "alignment" / "phone_intervals.csv"
    database = root / f"{corpus_name}.db"
    if not intervals.exists() or not database.exists():
        logger.info("No alignment artifacts under %s; run with "
                    "--temporary_directory and --no_clean to keep them", root)
        return {}

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        phone_labels = {row[0]: row[1] for row in
                        connection.execute("SELECT id, phone FROM phone")}
        utterances = {row[0]: row[1] for row in connection.execute(
            "SELECT u.id, f.name FROM utterance u JOIN file f ON u.file_id = f.id")}
    finally:
        connection.close()

    scores: dict[str, list[PhoneScore]] = {}
    with intervals.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = utterances.get(int(row["utterance_id"]))
            label = phone_labels.get(int(row["phone_id"]), "")
            if name is None or label in NON_SPEECH:
                continue
            goodness = row.get("phone_goodness")
            if goodness in (None, ""):
                continue
            scores.setdefault(name, []).append(PhoneScore(
                phone=label,
                start_s=float(row["begin"]),
                end_s=float(row["end"]),
                goodness=float(goodness),
            ))

    for phones in scores.values():
        phones.sort(key=lambda p: p.start_s)
    return scores


class GOPReference:
    """Per-phone distribution of per-frame goodness on correct speech."""

    def __init__(self, statistics: dict[str, tuple[float, float]] | None = None) -> None:
        #: phone -> (mean, standard deviation) of per-frame goodness.
        self.statistics = statistics or {}

    @classmethod
    def load(cls, path: Path) -> GOPReference:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.info("No GOP reference at %s (%s); scores will be raw "
                        "per-frame likelihoods, not z-scores", path, exc)
            return cls()
        return cls({k: (v["mean"], v["sd"]) for k, v in raw.items()})

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {k: {"mean": m, "sd": s} for k, (m, s) in sorted(self.statistics.items())},
            indent=1), encoding="utf-8")
        return path

    def z_score(self, score: PhoneScore) -> float | None:
        """How many standard deviations below native production this phone is.

        Returns ``None`` for a phone with no reference statistics, which must be
        treated as "no opinion" rather than as a pass or a fail.
        """
        entry = self.statistics.get(score.phone)
        if entry is None:
            return None
        mean, deviation = entry
        if deviation <= 0:
            return None
        return (score.per_frame - mean) / deviation
