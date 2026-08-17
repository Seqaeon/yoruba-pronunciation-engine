"""Calibrate the omission threshold from both sides at once.

``was_spoken`` decides whether an aligned word corresponds to real speech, and
it can fail in two directions that trade against each other:

* Too strict, and correctly spoken words are reported missing -- the complaint
  that started this, where a correct reading was told ``àbùbùtán`` was absent.
* Too loose, and a genuinely omitted word passes, so an attempt with two wrong
  words scores 100% text similarity, which is worse feedback than silence.

Measuring only the first is what let the threshold drift. This measures both on
the same corpus by aligning each utterance against a transcript with one extra
word spliced in: the speaker demonstrably did not say it, so every inserted word
is a known omission, and every original word is known-spoken.

    python scripts/calibrate_omissions.py --limit 250
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.durations import WordDurationModel  # noqa: E402
from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402


def build_corpus(config: EngineConfig, staging: Path, limit: int, seed: int = 0):
    """Copy utterances, splicing one unspoken word into each transcript."""
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "corpus").mkdir(parents=True)

    rows = []
    with config.line_index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                rows.append((row[0], row[1], row[2]))
    rng = random.Random(seed)
    rng.shuffle(rows)

    # Draw filler from the corpus vocabulary so it is always a real Yoruba word
    # the dictionary covers -- an unknown word would fail alignment outright and
    # measure nothing.
    vocabulary = sorted({
        normalize(t) for _, _, text in rows for t in tokenize(text, keep_noise=False)
    })

    inserted = {}
    staged = 0
    for utt, speaker, text in rows:
        if staged >= limit:
            break
        source = config.audio_dir / f"{utt}.wav"
        if not source.exists():
            continue
        words = [normalize(t) for t in tokenize(text, keep_noise=False)]
        if len(words) < 3:
            continue
        filler = rng.choice(vocabulary)
        if filler in words:
            continue
        position = rng.randrange(1, len(words))
        spliced = words[:position] + [filler] + words[position:]

        directory = staging / "corpus" / speaker
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, directory / f"{utt}.wav")
        (directory / f"{utt}.lab").write_text(" ".join(spliced), encoding="utf-8")
        inserted[utt] = (filler, position)
        staged += 1
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--staging", type=Path,
                        default=Path("/tmp/yoruba_omission_calibration"))
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    print(f"Splicing one unspoken word into {args.limit} utterances")
    inserted = build_corpus(config, args.staging, args.limit)
    print(f"staged {len(inserted)} utterances")

    aligned = args.staging / "aligned"
    executable = Path(config.mfa.executable)
    environment = {**os.environ,
                   "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
                   "TERM": "dumb"}
    result = subprocess.run(
        [str(executable), "align", str(args.staging / "corpus"),
         str(config.mfa.dictionary), str(config.mfa.acoustic_model), str(aligned),
         "--num_jobs", "12", "--clean", "--quiet"],
        capture_output=True, text=True, env=environment,
    )
    textgrids = sorted(aligned.rglob("*.TextGrid"))
    print(f"aligned {len(textgrids)} (exit {result.returncode})\n")
    if not textgrids:
        print((result.stderr or result.stdout)[-600:])
        return 1

    durations = WordDurationModel.load()
    spoken, omitted = [], []
    for textgrid in textgrids:
        entry = inserted.get(textgrid.stem)
        if entry is None:
            continue
        filler, position = entry
        alignment = alignment_from_textgrid(textgrid, utterance_id=textgrid.stem)
        for index, word in enumerate(alignment.words):
            phones = [p for p in alignment.phones
                      if word.start_s <= (p.start_s + p.end_s) / 2 <= word.end_s]
            record = (word.end_s - word.start_s, len(phones))
            (omitted if index == position else spoken).append(record)

    print(f"{len(spoken):,} known-spoken words, {len(omitted):,} known-omitted\n")
    print(f"{'fraction':>9}{'false alarms':>14}{'omissions caught':>18}")
    best = None
    for fraction in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
        durations.fraction = fraction
        false_alarm = sum(1 for d, n in spoken if not durations.was_spoken(d, n)) / max(len(spoken), 1)
        recall = sum(1 for d, n in omitted if not durations.was_spoken(d, n)) / max(len(omitted), 1)
        score = recall - false_alarm          # equal weight; adjust to taste
        flag = ""
        if best is None or score > best[1]:
            best, flag = (fraction, score), ""
        print(f"{fraction:>9.1f}{false_alarm:>13.1%}{recall:>17.1%}")
    print(f"\ncurrent setting: {EngineConfig().scoring.min_duration_fraction}")
    print(f"best recall-minus-false-alarms at fraction = {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
