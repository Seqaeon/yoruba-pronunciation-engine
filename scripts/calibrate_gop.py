"""Calibrate the GOP threshold from both sides.

Each corpus utterance is aligned twice: against its true transcript, and against
one with a single word swapped for a different real word. Words from the true
pass are known-correct, so flagging them is a false alarm; the swapped word is
known-wrong, so missing it is a miss.

GOP is scored as the **mean** z-score across a word's phones rather than the
worst. The worst is too noisy: a correctly-read ``àbùbùtán`` produced a single
phone at -15.6 while the word as a whole sat at -3.4, so a worst-phone rule
would fail correct speech.

    python scripts/calibrate_gop.py --limit 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.gop import GOPReference, read_scores  # noqa: E402
from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402


def stage(config: EngineConfig, root: Path, limit: int, seed: int = 0):
    """Write two corpora -- true transcripts and one-word-swapped."""
    shutil.rmtree(root, ignore_errors=True)
    rows = []
    with config.line_index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                rows.append((row[0], row[1], row[2]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    vocabulary = sorted({normalize(t) for _, _, text in rows
                         for t in tokenize(text, keep_noise=False)})

    swapped = {}
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
        position = rng.randrange(len(words))
        replacement = rng.choice(vocabulary)
        if replacement == words[position]:
            continue
        for name, sequence in (("true", words),
                               ("swap", words[:position] + [replacement]
                                + words[position + 1:])):
            directory = root / name / speaker
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, directory / f"{utt}.wav")
            (directory / f"{utt}.lab").write_text(" ".join(sequence), encoding="utf-8")
        swapped[utt] = position
        staged += 1
    return swapped


def align(config: EngineConfig, corpus: Path, temporary: Path, output: Path):
    executable = Path(config.mfa.executable)
    environment = {**os.environ,
                   "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
                   "TERM": "dumb"}
    return subprocess.run(
        [str(executable), "align", str(corpus), str(config.mfa.dictionary),
         str(config.mfa.acoustic_model), str(output),
         "--temporary_directory", str(temporary),
         "--no_clean", "--num_jobs", "12", "--quiet"],
        capture_output=True, text=True, env=environment,
    )


def word_scores(reference: GOPReference, temporary: Path, corpus_name: str,
                aligned: Path):
    """Mean GOP z-score per word, keyed by (utterance, word index)."""
    scores = read_scores(temporary, corpus_name)
    out = {}
    for grid in sorted(aligned.rglob("*.TextGrid")):
        phones = scores.get(grid.stem)
        if not phones:
            continue
        alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
        for index, word in enumerate(alignment.words):
            inside = [p for p in phones
                      if word.start_s - 0.02 <= p.start_s and p.end_s <= word.end_s + 0.02]
            zs = [z for z in (reference.z_score(p) for p in inside) if z is not None]
            if zs:
                out[(grid.stem, index)] = statistics.mean(zs)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--root", type=Path, default=Path("/tmp/yoruba_gop_calibration"))
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    reference = GOPReference.load(PROJECT_ROOT / "models" / "gop_reference.json")
    if not reference.statistics:
        print("No GOP reference; run scripts/build_gop_reference.py first")
        return 1

    print(f"Staging {args.limit} utterances (true and one-word-swapped)")
    swapped = stage(config, args.root, args.limit)
    print(f"staged {len(swapped)}")

    results = {}
    for name in ("true", "swap"):
        result = align(config, args.root / name, args.root / f"tmp_{name}",
                       args.root / f"aligned_{name}")
        if result.returncode != 0:
            print((result.stderr or result.stdout)[-500:])
            return 1
        results[name] = word_scores(reference, args.root / f"tmp_{name}", name,
                                    args.root / f"aligned_{name}")
        print(f"  {name}: {len(results[name]):,} scored words")

    correct = [z for (utt, index), z in results["true"].items() if utt in swapped]
    wrong = [z for (utt, index), z in results["swap"].items()
             if utt in swapped and index == swapped[utt]]
    print(f"\n{len(correct):,} known-correct words, {len(wrong):,} known-wrong\n")
    if not wrong:
        print("No swapped words scored; nothing to calibrate")
        return 1

    print(f"{'threshold':>10}{'false alarms':>14}{'substitutions caught':>22}")
    best = None
    for threshold in (-1.0, -1.5, -2.0, -2.5, -3.0, -3.5, -4.0, -5.0, -6.0):
        false_alarm = sum(1 for z in correct if z < threshold) / max(len(correct), 1)
        recall = sum(1 for z in wrong if z < threshold) / max(len(wrong), 1)
        score = recall - 2 * false_alarm
        if best is None or score > best[0]:
            best = (score, threshold, false_alarm, recall)
        print(f"{threshold:>10.1f}{false_alarm:>13.1%}{recall:>21.1%}")

    _, threshold, false_alarm, recall = best
    print(f"\ncorrect words: median {statistics.median(correct):+.2f}, "
          f"5th pct {sorted(correct)[len(correct)//20]:+.2f}")
    print(f"wrong words  : median {statistics.median(wrong):+.2f}")
    print(f"\nbest (recall - 2x false alarms): threshold {threshold}"
          f"  ->  {false_alarm:.1%} false, {recall:.1%} caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
