"""Calibrate the vowel-suspicion thresholds from both sides.

The segmental check answers "did the learner produce the vowels this word
needs?" It is the only mechanism that can catch a substitution the learner
invented -- ``àbùbùtu`` for ``àbùbùtán`` is not a Yoruba word, so the
minimal-pair decoder can never name it, but the check does not need a name. It
only needs to notice the vowels are wrong.

Both error directions are measured on the same corpus. Each utterance is
aligned twice: once against its true transcript, and once against a transcript
with one word swapped for a different real word of the same vowel count. Vowels
from the true pass are known-correct, so flagging them is a false alarm; vowels
of the swapped word are known-wrong, so missing them is a miss.

    python scripts/calibrate_suspect.py --limit 200
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
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.dictionary import grapheme_to_phonemes  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402
from yoruba_engine.segmental.classifier import VowelClassifier, vowel_features  # noqa: E402
from yoruba_engine.segmental.formants import FormantExtractor, normalize_vowel_space  # noqa: E402
from yoruba_engine.segmental.mfcc import MFCCExtractor  # noqa: E402


def vowels_of(word: str):
    try:
        return [p.symbol.split("_")[0] for p in grapheme_to_phonemes(word) if p.is_vowel]
    except Exception:
        return []


def build(config: EngineConfig, staging: Path, limit: int, seed: int = 0):
    """Stage two corpora: true transcripts, and one word swapped per utterance."""
    shutil.rmtree(staging, ignore_errors=True)
    rows = []
    with config.line_index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                rows.append((row[0], row[1], row[2]))
    rng = random.Random(seed)
    rng.shuffle(rows)

    # Index the vocabulary by vowel count so a swap keeps the word length
    # plausible -- otherwise the duration check would catch it and the
    # segmental check would never be exercised.
    by_shape = defaultdict(list)
    for _, _, text in rows:
        for token in tokenize(text, keep_noise=False):
            word = normalize(token)
            shape = tuple(vowels_of(word))
            if shape:
                by_shape[len(shape)].append(word)

    swapped = {}
    staged = 0
    for utt, speaker, text in rows:
        if staged >= limit:
            break
        source = config.audio_dir / f"{utt}.wav"
        if not source.exists():
            continue
        words = [normalize(t) for t in tokenize(text, keep_noise=False)]
        candidates = [i for i, w in enumerate(words) if len(vowels_of(w)) >= 2]
        if len(words) < 3 or not candidates:
            continue
        position = rng.choice(candidates)
        original = words[position]
        pool = [w for w in by_shape[len(vowels_of(original))]
                if w != original and vowels_of(w) != vowels_of(original)]
        if not pool:
            continue
        replacement = rng.choice(pool)

        for name, sequence in (("true", words),
                               ("swapped", words[:position] + [replacement]
                                + words[position + 1:])):
            directory = staging / name / speaker
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, directory / f"{utt}.wav")
            (directory / f"{utt}.lab").write_text(" ".join(sequence), encoding="utf-8")
        swapped[utt] = (position, original, replacement)
        staged += 1
    return swapped


def align(config: EngineConfig, corpus: Path, out: Path):
    executable = Path(config.mfa.executable)
    environment = {**os.environ,
                   "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
                   "TERM": "dumb"}
    subprocess.run(
        [str(executable), "align", str(corpus), str(config.mfa.dictionary),
         str(config.mfa.acoustic_model), str(out),
         "--num_jobs", "12", "--clean", "--quiet"],
        capture_output=True, text=True, env=environment,
    )
    return sorted(out.rglob("*.TextGrid"))


def measure(textgrids, audio_root: Path, classifier, target_index=None):
    """Return per-word lists of P(expected) for each measured vowel."""
    formant_extractor, mfcc_extractor = FormantExtractor(), MFCCExtractor()
    position = {label: i for i, label in enumerate(classifier.labels)}
    audio_index = {p.stem: p for p in audio_root.rglob("*.wav")}
    out = defaultdict(dict)

    for textgrid in textgrids:
        audio = audio_index.get(textgrid.stem)
        if audio is None:
            continue
        try:
            alignment = alignment_from_textgrid(textgrid, utterance_id=textgrid.stem)
            formants = formant_extractor.analyse(audio)
            times, frames = mfcc_extractor.analyse(audio)
        except Exception:
            continue
        measured = []
        for phone in alignment.phones:
            if not is_vowel_phone(phone.label):
                continue
            f = formant_extractor.measure_segment(formants, phone.start_s, phone.end_s)
            c = mfcc_extractor.measure_segment(times, frames, phone.start_s, phone.end_s)
            if f and c:
                measured.append((phone, f, c))
        if not measured:
            continue
        f1, f2 = normalize_vowel_space([m[1] for m in measured])
        proba = classifier.model.predict_proba(
            [vowel_features(f, f1, f2) + list(c) for _, f, c in measured])
        for index, word in enumerate(alignment.words):
            values = []
            for (phone, _, _), distribution in zip(measured, proba):
                if not (word.start_s <= (phone.start_s + phone.end_s) / 2 <= word.end_s):
                    continue
                expected = phone.label.split("_")[0]
                if expected in position:
                    values.append(float(distribution[position[expected]]))
            if values:
                out[textgrid.stem][index] = values
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--staging", type=Path, default=Path("/tmp/yoruba_suspect_calibration"))
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    print(f"Staging {args.limit} utterances, true and one-word-swapped")
    swapped = build(config, args.staging, args.limit)
    print(f"staged {len(swapped)}")

    classifier = VowelClassifier.load()
    true_grids = align(config, args.staging / "true", args.staging / "aligned_true")
    swap_grids = align(config, args.staging / "swapped", args.staging / "aligned_swapped")
    print(f"aligned {len(true_grids)} true, {len(swap_grids)} swapped\n")

    true_words = measure(true_grids, args.staging / "true", classifier)
    swap_words = measure(swap_grids, args.staging / "swapped", classifier)

    correct, wrong = [], []
    for utt, (position, _, _) in swapped.items():
        for index, values in true_words.get(utt, {}).items():
            correct.append(values)
        for index, values in swap_words.get(utt, {}).items():
            if index == position:
                wrong.append(values)

    print(f"{len(correct):,} known-correct words, {len(wrong):,} known-wrong words\n")

    def flags(values, suspect, weak, min_weak):
        return any(v < suspect for v in values) or sum(v < weak for v in values) >= min_weak

    print(f"{'suspect':>9}{'weak':>7}{'min':>5}{'false alarms':>14}{'substitutions caught':>22}")
    best = None
    for suspect in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
        for weak, min_weak in ((0.05, 2), (0.15, 2), (0.30, 2)):
            if weak < suspect:
                continue
            false_alarm = sum(flags(v, suspect, weak, min_weak) for v in correct) / max(len(correct), 1)
            recall = sum(flags(v, suspect, weak, min_weak) for v in wrong) / max(len(wrong), 1)
            score = recall - 2 * false_alarm       # false alarms hurt a learner twice over
            if best is None or score > best[0]:
                best = (score, suspect, weak, min_weak, false_alarm, recall)
            print(f"{suspect:>9.2f}{weak:>7.2f}{min_weak:>5}{false_alarm:>13.1%}{recall:>21.1%}")

    print(f"\ncurrent: suspect={config.segmental.suspect_probability} "
          f"weak={config.segmental.weak_probability} "
          f"min_weak={config.segmental.min_weak_vowels}")
    _, suspect, weak, min_weak, false_alarm, recall = best
    print(f"best (recall - 2x false alarms): suspect={suspect} weak={weak} "
          f"min_weak={min_weak}  ->  {false_alarm:.1%} false, {recall:.1%} caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
