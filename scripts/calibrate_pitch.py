"""Choose the pitch analysis window from measured tone accuracy.

A vowel whose tone cannot be read is not a neutral outcome. It silently excuses
the learner: the comparison skips it, so a wrong tone there is never reported.
Observed on a learner recording of ``àbùbùtán``, whose four vowels aligned at
190, 30, 30 and 30 ms -- only the first was analysable, so three quarters of the
word's tone was unjudgeable.

The cause is arithmetic. At ``time_step_s`` 0.01 and ``edge_trim_ratio`` 0.20, a
30 ms vowel yields 18 ms of usable span and therefore one or two frames, and
``min_voiced_frames`` 3 rejects it. Short vowels are common in fluent speech, so
this is not an edge case.

Both directions matter and are measured together:

* **coverage** -- the share of vowels that get any tone at all. Raising it is the
  point of the exercise.
* **accuracy** -- of the vowels that do get a tone, how many match the tone the
  spelling requires. A looser window that reads more vowels wrongly is worse
  than a strict one that stays quiet, so coverage bought with accuracy is not a
  gain.

SLR86 is used because it is fully diacritized, making the orthography ground
truth rather than another estimate.

    python scripts/calibrate_pitch.py --limit 250
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig, PitchConfig  # noqa: E402
from yoruba_engine.datatypes import Tone  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tone_pattern  # noqa: E402
from yoruba_engine.pitch.extractor import PitchExtractor  # noqa: E402
from yoruba_engine.segmental.verifier import words_from_alignment  # noqa: E402
from yoruba_engine.tone.analyzer import ToneAnalyzer  # noqa: E402
from yoruba_engine.tone.classifier import ToneClassifier  # noqa: E402


def evaluate(config: EngineConfig, pitch_config: PitchConfig, pairs, classifier):
    """Return (coverage, accuracy, analysed, total) for one pitch setting."""
    extractor = PitchExtractor(pitch_config)
    analyzer = ToneAnalyzer(config.tone, classifier=classifier)
    analysed = correct = total = 0

    for grid, wav in pairs:
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            words, _ = extractor.extract(wav, alignment)
            words = analyzer.analyse(words)
        except Exception:
            continue
        for word in words:
            written = tone_pattern(normalize(word.word))
            if len(written) != len(word.vowels):
                continue          # G2P and the aligner disagree; not measurable
            for expected, vowel in zip(written, word.vowels):
                if expected is Tone.UNKNOWN:
                    continue
                total += 1
                if vowel.detected_tone is Tone.UNKNOWN:
                    continue
                analysed += 1
                correct += int(vowel.detected_tone == expected)
    coverage = analysed / max(total, 1)
    accuracy = correct / max(analysed, 1)
    return coverage, accuracy, analysed, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    try:
        classifier = ToneClassifier.load()
    except Exception:
        classifier = None
        print("(no trained tone classifier; using the threshold rules)")

    index = {p.stem: p for p in args.audio.rglob("*.wav")}
    pairs = [(g, index[g.stem])
             for g in sorted(args.aligned.rglob("*.TextGrid"))[: args.limit]
             if g.stem in index]
    print(f"{len(pairs)} utterances\n")

    base = config.pitch
    print(f"{'step':>6}{'trim':>6}{'min':>5}{'coverage':>10}{'accuracy':>10}"
          f"{'analysed':>10}{'usable score':>14}")
    best = None
    for step in (0.010, 0.005):
        for trim in (0.20, 0.10):
            for minimum in (3, 2):
                pitch_config = PitchConfig(
                    floor_hz=base.floor_hz, ceiling_hz=base.ceiling_hz,
                    time_step_s=step, edge_trim_ratio=trim,
                    min_voiced_frames=minimum,
                )
                coverage, accuracy, analysed, total = evaluate(
                    config, pitch_config, pairs, classifier)
                # Vowels correctly judged as a share of all vowels: rewards
                # coverage only when the extra readings are right.
                usable = coverage * accuracy
                marker = ""
                if best is None or usable > best[0]:
                    best, marker = (usable, step, trim, minimum), "  <-"
                print(f"{step:>6.3f}{trim:>6.2f}{minimum:>5}{coverage:>9.1%}"
                      f"{accuracy:>10.1%}{analysed:>10,}{usable:>13.1%}{marker}",
                      flush=True)

    print(f"\ncurrent: step={base.time_step_s} trim={base.edge_trim_ratio} "
          f"min={base.min_voiced_frames}")
    print(f"best by coverage x accuracy: step={best[1]} trim={best[2]} min={best[3]}")
    print("\n'usable score' is coverage x accuracy -- the share of all vowels that "
          "get a\ntone AND get it right, which is what the learner actually "
          "benefits from.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
