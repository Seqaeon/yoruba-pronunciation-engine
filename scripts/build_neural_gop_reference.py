"""Per-phone goodness statistics for the neural aligner.

A raw score cannot say whether -0.8 is good. Measured against native production
of the same phone it can, which is what ``models/gop_reference.json`` provides
for MFA: a mean and standard deviation per phone, from correctly-read corpus
speech.

**Those numbers do not transfer.** MFA reports a GMM log-likelihood over a
phone's span; :class:`~yoruba_engine.alignment.neural.NeuralAligner` reports the
mean log-posterior of the aligned phone. Both answer "how well did the audio fit
this phone" and neither is on the other's scale -- the GMM figure is unbounded
below and typically around -4 to -5, while a log-posterior is bounded above by
zero. Using MFA's reference against neural scores would compare a phone to a
distribution it was never drawn from, and the GOP check would flag or excuse
words for arithmetic reasons.

So this rebuilds the reference the same way, from the same kind of speech, with
the neural aligner in place of MFA.

    python scripts/build_neural_gop_reference.py --limit 3000

Corpus speech is the right source for the same reason it was for MFA: these are
people reading their own language correctly, so the spread of scores describes
normal production rather than error.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    from yoruba_engine.alignment.neural import SILENCE, NeuralAligner  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "frame_aligner.pt")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "models" / "gop_reference_neural.json")
    parser.add_argument("--limit", type=int, default=3000,
                        help="How many utterances to measure")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Phones with fewer observations are omitted rather "
                             "than given a statistic nothing supports")
    parser.add_argument("--min-frames", type=int, default=2)
    args = parser.parse_args()

    aligner = NeuralAligner(model_path=args.model, min_frames=args.min_frames)
    labs = sorted(args.corpus.rglob("*.lab"))[:args.limit]
    if not labs:
        print(f"No transcripts under {args.corpus}", file=sys.stderr)
        return 1
    print(f"{len(labs)} utterances")

    scores: dict[str, list[float]] = defaultdict(list)
    aligned = failed = 0
    for position, lab in enumerate(labs, 1):
        wav = lab.with_suffix(".wav")
        if not wav.exists():
            failed += 1
            continue
        transcript = lab.read_text(encoding="utf-8").strip()
        if not transcript:
            failed += 1
            continue
        try:
            alignment = aligner.align_single(wav, transcript,
                                             utterance_id=lab.stem, collect_gop=True)
        except Exception:
            failed += 1
            continue
        for score in alignment.phone_scores:
            if score.phone != SILENCE:
                scores[score.phone].append(float(score.goodness))
        aligned += 1
        if position % 500 == 0:
            print(f"  {position}/{len(labs)}  {aligned} aligned, {failed} skipped",
                  flush=True)

    reference = {}
    thin = []
    for phone, values in sorted(scores.items()):
        if len(values) < args.min_samples:
            thin.append((phone, len(values)))
            continue
        reference[phone] = {
            "mean": statistics.fmean(values),
            "sd": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "samples": len(values),
        }

    args.out.write_text(json.dumps(reference, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n{aligned} utterances aligned, {failed} skipped")
    print(f"{len(reference)} phones with at least {args.min_samples} observations")
    if thin:
        print(f"omitted for too few observations: "
              f"{', '.join(f'{p}({n})' for p, n in thin)}")
    print(f"\nWrote {args.out}")

    shipped = ROOT / "models" / "gop_reference.json"
    if shipped.exists():
        mfa = json.loads(shipped.read_text(encoding="utf-8"))
        shared = sorted(set(mfa) & set(reference))[:6]
        print("\nthe two scales, on the same phones:")
        print(f"  {'phone':<7}{'MFA mean':>10}{'MFA sd':>8}"
              f"{'neural mean':>13}{'neural sd':>11}")
        for phone in shared:
            print(f"  {phone:<7}{mfa[phone]['mean']:>10.2f}{mfa[phone]['sd']:>8.2f}"
                  f"{reference[phone]['mean']:>13.2f}{reference[phone]['sd']:>11.2f}")
        print("\nThey are not comparable, which is the point of rebuilding it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
