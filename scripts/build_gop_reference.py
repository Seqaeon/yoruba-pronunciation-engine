"""Accumulate per-phone goodness statistics from correctly-read corpus speech.

A raw likelihood cannot say whether -4.0 is good. Measured against native
production of the same phone it can. This aligns the corpus with the shipped
model, keeps MFA's working directory, and summarises the per-frame goodness of
every phone it produced.

    python scripts/build_gop_reference.py --corpus data/mfa_corpus
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.gop import GOPReference, read_scores  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--temporary", type=Path, default=Path("/tmp/yoruba_gop_reference"))
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "gop_reference.json")
    parser.add_argument("--reuse", action="store_true",
                        help="Skip alignment and read an existing run")
    args = parser.parse_args()

    config = EngineConfig()
    if not args.reuse:
        executable = Path(config.mfa.executable)
        environment = {**os.environ,
                       "PATH": f"{executable.parent}{os.pathsep}{os.environ['PATH']}",
                       "TERM": "dumb"}
        output = args.temporary / "aligned"
        print(f"Aligning {args.corpus} (keeping artifacts in {args.temporary})")
        result = subprocess.run(
            [str(executable), "align", str(args.corpus), str(config.mfa.dictionary),
             str(config.mfa.acoustic_model), str(output),
             "--temporary_directory", str(args.temporary),
             "--no_clean", "--num_jobs", "12", "--quiet"],
            capture_output=True, text=True, env=environment,
        )
        if result.returncode != 0:
            print((result.stderr or result.stdout)[-800:])
            return 1

    scores = read_scores(args.temporary, args.corpus.name)
    print(f"read scores for {len(scores):,} utterances")
    if not scores:
        return 1

    per_phone: dict[str, list[float]] = defaultdict(list)
    for phones in scores.values():
        for phone in phones:
            per_phone[phone.phone].append(phone.per_frame)

    reference = GOPReference({
        phone: (statistics.mean(values), statistics.pstdev(values))
        for phone, values in per_phone.items()
        if len(values) >= 20          # too few and the deviation is noise
    })
    reference.save(args.out)

    print(f"\n{'phone':<8}{'n':>8}{'mean':>9}{'sd':>8}")
    for phone in sorted(reference.statistics, key=lambda p: reference.statistics[p][0]):
        mean, deviation = reference.statistics[phone]
        print(f"{phone:<8}{len(per_phone[phone]):>8,}{mean:>9.3f}{deviation:>8.3f}")
    skipped = sorted(p for p, v in per_phone.items() if len(v) < 20)
    if skipped:
        print(f"\ntoo few samples for: {', '.join(skipped)}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
