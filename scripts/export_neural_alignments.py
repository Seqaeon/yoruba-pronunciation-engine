"""Write neural-aligner alignments as TextGrids, so models can be refitted on them.

The vowel classifier was fitted on 575,125 vowels cut by MFA and is now fed spans
cut by the neural aligner. Measured on held-out native readings, that costs it
about 3.7 points -- 92.4% on the spans it trained on against 88.7% on the spans
it is given. The boundaries agree within 20 ms on 86% of edges, but the
classifier mean-pools encoder frames over exactly those edges, so a systematic
shift changes what it reads.

Refitting on the deployed spans is the fix, and every training script here takes
TextGrids. So rather than teaching each of them to call an aligner, this writes
the neural alignments out in the format they already read.

At about 0.04 s per utterance the whole corpus is roughly two hours, which is
also a reminder of the size of the change being measured: MFA would take three
weeks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def write_textgrid(path: Path, words, phones, duration: float) -> None:
    """Emit the long TextGrid format MFA produces and this project parses."""
    def tier(name, intervals, index):
        lines = [f'    item [{index}]:',
                 '        class = "IntervalTier"',
                 f'        name = "{name}"',
                 '        xmin = 0',
                 f'        xmax = {duration}',
                 f'        intervals: size = {len(intervals)}']
        for position, interval in enumerate(intervals, 1):
            lines += [f'        intervals [{position}]:',
                      f'            xmin = {interval.start_s}',
                      f'            xmax = {interval.end_s}',
                      f'            text = "{interval.label}"']
        return lines

    head = ['File type = "ooTextFile"', 'Object class = "TextGrid"', '',
            'xmin = 0', f'xmax = {duration}', 'tiers? <exists>', 'size = 2',
            'item []:']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(head + tier("words", words, 1)
                              + tier("phones", phones, 2)) + "\n",
                    encoding="utf-8")


def main() -> int:
    import soundfile as sf

    from yoruba_engine.alignment.neural import NeuralAligner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "neural_aligned")
    parser.add_argument("--limit", type=int,
                        help="Only this many utterances. The default is all of "
                             "them; the corpus is about two hours at 0.04 s each.")
    args = parser.parse_args()

    aligner = NeuralAligner()
    aligner.check_available()

    labs = sorted(args.corpus.rglob("*.lab"))
    if args.limit:
        labs = labs[:args.limit]
    print(f"{len(labs)} utterances", flush=True)

    written = skipped = 0
    started = time.perf_counter()
    for position, lab in enumerate(labs, 1):
        wav = lab.with_suffix(".wav")
        target = args.out / lab.parent.name / f"{lab.stem}.TextGrid"
        if target.exists():
            written += 1
            continue
        if not wav.exists():
            skipped += 1
            continue
        transcript = lab.read_text(encoding="utf-8").strip()
        if not transcript:
            skipped += 1
            continue
        try:
            alignment = aligner.align_single(wav, transcript, utterance_id=lab.stem)
            info = sf.info(str(wav))
            write_textgrid(target, alignment.words, alignment.phones,
                           info.frames / info.samplerate)
            written += 1
        except Exception:
            # One unalignable utterance must not stop a two-hour job. They are
            # counted, so a silent collapse shows up as a number.
            skipped += 1
        if position % 5000 == 0:
            rate = position / (time.perf_counter() - started)
            print(f"  {position}/{len(labs)}  {rate:.0f}/s  "
                  f"{written} written, {skipped} skipped", flush=True)

    print(f"\n{written} TextGrids in {args.out}, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
