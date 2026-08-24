"""Cut the igba minimal set out of the source video.

These five words differ in nothing but tone, so they are the sharpest test the
app has of whether the engine hears tone at all::

    Igba   Mid-Mid    two hundred
    Ìgbá   Low-High   garden egg
    Igbá   Mid-High   calabash
    Igbà   Mid-Low    the climbing rope a palm-wine tapper uses
    Ìgbà   Low-Low    time, season

Getting them out of an 80 s teaching video took three wrong attempts, and the
reason each failed is worth keeping, because they failed at different stages.

**The first attempt cut a fixed window** around each word and left the
neighbours in: a one-second `ìgbà` still contained `bí`, `tẹ` and `bá`, and the
engine scored the learner on all four.

**The second cut at the aligner's word boundary**, which is the right technique
and still left remnants -- because the window it was given was in the wrong
place.

**The third trimmed by energy**, looking for a pause before the word. That is
wrong for this word specifically: `igba` contains /gb/, a stop, and a stop
closure *is* a silence inside the word. Cutting at the quiet part threw away the
first vowel and turned `ìgbá` into `gbá`.

All three were refinements of a cut whose location came from eyeballing a
waveform, recorded in a table of timestamps. The garden egg entry pointed at
28.7 s, which is in the middle of the *two hundred* explanation, so no amount of
better trimming could have produced the right word.

So this locates the words by evidence instead:

1. **The regions come from the English glosses.** It is a teaching video, so the
   speaker says each word and then explains it -- "two hundred", "garden egg",
   "calabash", "climbing rope" -- and those English words survive transcription
   even when the Yoruba around them does not.

2. **The tokens come from the /gb/ burst.** Every one of these words contains
   /gb/, which is rare and acoustically distinctive, and the frame aligner this
   project already trains gives it a posterior above 0.95 where it occurs. That
   finds all 42 `gb` bursts in the video without any guessing about timing.

3. **The boundaries come from the frame labels.** Around each burst, the `i` run
   before and the `a` run after are the word, and their edges are the cut.

Which burst is which sense was then confirmed by a native speaker listening to
the candidates, and those choices are what ``OCCURRENCES`` records below. Run
with ``--candidates`` to regenerate the full set for review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16000
FRAME_S = 0.02

#: The confirmed token for each sense, as the time of its /gb/ burst. These are
#: not eyeballed: each is a posterior peak the aligner found, cut automatically,
#: and then chosen by ear from the candidates in its region.
OCCURRENCES = [
    ("igba_mid_mid_200",           "Igba", 22.68),
    ("igba_low_high_garden_egg",   "Ìgbá", 29.16),
    ("igba_mid_high_calabash",     "Igbá", 36.28),
    ("igba_mid_low_climbing_rope", "Igbà", 40.40),
    ("igba_low_low_time",          "Ìgbà", 49.04),
]

#: Where each sense is explained, from the English gloss. Only used by
#: ``--candidates``, to group what it writes.
REGIONS = [
    ("200",           20.0, 28.0),
    ("garden_egg",    28.0, 35.0),
    ("calabash",      35.0, 39.0),
    ("climbing_rope", 39.0, 44.0),
    ("time",          47.0, 54.0),
]

#: A burst has to be at least this confident to be considered a /gb/.
BURST_FLOOR = 0.30


def frame_posteriors(aligner, samples):
    """Per-frame log posteriors over the aligner's phone set."""
    torch = aligner._torch
    with torch.inference_mode():
        batch = torch.tensor(samples, dtype=aligner._dtype).unsqueeze(0)
        hidden = aligner._encoder(batch.to(aligner._resolved_device)).last_hidden_state
        logits = aligner._head(hidden[0].to(torch.float32))
        return torch.log_softmax(logits, -1).cpu().numpy()


def bursts(probability, floor=BURST_FLOOR, apart=5):
    """Local maxima of P(gb), keeping the strongest of any close pair."""
    found: list[int] = []
    for i in range(1, len(probability) - 1):
        if probability[i] <= floor:
            continue
        if probability[i] < probability[i - 1] or probability[i] <= probability[i + 1]:
            continue
        if found and i - found[-1] < apart:
            if probability[i] > probability[found[-1]]:
                found[-1] = i
        else:
            found.append(i)
    return found


def word_span(labels, phones, burst):
    """The frame range of the whole word around one /gb/ burst.

    Walks out from the burst through the closure, then back over the `i` run and
    forward over the `a` run. Those runs are the two vowels, and their outer
    edges are where the word begins and ends.
    """
    closure = burst
    while closure > 0 and phones[labels[closure - 1]] in ("gb", "b", "g"):
        closure -= 1
    first = closure
    while first > 0 and phones[labels[first - 1]] in ("i", "in"):
        first -= 1

    release = burst + 1
    while release < len(labels) and phones[labels[release]] in ("gb", "b", "g"):
        release += 1
    last = release
    while last < len(labels) and phones[labels[last]] in ("a", "an"):
        last += 1

    if closure - first < 2 or last - release < 2:
        return None
    return first, last


def main() -> int:
    import numpy as np
    import soundfile as sf

    from yoruba_engine.alignment.neural import NeuralAligner
    from yoruba_engine.audio import read_mono

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "igba.mp4")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "igba_clips")
    parser.add_argument("--pad", type=float, default=0.02,
                        help="Kept either side of the word, so an onset is not clipped")
    parser.add_argument("--candidates", action="store_true",
                        help="Write every igba token in every region, for review, "
                             "instead of the confirmed five")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "_source.wav"
    if not source.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                        "-ac", "1", "-ar", str(SAMPLE_RATE), str(source)], check=True)
    samples = read_mono(source, SAMPLE_RATE)
    print(f"source: {len(samples) / SAMPLE_RATE:.1f}s")

    aligner = NeuralAligner()
    aligner._load()
    posteriors = frame_posteriors(aligner, samples)
    labels = posteriors.argmax(1)
    phones = aligner._phones
    probability = np.exp(posteriors[:, phones.index("gb")])
    peaks = bursts(probability)
    print(f"{len(peaks)} /gb/ bursts in {len(labels) * FRAME_S:.1f}s")

    pad = int(args.pad * SAMPLE_RATE)

    def cut(burst):
        span = word_span(labels, phones, burst)
        if span is None:
            return None, None
        first, last = span
        start = max(0, int(first * FRAME_S * SAMPLE_RATE) - pad)
        end = min(len(samples), int(last * FRAME_S * SAMPLE_RATE) + pad)
        return samples[start:end], (first * FRAME_S, last * FRAME_S)

    if args.candidates:
        out = ROOT / "data" / "igba_candidates"
        out.mkdir(parents=True, exist_ok=True)
        for stale in out.glob("*.wav"):
            stale.unlink()
        written = 0
        for tag, low, high in REGIONS:
            for burst in peaks:
                at = burst * FRAME_S
                if not low <= at <= high:
                    continue
                clip, span = cut(burst)
                if clip is None:
                    continue
                name = f"{tag}__{at:05.2f}s.wav"
                sf.write(str(out / name), clip, SAMPLE_RATE, subtype="PCM_16")
                print(f"  {name:<30}{len(clip) / SAMPLE_RATE:.2f}s")
                written += 1
        print(f"\n{written} candidates in {out}")
        return 0

    results = []
    for stem, word, at in OCCURRENCES:
        # The recorded time is the burst; find the peak the aligner actually
        # reports nearest to it rather than trusting the number to the frame.
        burst = min(peaks, key=lambda p: abs(p * FRAME_S - at))
        drift = abs(burst * FRAME_S - at)
        if drift > 0.10:
            print(f"  {stem}: no /gb/ burst within 100ms of {at:.2f}s "
                  f"(nearest is {burst * FRAME_S:.2f}s)", file=sys.stderr)
            continue
        clip, span = cut(burst)
        if clip is None:
            print(f"  {stem}: the burst at {at:.2f}s has no vowel either side",
                  file=sys.stderr)
            continue
        path = args.out / f"{stem}.wav"
        sf.write(str(path), clip, SAMPLE_RATE, subtype="PCM_16")
        print(f"  {stem:<30}{word:<7}{len(clip) / SAMPLE_RATE:.2f}s "
              f"from {span[0]:.2f}-{span[1]:.2f}s")
        results.append({"stem": stem, "word": word, "burst_s": burst * FRAME_S,
                        "start_s": span[0], "end_s": span[1],
                        "duration_s": len(clip) / SAMPLE_RATE})

    (args.out / "clip_metadata.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    source.unlink(missing_ok=True)
    print(f"\n{len(results)} clips in {args.out}")
    return 0 if len(results) == len(OCCURRENCES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
