"""Turn the corpus alignments into per-frame phone labels for a neural aligner.

MFA is 96% of an assessment's 14.3 s, and almost none of that is alignment -- it
is corpus setup, a database, and feature extraction, charged per invocation. No
combination of its flags moves it: ``--single_speaker``, ``--num_jobs 1`` and
``--no_use_mp`` all land within a second of 13.8 s and leave boundaries
bit-identical.

The replacement is an acoustic model plus Viterbi, which is what MFA already is,
with the GMM swapped for the encoder this project already trains. What that
needs is frame-level supervision, and it already exists.

**Why frame labels rather than CTC.** CTC never sees timings; it marginalises
over every alignment producing the right sequence, and the known result is peaky
posteriors -- a spike somewhere inside each phone and blank elsewhere. Nothing
in that objective cares where a phone begins. Measured against MMS_FA, a trained
CTC aligner, vowel edges scatter 36 ms around MFA's with a +5 ms systematic
offset, so there is nothing to correct away. Frame-level cross-entropy asks the
question actually being asked: what is sounding at time t.

**MFA is the ceiling here, not the floor.** Nobody hand-labelled these
boundaries. Distilling MFA keeps every threshold in this project valid at a
thirtieth of the runtime; it cannot align better than what it imitates.

Labels come from ``output/train_aligned`` -- 192,847 TextGrids covering 99.85%
of the corpus, with both a words and a phones tier. Every phone is supervised,
not only vowels, so the result can replace what MFA does rather than only its
vowel half: word boundaries feed the omission check and per-phone posteriors
give GOP a direct analogue.

Phones are the acoustic model's own bare set, with no ``_H/_M/_L`` suffixes,
because it was trained tone-stripped. Tone is measured from pitch, so nothing is
lost by that.

Audio is never copied. The manifest points at the corpus, which is already on
disk and already 16 kHz mono.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: wav2vec2 emits one frame per 20 ms. MFA works at 10 ms, so quantisation alone
#: costs about 5 ms of mean boundary error -- the floor this design works against.
FRAME_S = 0.02

#: Index 0. MFA writes silence as an empty label, ``sil`` or ``sp``; all three
#: mean the same thing to a frame classifier and are collapsed here.
SILENCE = "sil"
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>"}


def frame_labels(intervals, duration_s: float, index_of: dict) -> np.ndarray:
    """One label per 20 ms frame, from a TextGrid's phone tier.

    A frame takes the phone covering its midpoint, which is the rule the pitch
    extractor already uses to assign phones to words, so the two cannot disagree
    about which frames belong to which phone.
    """
    frames = max(int(duration_s / FRAME_S), 1)
    labels = np.zeros(frames, dtype=np.int16)
    for interval in intervals:
        label = interval.label.strip()
        target = index_of[SILENCE if label in SILENCE_LABELS else label]
        first = max(int(interval.start_s / FRAME_S), 0)
        last = min(int(np.ceil(interval.end_s / FRAME_S)) + 1, frames)
        for frame in range(first, last):
            midpoint = (frame + 0.5) * FRAME_S
            if interval.start_s <= midpoint < interval.end_s:
                labels[frame] = target
    return labels


def main() -> int:
    import soundfile as sf  # noqa: PLC0415

    from yoruba_engine.alignment.textgrid import parse_textgrid  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=ROOT / "output" / "train_aligned",
                        help="Directory of MFA TextGrids")
    parser.add_argument("--audio", type=Path, default=ROOT / "data" / "train_corpus",
                        help="Corpus the TextGrids describe")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "aligner_labels")
    parser.add_argument("--limit", type=int, help="only this many utterances, for a trial")
    args = parser.parse_args()

    grids = sorted(args.aligned.rglob("*.TextGrid"))
    if args.limit:
        grids = grids[:args.limit]
    print(f"{len(grids)} TextGrids", flush=True)

    # One pass to learn the phone inventory, so class indices are stable across
    # runs and the count is reported rather than assumed.
    seen: Counter = Counter()
    for grid in grids[:min(len(grids), 5000)]:
        for interval in parse_textgrid(grid).get("phones", []):
            label = interval.label.strip()
            seen[SILENCE if label in SILENCE_LABELS else label] += 1
    vocabulary = [SILENCE] + sorted(k for k in seen if k != SILENCE)
    index_of = {label: position for position, label in enumerate(vocabulary)}
    print(f"{len(vocabulary)} classes: {' '.join(vocabulary)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    store: dict[str, np.ndarray] = {}
    rows: list[tuple] = []
    frame_total = silent = skipped = 0

    for position, grid in enumerate(grids, 1):
        wav = args.audio / grid.parent.name / f"{grid.stem}.wav"
        if not wav.exists():
            skipped += 1
            continue
        try:
            info = sf.info(str(wav))
            phones = [i for i in parse_textgrid(grid).get("phones", [])
                      if (i.label.strip() in SILENCE_LABELS
                          or i.label.strip() in index_of)]
        except Exception:
            skipped += 1
            continue
        if not phones:
            skipped += 1
            continue

        labels = frame_labels(phones, info.frames / info.samplerate, index_of)
        if labels.size < 5:
            skipped += 1
            continue

        store[grid.stem] = labels
        rows.append((grid.stem, str(wav.relative_to(ROOT)), grid.parent.name, labels.size))
        frame_total += labels.size
        silent += int((labels == 0).sum())
        if position % 20000 == 0:
            print(f"  {position}/{len(grids)}", flush=True)

    with (args.out / "frames.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["utterance", "audio", "speaker", "frames"])
        writer.writerows(rows)
    (args.out / "phones.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    np.savez(args.out / "labels.npz", **store)

    print(f"\n{len(rows)} utterances, {skipped} skipped")
    print(f"{frame_total} frames = {frame_total * FRAME_S / 3600:.1f} h")
    print(f"{silent / max(frame_total, 1):.1%} of frames are silence")
    print(f"\nWrote {args.out}/frames.tsv, phones.txt, labels.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
