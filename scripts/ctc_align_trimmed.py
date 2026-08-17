"""Does trimming silence fix CTC alignment's edge failure?

Pooled over all words, CTC disagreed with MFA by 70 ms median and 262 ms mean.
Broken down by position, almost all of that was structural:

    first word   700 ms median, 0% within 100 ms
    last word    855 ms median
    interior      50 ms median, 88% within 100 ms

MFA models silence explicitly with ``sil``/``spn`` phones and starts the first
word after it; CTC has only a blank symbol and no silence model, so it places the
first token near the start of the file. If that diagnosis is right, removing the
silence before aligning should fix the first and last words too.

The trim uses **energy-based voice activity detection**, deliberately not MFA's
own boundaries -- borrowing the reference's answer to help the candidate would
prove nothing, and a deployment would not have MFA available anyway.

    python scripts/ctc_align_trimmed.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import unicodedata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.g2p.orthography import normalize  # noqa: E402

FRAME_S = 0.02


def speech_bounds(samples, rate=16000, window_s=0.02, floor_db=-40.0, pad_s=0.05):
    """First and last sample of speech, by frame energy relative to the peak.

    Returns ``(start, end)`` sample indices, padded outward so a soft onset is
    not clipped -- cutting into the first consonant would move the very boundary
    being measured.
    """
    window = max(int(window_s * rate), 1)
    frames = len(samples) // window
    if frames < 3:
        return 0, len(samples)
    energy = np.abs(samples[: frames * window].reshape(frames, window)).mean(axis=1)
    peak = energy.max()
    if peak <= 0:
        return 0, len(samples)
    loud = np.flatnonzero(20 * np.log10(np.maximum(energy, 1e-10) / peak) > floor_db)
    if loud.size == 0:
        return 0, len(samples)
    pad = int(pad_s * rate)
    start = max(int(loud[0]) * window - pad, 0)
    end = min((int(loud[-1]) + 1) * window + pad, len(samples))
    return start, end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="models/lyngual-yoruba")
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(args.device)
    raw = json.loads((Path(args.model) / "vocab.json").read_text(encoding="utf-8"))
    vocab = raw.get("yor") or raw if isinstance(next(iter(raw.values())), dict) else raw
    delimiter = "|" if "|" in vocab else " "
    blank = vocab.get("<pad>", 0)

    grids = sorted(args.aligned.rglob("*.TextGrid"))[: args.limit]
    index = {p.stem: p for p in args.audio.rglob("*.wav")}

    # Same buckets as the untrimmed run, so the two are directly comparable.
    buckets: dict[str, list[float]] = {
        "first word": [], "interior": [], "last word": [], "ALL": []}
    trims: list[float] = []

    for grid in grids:
        wav = index.get(grid.stem)
        if wav is None:
            continue
        try:
            reference = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            count = int(len(samples) / rate * 16000)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")

        words = [w for w in reference.words
                 if normalize(w.label) and not w.label.startswith("[")]
        if not words:
            continue

        begin, finish = speech_bounds(samples)
        offset_s = begin / 16000
        trims.append(offset_s)
        trimmed = samples[begin:finish]
        if len(trimmed) < 1600:
            continue

        ids, owner = [], []
        for position, word in enumerate(words):
            if position:
                ids.append(vocab[delimiter]); owner.append(-1)
            for char in normalize(word.label):
                if char in vocab:
                    ids.append(vocab[char]); owner.append(position)
                else:
                    for piece in unicodedata.normalize("NFD", char):
                        if piece in vocab:
                            ids.append(vocab[piece]); owner.append(position)
        if not ids:
            continue

        with torch.inference_mode():
            logits = model(torch.from_numpy(trimmed)[None].to(args.device)).logits
            logprobs = torch.log_softmax(logits, dim=-1).cpu()
        try:
            tokens, scores = AF.forced_align(
                logprobs, torch.tensor([ids], dtype=torch.int32), blank=blank)
            spans = AF.merge_tokens(tokens[0], scores[0])
        except Exception:
            continue
        if len(spans) != len(ids):
            continue

        starts: dict[int, int] = {}
        for span, owner_index in zip(spans, owner):
            if owner_index >= 0:
                starts.setdefault(owner_index, span.start)

        for position, word in enumerate(words):
            if position not in starts:
                continue
            # Frames are relative to the trimmed audio; add the offset back.
            predicted = starts[position] * FRAME_S + offset_s
            delta = abs(predicted - word.start_s)
            buckets["ALL"].append(delta)
            if position == 0:
                buckets["first word"].append(delta)
            elif position == len(words) - 1:
                buckets["last word"].append(delta)
            else:
                buckets["interior"].append(delta)

    print(f"trimmed {len(trims)} utterances; leading silence removed: "
          f"median {statistics.median(trims)*1000:.0f} ms, "
          f"max {max(trims)*1000:.0f} ms\n")
    print(f"{'position':<14}{'n':>7}{'median':>9}{'mean':>9}{'<=50ms':>9}{'<=100ms':>9}"
          f"   untrimmed median")
    was = {"first word": 700, "interior": 50, "last word": 855, "ALL": 70}
    for name, deltas in buckets.items():
        if not deltas:
            continue
        print(f"{name:<14}{len(deltas):>7}{statistics.median(deltas)*1000:>8.0f}ms"
              f"{statistics.mean(deltas)*1000:>8.0f}ms"
              f"{sum(1 for d in deltas if d <= .05)/len(deltas):>9.0%}"
              f"{sum(1 for d in deltas if d <= .10)/len(deltas):>9.0%}"
              f"{was[name]:>16} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
