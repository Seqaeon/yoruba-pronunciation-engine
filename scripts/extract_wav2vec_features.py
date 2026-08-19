"""Extract wav2vec2 frame embeddings over aligned vowel spans.

This is the same job ``extract_vowel_features.py`` does, with the 39
hand-designed measurements (formants, bandwidths, cepstral means) replaced by a
learned 1024-dimensional representation. Everything downstream -- the cache
format, the speaker-disjoint split, the classifier -- is unchanged, so the two
are directly comparable.

**Layer choice is not obvious and is therefore swept.** The final layer of a CTC
model is optimised to predict characters, and a transcription objective rewards
discarding detail the transcript does not need -- which is exactly the detail
that separates ``an`` from ``un``. Middle layers usually retain more phonetic
information. ``--layers`` extracts several in one forward pass, which costs
nothing extra, so the best one can be chosen by measurement.

    # sweep layers on a subset to pick one
    python scripts/extract_wav2vec_features.py --limit 1200 \\
        --layers 4 8 12 16 20 24 --out models/w2v_sweep.npz

    # then the full corpus at the winner
    python scripts/extract_wav2vec_features.py --layers 12 \\
        --out models/w2v_layer12.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

ARTEFACT = {"en": "e", "on": "o"}
#: wav2vec2 emits one frame per 20 ms at 16 kHz.
FRAME_S = 0.02


def spans(alignment):
    """Vowel intervals worth measuring, in time order."""
    return [p for p in alignment.phones if is_vowel_phone(p.label)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="models/lyngual-yoruba")
    parser.add_argument("--corpora", nargs="+", default=[
        "output/aligned:data/mfa_corpus", "output/studio_aligned:data/studio_corpus"])
    parser.add_argument("--layers", type=int, nargs="+", default=[12])
    parser.add_argument("--limit", type=int, help="Utterances per corpus, for a sweep")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many utterances before extracting. With "
                             "--take, splits one long run into slices that "
                             "checkpoint independently -- a GPU that hangs then "
                             "costs one slice rather than the whole job.")
    parser.add_argument("--take", type=int, default=0,
                        help="Extract at most this many utterances after --offset.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    import torch
    import soundfile as sf
    from transformers import Wav2Vec2ForCTC

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(device)
    print(f"{args.model} on {device}; layers {args.layers}")

    pairs = []
    for spec in args.corpora:
        aligned, audio = spec.split(":")
        grids = sorted(Path(aligned).rglob("*.TextGrid"))[: args.limit or None]
        index = {p.stem: p for p in Path(audio).rglob("*.wav")}
        pairs += [(g, index[g.stem]) for g in grids if g.stem in index]
    if args.offset or args.take:
        end = args.offset + args.take if args.take else None
        pairs = pairs[args.offset:end]
    print(f"{len(pairs):,} utterances with audio")

    out = {layer: [] for layer in args.layers}
    labels, speakers = [], []
    started = time.time()
    skipped = 0

    for number, (grid, wav) in enumerate(pairs, start=1):
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            skipped += 1
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:                      # the model is 16 kHz only
            count = int(len(samples) / rate * 16000)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")
        vowels = spans(alignment)
        if not vowels or len(samples) < 1600:
            continue

        with torch.inference_mode():
            result = model(torch.from_numpy(samples)[None].to(device),
                           output_hidden_states=True)
        frames = result.hidden_states           # tuple, index 0 is the CNN output
        n = frames[0].shape[1]

        for phone in vowels:
            start = int(phone.start_s / FRAME_S)
            end = max(int(phone.end_s / FRAME_S), start + 1)
            if start >= n:
                continue
            end = min(end, n)
            for layer in args.layers:
                # Mean over the vowel's frames, matching how the MFCC baseline
                # summarises a span, so the comparison isolates the features.
                out[layer].append(
                    frames[layer][0, start:end].mean(dim=0).float().cpu().numpy())
            base = phone.label.split("_")[0]
            labels.append(ARTEFACT.get(base, base))
            speakers.append(wav.parent.name)

        if number % 250 == 0:
            rate_per_s = number / max(time.time() - started, 1e-9)
            print(f"  {number:,}/{len(pairs):,}  {len(labels):,} vowels  "
                  f"{rate_per_s:.1f} utt/s  ~{(len(pairs)-number)/rate_per_s/60:.0f} min left",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{layer}": np.asarray(rows, dtype="float16")
               for layer, rows in out.items()}
    payload["labels"] = np.asarray(labels)
    payload["speakers"] = np.asarray(speakers)
    np.savez_compressed(args.out, **payload)

    from collections import Counter
    print(f"\n{len(labels):,} vowels from {len(pairs) - skipped:,} utterances, "
          f"{len(set(speakers)):,} speakers, {time.time()-started:.0f}s")
    print("class balance:", dict(sorted(Counter(labels).items())))
    print(f"wrote {args.out} ({args.out.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
