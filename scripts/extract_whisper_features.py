"""Extract Whisper encoder embeddings over aligned vowel spans.

The same job ``extract_wav2vec_features.py`` does, from a different encoder, so
the two can be compared by swapping only the features and holding the classifier
and the speaker split fixed.

The question is whether one model can do both jobs. Whisper transcribes far
better than forced alignment can -- it is open-vocabulary, so it names words the
9,092-entry lexicon does not contain -- and if its encoder also carries vowel
identity, the engine could drop wav2vec2 and run a single 968 MB model instead
of two.

There is a reason to doubt it. Whisper is trained to transcribe, and a
transcriber can afford to blur ``ẹ`` against ``e`` wherever context
disambiguates. That is the same objection that made hand-designed features lose
to wav2vec2's self-supervised ones by 0.379 macro F1. Doubt is not evidence
either way, hence this script.

**Two mechanical differences from wav2vec2**, both handled here:

Whisper pads every input to 30 seconds, so the encoder always returns 1500
frames whatever the clip length. The padding is harmless -- vowel spans index
into the real region -- but it means a two-second clip costs the same forward
pass as a thirty-second one.

Its frame rate is nevertheless the same 50 per second, one frame per 20 ms, so
the span-to-frame arithmetic is identical to the wav2vec2 path and pooling stays
comparable.

    python scripts/extract_whisper_features.py --layers 6 9 12 \\
        --offset 0 --take 4000 --out models/whisper_slices/slice_000000.npz
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

from yoruba_engine.audio import resample  # noqa: E402
from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

ARTEFACT = {"en": "e", "on": "o"}
#: Whisper emits one frame per 20 ms, the same as wav2vec2.
FRAME_S = 0.02
RATE = 16000


def spans(alignment):
    return [p for p in alignment.phones if is_vowel_phone(p.label)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="models/whisper-yoruba")
    parser.add_argument("--corpora", nargs="+",
                        default=["output/aligned_v3:data/train_corpus"])
    parser.add_argument("--layers", type=int, nargs="+", default=[12],
                        help="encoder layers to pool; whisper-small has 12")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--take", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extractor",
                        help="where to read the feature extractor from. An "
                             "encoder-only checkpoint does not carry one.")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    import soundfile as sf
    import torch
    import json as _json
    from transformers import (
        WhisperFeatureExtractor, WhisperModel,
    )
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # Fine-tuning here saves the encoder alone, since the decoder takes no part
    # in vowel identity; the upstream checkpoint is a full WhisperModel. Reading
    # the config tells the two apart, rather than guessing and getting either a
    # missing attribute or silently randomised weights.
    config_path = Path(args.model) / "config.json"
    architectures = []
    if config_path.exists():
        architectures = _json.loads(
            config_path.read_text(encoding="utf-8")).get("architectures") or []
    if "WhisperEncoder" in architectures:
        model = WhisperEncoder.from_pretrained(args.model).eval().to(device)
        extractor_source = args.extractor or "models/whisper-yoruba"
    else:
        model = WhisperModel.from_pretrained(args.model).encoder.eval().to(device)
        extractor_source = args.extractor or args.model
    features = WhisperFeatureExtractor.from_pretrained(extractor_source)
    print(f"{args.model} encoder on {device}; layers {args.layers} "
          f"of {model.config.encoder_layers}")

    pairs = []
    for spec in args.corpora:
        aligned, audio = spec.split(":")
        grids = sorted(Path(aligned).rglob("*.TextGrid"))
        index = {p.stem: p for p in Path(audio).rglob("*.wav")}
        pairs += [(g, index[g.stem]) for g in grids if g.stem in index]
    if args.offset or args.take:
        end = args.offset + args.take if args.take else None
        pairs = pairs[args.offset:end]
    print(f"{len(pairs):,} utterances with audio")

    out = {layer: [] for layer in args.layers}
    labels, speakers = [], []
    started, skipped = time.time(), 0

    for number, (grid, wav) in enumerate(pairs, start=1):
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            skipped += 1
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != RATE:
            samples = resample(samples, rate, RATE)
        vowels = spans(alignment)
        if not vowels or len(samples) < 1600:
            continue

        mel = features(samples, sampling_rate=RATE, return_tensors="pt")
        with torch.inference_mode():
            result = model(mel.input_features.to(device), output_hidden_states=True)
        frames = result.hidden_states          # index 0 is the convolutional output
        length = frames[0].shape[1]

        for phone in vowels:
            start = int(phone.start_s / FRAME_S)
            end = max(int(phone.end_s / FRAME_S), start + 1)
            if start >= length:
                continue
            end = min(end, length)
            for layer in args.layers:
                out[layer].append(
                    frames[layer][0, start:end].mean(dim=0).float().cpu().numpy())
            base = phone.label.split("_")[0]
            labels.append(ARTEFACT.get(base, base))
            speakers.append(wav.parent.name)

        if number % 250 == 0:
            rate_per_s = number / max(time.time() - started, 1e-9)
            print(f"  {number:,}/{len(pairs):,}  {len(labels):,} vowels  "
                  f"{rate_per_s:.1f} utt/s  "
                  f"~{(len(pairs) - number) / rate_per_s / 60:.0f} min left",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"layer_{layer}": np.asarray(rows, dtype="float16")
               for layer, rows in out.items()}
    payload["labels"] = np.asarray(labels)
    payload["speakers"] = np.asarray(speakers)
    np.savez_compressed(args.out, **payload)

    from collections import Counter
    print(f"\n{len(labels):,} vowels from {len(pairs) - skipped:,} utterances, "
          f"{len(set(speakers)):,} speakers, {time.time() - started:.0f}s")
    print("class balance:", dict(sorted(Counter(labels).items())))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
