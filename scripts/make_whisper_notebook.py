"""Generate notebooks/finetune_whisper_kaggle.ipynb.

Written by a script rather than by hand because a .ipynb is JSON with source
split into lines, which is unpleasant to edit and easy to corrupt.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


CELLS = [
md("""# Fine-tuning Whisper's encoder for vowel identity

The engine currently runs two large models: a fine-tuned wav2vec2 for vowel
identity, and — if word identity is added — Whisper for transcription. This
notebook tests whether one model can do both.

Measured on identical rows, with the same classifier and speaker split, only the
encoder differing:

| encoder | accuracy | macro F1 |
|---|---|---|
| frozen wav2vec2 (upstream) | 0.906 | 0.879 |
| **Whisper, off the shelf** | 0.912 | **0.888** |
| **fine-tuned wav2vec2** | 0.923 | **0.904** |

Whisper's encoder already beats *untuned* wav2vec2 at telling `ẹ` from `e`,
which is the opposite of what a transcription objective was expected to do. What
it has not had is the fine-tuning that took wav2vec2 from 0.879 to 0.904.

**The number to beat is 0.904.** Clearing it means one 968 MB model replaces
1.8 GB of two, latency falls from 19.6 s to ~16.4 s because the encoder pass is
shared with transcription, and word identity becomes open-vocabulary — Whisper
names words the 9,092-entry lexicon does not contain, which is why a learner
saying `ọkun` is currently flagged but never named.

## Why the utterance pack, not the windowed one

Whisper pads every input to 30 seconds, so a forward pass costs the same whether
it carries 0.4 s of audio or 12 s:

| pack | real audio per pass | labelled vowels per pass |
|---|---|---|
| windowed | 1.3% | 1 |
| **utterance** | 10% | **~15** |

Fifteen times the training signal for the same compute. The windowed pack was a
reasonable trade for wav2vec2, which processes actual length; for Whisper it
simply wastes the pass. It also matches inference, which feeds whole
utterances."""),

code("""!pip -q install transformers==4.44.2 'huggingface_hub<1.0' safetensors

import torch
print("torch", torch.__version__)
count = torch.cuda.device_count()
for i in range(count):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i}  {p.name}  {p.total_memory/1e9:.1f} GB")
if count == 0:
    raise SystemExit("Enable a GPU accelerator (T4 x2 recommended).")"""),

md("""## 1. The pack

`build_utterance_pack.py` concatenates whole utterances into shards, with a
manifest giving each utterance's offset and every vowel span inside it."""),

code("""from pathlib import Path
import numpy as np, pandas as pd

PACK = Path("/kaggle/input/yoruba-utterance-pack")
if not (PACK / "utterances.tsv").exists():
    found = sorted(Path("/kaggle/input").glob("*/utterances.tsv"))
    if not found:
        raise SystemExit("Attach the utterance_pack dataset (Add Input).")
    PACK = found[0].parent
print("pack:", PACK)

utterances = pd.read_csv(PACK / "utterances.tsv", sep="\\t")
vowels = pd.read_csv(PACK / "vowels.tsv", sep="\\t")
print(f"{len(utterances):,} utterances, {len(vowels):,} vowels")
print(f"{len(vowels)/len(utterances):.1f} vowels per forward pass")
print(vowels.label.value_counts().to_string())"""),

code("""import hashlib

def held_out(speaker, percent=20):
    return int(hashlib.md5(str(speaker).encode()).hexdigest()[:8], 16) % 100 < percent

LABELS = sorted(vowels.label.unique())
INDEX = {name: i for i, name in enumerate(LABELS)}
spans_by_utterance = {u: g[["label", "start", "end"]].values.tolist()
                      for u, g in vowels.groupby("utterance")}
utterances = utterances[utterances.utterance.isin(spans_by_utterance)].reset_index(drop=True)
utterances["hold"] = utterances.speaker.map(held_out)

train_rows = utterances[~utterances.hold].reset_index(drop=True)
valid_rows = utterances[utterances.hold].reset_index(drop=True)
print(f"train {len(train_rows):,} utterances / {train_rows.speaker.nunique():,} speakers")
print(f"valid {len(valid_rows):,} utterances / {valid_rows.speaker.nunique():,} speakers")
held = vowels[vowels.utterance.isin(set(valid_rows.utterance))]
print("\\nheld-out vowel support:")
print(held.label.value_counts().to_string())"""),

md("""### Batching

No length bucketing here, unlike the wav2vec2 notebook. Whisper's feature
extractor pads everything to the same 3000 mel frames, so every item in a batch
is already the same size and sorting by duration would buy nothing."""),

code("""from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

ENCODER = "LyngualLabs/whisper-small-yoruba"
FRAME = 320          # 320 samples = 20 ms = one encoder frame
extractor = WhisperFeatureExtractor.from_pretrained(ENCODER)

class Utterances(Dataset):
    def __init__(self, rows, pack):
        self.rows = rows
        self.shards = {s: np.load(pack / f"audio_{s:03d}.npy", mmap_mode="r")
                       for s in sorted(rows.shard.unique())}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        audio = self.shards[r.shard][r.offset: r.offset + r.samples]
        audio = np.asarray(audio, dtype=np.float32) / 32768.0
        return audio, spans_by_utterance[r.utterance]

def collate(batch):
    mel = extractor([a for a, _ in batch], sampling_rate=16000,
                    return_tensors="pt").input_features
    items = []
    for i, (_, spans) in enumerate(batch):
        for label, start, end in spans:
            items.append((i, INDEX[label], int(start), int(end)))
    return mel, torch.tensor(items, dtype=torch.long)

BATCH = 16 * max(torch.cuda.device_count(), 1)
train_loader = DataLoader(Utterances(train_rows, PACK), batch_size=BATCH,
                          shuffle=True, collate_fn=collate, num_workers=2,
                          pin_memory=True, drop_last=True)
valid_loader = DataLoader(Utterances(valid_rows, PACK), batch_size=BATCH,
                          shuffle=False, collate_fn=collate, num_workers=2)
print(f"batch {BATCH} across {torch.cuda.device_count()} GPU(s); "
      f"{len(train_loader):,} train batches")"""),

md("""## 2. Model

Only the **encoder** is trained. Whisper's decoder generates text and takes no
part in vowel identity, so carrying it through training would cost memory for
nothing — it is reattached afterwards for transcription.

The convolutional front end is frozen, matching how the wav2vec2 runs treated
their feature extractor.

Pooling is a mean over each vowel's frames at one frame per 20 ms, identical to
`Wav2VecExtractor.measure_segment`, so the classifier trained downstream sees
the same shape of input it will see at inference.

Layer 12 is the last, so nothing is truncated. The sweep found Whisper rises
monotonically with depth — 0.851, 0.876, 0.888 at layers 6, 9 and 12 — unlike
wav2vec2, whose middle layer beat its last."""),

code("""from transformers import WhisperModel
import torch.nn as nn

class WhisperVowels(nn.Module):
    def __init__(self, path, classes):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(path).encoder
        # The convolutional front end stays fixed, as in the wav2vec2 runs.
        for module in (self.encoder.conv1, self.encoder.conv2):
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.head = nn.Linear(self.encoder.config.d_model, classes)

    def forward(self, mel):
        return self.encoder(mel).last_hidden_state

def pool(frames, items):
    out = []
    for row, _, start, end in items.tolist():
        a = start // FRAME
        b = min(max(end // FRAME, a + 1), frames.shape[1])
        out.append(frames[row, min(a, b - 1):b].mean(dim=0))
    return torch.stack(out)

model = WhisperVowels(ENCODER, len(LABELS)).cuda()
model.encoder.gradient_checkpointing_enable()
parallel = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{trainable/1e6:.0f}M trainable of "
      f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M")"""),

code("""from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight

y_train = vowels[vowels.utterance.isin(set(train_rows.utterance))].label.map(INDEX).values
weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).cuda())
print({LABELS[i]: round(float(w), 2) for i, w in enumerate(weights)})

optimiser = AdamW([
    {"params": model.encoder.parameters(), "lr": 1e-5},
    {"params": model.head.parameters(),    "lr": 1e-3},
], weight_decay=0.01)
scaler = torch.amp.GradScaler("cuda")"""),

md("""## 3. Train

Watch macro F1 against **0.904**, the fine-tuned wav2vec2 figure on the same
held-out design. Whisper starts at 0.888 untuned, so a gain of the size
fine-tuning gave wav2vec2 (+0.025) would land near 0.913.

Also watch `in` and `u`, the two genuinely weak classes: 0.779 and 0.843 for
Whisper untuned, against 0.834 and 0.856 for fine-tuned wav2vec2."""),

code("""from sklearn.metrics import f1_score, accuracy_score
import time

def evaluate():
    model.eval(); truth, predicted = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for mel, items in valid_loader:
            frames = parallel(mel.cuda(non_blocking=True))
            predicted += model.head(pool(frames, items)).argmax(dim=1).cpu().tolist()
            truth += items[:, 1].tolist()
    return (accuracy_score(truth, predicted),
            f1_score(truth, predicted, average="macro", zero_division=0),
            f1_score(truth, predicted, average=None,
                     labels=list(range(len(LABELS))), zero_division=0))

EPOCHS, best = 3, -1.0
for epoch in range(1, EPOCHS + 1):
    model.train(); started = time.time(); running = 0.0
    for step, (mel, items) in enumerate(train_loader, start=1):
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            frames = parallel(mel.cuda(non_blocking=True))
            loss = criterion(model.head(pool(frames, items)), items[:, 1].cuda())
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimiser); scaler.update()
        running += loss.item()
        if step % 200 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                  f"loss {running/step:.4f} {(time.time()-started)/60:.0f} min",
                  flush=True)

    accuracy, macro, per = evaluate()
    print(f"\\nepoch {epoch}: accuracy {accuracy:.4f}  macro F1 {macro:.4f}"
          f"   (target 0.904)")
    print("  " + "  ".join(f"{n}={per[i]:.3f}" for i, n in enumerate(LABELS)))
    if macro > best:
        best = macro
        model.encoder.save_pretrained("/kaggle/working/whisper_vowel_encoder")
        print(f"  saved (best macro F1 {best:.4f})")"""),

md("""## 4. Taking it home

The saved directory holds the **encoder only**. To transcribe as well, load the
full model and replace its encoder weights — the decoder is untouched by this
training, so it still works:

```python
full = WhisperForConditionalGeneration.from_pretrained("LyngualLabs/whisper-small-yoruba")
full.model.encoder.load_state_dict(tuned.state_dict())
full.save_pretrained("models/whisper-yoruba-tuned")
```

Then measure it locally on the same three slices the comparison used:

```bash
python scripts/extract_whisper_features.py --model models/whisper-yoruba-tuned \\
    --layers 12 --offset 0    --take 4000 --out models/whisper_tuned/slice_000000.npz
python scripts/extract_whisper_features.py --model models/whisper-yoruba-tuned \\
    --layers 12 --offset 4000 --take 4000 --out models/whisper_tuned/slice_004000.npz
python scripts/extract_whisper_features.py --model models/whisper-yoruba-tuned \\
    --layers 12 --offset 8000 --take 4000 --out models/whisper_tuned/slice_008000.npz
python scripts/train_vowel_v2.py models/whisper_tuned/*.npz --dry-run
```

Above 0.904 and one model replaces two. Below it, keep wav2vec2 for vowels and
use Whisper only for transcription, at +2.8 s per assessment on CPU.

One thing this cannot change either way: MFA is 13.2 s of the 16.8 s an
assessment takes, and Whisper gives no phone boundaries, so alignment stays."""),

code("""import shutil
shutil.make_archive("/kaggle/working/whisper_vowel_encoder", "zip",
                    "/kaggle/working/whisper_vowel_encoder")
print(f"best macro F1: {best:.4f}  (wav2vec2 fine-tuned: 0.904)")
print("download /kaggle/working/whisper_vowel_encoder.zip")"""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "notebooks" / "finetune_whisper_kaggle.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"wrote {target} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
