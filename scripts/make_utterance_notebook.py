"""Generate notebooks/finetune_utterance_kaggle.ipynb.

The notebook is written by a script rather than by hand because a .ipynb is
JSON with source split into lines, which is unpleasant to edit directly and easy
to corrupt.
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
md("""# Fine-tuning on whole utterances

The companion notebook trains on 0.4 s windows cut around each vowel, so classes
could be balanced by resampling -- take all 4,504 `en`, cap `i` at 40,000 --
which whole utterances cannot do.

It worked. Measured on identical rows with the engine's own classifier, the
windowed fine-tune lifted macro F1 from **0.879 to 0.904** and improved all
twelve classes.

But wav2vec2 attends across its entire input, so a vowel inside a 400 ms window
is not represented the way the same vowel is inside a three-second utterance --
and the engine feeds whole utterances. This notebook removes that mismatch and
measures whether it was costing anything.

**Two things the first notebook wasted**, fixed here:

- it used **one** GPU of a T4 x2, leaving 16 GB idle
- it used **batch 24** while occupying 4 GB of 32 GB

Balancing moves out of the sampling and into the loss, weighted per vowel."""),

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
manifest recording each utterance's offset and every vowel span inside it.
Variable-length audio does not fit a rectangular array, and one file per
utterance would mean 58,000 files."""),

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
print(f"duration: median {utterances.samples.median()/16000:.1f}s, "
      f"max {utterances.samples.max()/16000:.1f}s")
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

md("""### Batching by duration

Padding a three-second utterance out to twelve spends most of the batch on
silence. Sorting by length and forming batches from neighbours keeps padding
small; the batches are then shuffled, so length does not correlate with
training order."""),

code("""import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

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
        return np.asarray(audio, dtype=np.float32) / 32768.0, \\
            spans_by_utterance[r.utterance]

def collate(batch):
    longest = max(len(a) for a, _ in batch)
    audio = torch.zeros(len(batch), longest)
    mask = torch.zeros(len(batch), longest, dtype=torch.long)
    items = []
    for i, (a, spans) in enumerate(batch):
        audio[i, :len(a)] = torch.from_numpy(a)
        mask[i, :len(a)] = 1
        for label, start, end in spans:
            items.append((i, INDEX[label], int(start), int(end)))
    return audio, mask, torch.tensor(items, dtype=torch.long)

class Bucketed(Sampler):
    def __init__(self, rows, batch):
        order = rows.samples.sort_values().index.tolist()
        self.groups = [order[i:i + batch] for i in range(0, len(order), batch)]
        random.Random(0).shuffle(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __len__(self):
        return len(self.groups)

BATCH = 16 * max(torch.cuda.device_count(), 1)
train_loader = DataLoader(Utterances(train_rows, PACK),
                          batch_sampler=Bucketed(train_rows, BATCH),
                          collate_fn=collate, num_workers=2, pin_memory=True)
valid_loader = DataLoader(Utterances(valid_rows, PACK),
                          batch_sampler=Bucketed(valid_rows, BATCH),
                          collate_fn=collate, num_workers=2)
print(f"batch {BATCH} across {torch.cuda.device_count()} GPU(s); "
      f"{len(train_loader):,} train batches")"""),

md("""## 2. Model

Same shape as the windowed notebook: encoder truncated to 16 layers (the layer
the engine reads), CNN front end frozen, mean-pool over each vowel's own frames.

What differs is that pooling now happens at many positions inside one long
utterance rather than once in the middle of a short window -- which is the case
inference actually presents.

`DataParallel` puts both T4s to work. Pooling stays outside the module so the
frames are gathered back before it runs."""),

code("""from transformers import Wav2Vec2Model
import torch.nn as nn

ENCODER = "LyngualLabs/mms-300m-yoruba-english"
LAYER, FRAME = 16, 320          # 320 samples = 20 ms at 16 kHz

class UtteranceEncoder(nn.Module):
    def __init__(self, path, classes):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(path)
        self.encoder.encoder.layers = self.encoder.encoder.layers[:LAYER]
        self.encoder.freeze_feature_encoder()
        self.head = nn.Linear(self.encoder.config.hidden_size, classes)

    def forward(self, audio, mask):
        return self.encoder(audio, attention_mask=mask).last_hidden_state

def pool(frames, items):
    \"\"\"Mean over each vowel's frames, matching Wav2VecExtractor exactly.\"\"\"
    out = []
    for row, _, start, end in items.tolist():
        a = start // FRAME
        b = min(max(end // FRAME, a + 1), frames.shape[1])
        out.append(frames[row, min(a, b - 1):b].mean(dim=0))
    return torch.stack(out)

model = UtteranceEncoder(ENCODER, len(LABELS)).cuda()
model.encoder.gradient_checkpointing_enable()
parallel = nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
print(f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M parameters")"""),

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

Compare on macro F1 and the per-class line. The number to beat is **0.904** --
the windowed encoder's score once its features were measured properly on whole
utterances with the engine's own classifier, not that notebook's own 0.9186,
which came from a balanced pack and a friendlier held-out split."""),

code("""from sklearn.metrics import f1_score, accuracy_score
import time

def evaluate():
    model.eval(); truth, predicted = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for audio, mask, items in valid_loader:
            frames = parallel(audio.cuda(non_blocking=True),
                              mask.cuda(non_blocking=True))
            predicted += model.head(pool(frames, items)).argmax(dim=1).cpu().tolist()
            truth += items[:, 1].tolist()
    return (accuracy_score(truth, predicted),
            f1_score(truth, predicted, average="macro", zero_division=0),
            f1_score(truth, predicted, average=None,
                     labels=list(range(len(LABELS))), zero_division=0))

EPOCHS, best = 3, -1.0
for epoch in range(1, EPOCHS + 1):
    model.train(); started = time.time(); running = 0.0
    for step, (audio, mask, items) in enumerate(train_loader, start=1):
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            frames = parallel(audio.cuda(non_blocking=True),
                              mask.cuda(non_blocking=True))
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
    print(f"\\nepoch {epoch}: accuracy {accuracy:.4f}  macro F1 {macro:.4f}")
    print("  " + "  ".join(f"{n}={per[i]:.3f}" for i, n in enumerate(LABELS)))
    if macro > best:
        best = macro
        # Truncating by mutating .layers leaves the config claiming 24 layers
        # while the checkpoint holds 16. from_pretrained then initialises the
        # missing 8 at random, warning but not failing -- a fine-tuned model
        # that silently loses half its depth. Correct it before saving.
        model.encoder.config.num_hidden_layers = len(model.encoder.encoder.layers)
        model.encoder.save_pretrained("/kaggle/working/finetuned_utterance")
        print(f"  saved (best macro F1 {best:.4f})")"""),

md("""## 4. Taking it home

Three slices decide it -- the windowed encoder scored 0.904 on exactly that set,
against the frozen encoder's 0.879:

```bash
MODEL=models/finetuned_utterance OUT=models/w2v_slices_utt \\
    SLICES="0 4000 8000" bash scripts/extract_finetuned_subset.sh
python scripts/train_vowel_v2.py models/w2v_slices_utt/*.npz --dry-run
```

Whichever wins gets the full fifteen-slice extraction and ships.

If they tie, prefer the windowed pack: it balances classes by construction,
which matters more as `en` and `in` grow rarer relative to everything else."""),

code("""import shutil
shutil.make_archive("/kaggle/working/finetuned_utterance", "zip",
                    "/kaggle/working/finetuned_utterance")
print(f"best macro F1: {best:.4f}")
print("download /kaggle/working/finetuned_utterance.zip")"""),
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
    target = ROOT / "notebooks" / "finetune_utterance_kaggle.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
    print(f"wrote {target} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
