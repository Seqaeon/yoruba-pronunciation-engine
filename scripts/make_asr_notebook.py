"""Generate notebooks/finetune_whisper_asr_kaggle.ipynb.

Written by a script rather than by hand because a .ipynb is JSON with source
split into lines, which is unpleasant to edit and easy to corrupt.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(True)}


CELLS = [
md("""# Fine-tuning Whisper for Yoruba transcription

The engine transcribes with `LyngualLabs/whisper-small-yoruba` — someone else's
fine-tune of `whisper-small`, on someone else's data. This notebook fine-tunes
for transcription on **this** project's corpus: 193,309 utterances from 2,801
speakers, about 118 hours, already transcript-matched for acoustic-model
training.

## Read this before you start: the bar is higher than it looks

Measured on 60 native-speaker studio readings of 3–4 words, the shipped
checkpoint gets:

| decode | words right | phrases entirely right |
|---|---|---|
| greedy | 82% | 48% |
| beam 5 | 86% | 55% |
| **greedy, conditioned on the prompt** | **97%** | **88%** |

The engine now conditions on the prompt, because it can — a practice app always
knows what the learner was asked to say. That is already 97%, for free, with no
training.

**So a fine-tune has to beat 97%, not 82%, to be worth shipping.** Evaluate both
ways at the end; a model that reaches 90% unconditioned and 96% conditioned is a
better recogniser and a worse product.

Fine-tuning is still worth trying for two reasons the conditioning trick does
not cover: it would help wherever the prompt is unknown or wrong, and a stronger
unconditioned model reduces how much work the 3% prompt-override rate has to do.

## What is not worth repeating

Fine-tuning the encoder on vowel windows. That was already done, for vowel
identity, and grafting the result back onto the decoder changed transcription
not at all — 82% before, 82% after, with 1 of 60 transcripts differing. It
optimised a different objective, and the weights barely moved (max drift 0.011).
"""),

md("""## 1. Data

Built by `scripts/build_asr_pack.py` and uploaded as a Kaggle dataset:

```bash
python scripts/build_asr_pack.py --hours 40      # 3.0 GB, FLAC + manifest
```

The split is **already in the manifest, decided by hashing the speaker**. Do not
re-split here. Whisper memorises a voice quickly, so a random utterance split
would report a number that says nothing about a learner it has never heard.
"""),

code("""import os, sys, math, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PACK = Path("/kaggle/input/yoruba-asr-pack")          # adjust to your dataset slug
if not PACK.exists():
    PACK = Path("data/asr_pack")                      # running locally
AUDIO = PACK / "audio"

manifest = pd.read_csv(PACK / "manifest.tsv", sep="\\t")
print(manifest.groupby("split").agg(
    utterances=("utterance", "size"), hours=("seconds", lambda s: round(s.sum()/3600, 1))))
print()
print(manifest.groupby("source").size().sort_values(ascending=False))

# The split must be speaker-disjoint. Assert it rather than trust it: a leak here
# inflates every number in this notebook and nothing downstream would notice.
overlap = (set(manifest[manifest.split == "train"].speaker)
           & set(manifest[manifest.split == "test"].speaker))
assert not overlap, f"{len(overlap)} speakers appear in both train and test"
print(f"\\nno speaker overlap between train and test")"""),

md("""## 2. Base checkpoint

`openai/whisper-small` or the Yoruba fine-tune already in use. Starting from the
Yoruba one usually converges faster; starting from the base gives a cleaner
answer about what this corpus alone is worth. Try the Yoruba one first."""),

code("""from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

BASE = "LyngualLabs/whisper-small-yoruba"
LANGUAGE, TASK = "yoruba", "transcribe"

processor = WhisperProcessor.from_pretrained(BASE, language=LANGUAGE, task=TASK)
model = WhisperForConditionalGeneration.from_pretrained(BASE)

# Whisper's generation config carries forced tokens from pre-training that
# override anything set at generate() time and silently produce the wrong
# language. Clear them and set the pair explicitly.
model.generation_config.language = LANGUAGE
model.generation_config.task = TASK
model.generation_config.forced_decoder_ids = None
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

total = sum(p.numel() for p in model.parameters())
print(f"{BASE}: {total/1e6:.0f}M parameters, "
      f"{model.config.encoder_layers}L encoder / {model.config.decoder_layers}L decoder")"""),

md("""## 3. Dataset

Mel spectrogram in, token ids out. Whisper pads every input to 30 seconds, so a
0.5 s utterance costs exactly what a 20 s one does — which is why the pack caps
length rather than letting long recordings in for free.

Tone marks are kept. They are part of Yoruba spelling, the corpus carries them,
and the engine strips them itself when it compares words. Training on stripped
text would throw away information the tokenizer can represent."""),

code("""import soundfile as sf
from torch.utils.data import Dataset

MAX_LABEL_TOKENS = 200

class Utterances(Dataset):
    def __init__(self, frame, processor):
        self.rows = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        samples, rate = sf.read(str(AUDIO / f"{row.utterance}.flac"), dtype="float32")
        assert rate == 16000, f"{row.utterance} is {rate} Hz; the pack should be 16 kHz"
        features = self.processor.feature_extractor(
            samples, sampling_rate=16000).input_features[0]
        labels = self.processor.tokenizer(str(row.text)).input_ids[:MAX_LABEL_TOKENS]
        return {"input_features": features, "labels": labels}


def collate(batch):
    features = torch.tensor(np.stack([b["input_features"] for b in batch]))
    longest = max(len(b["labels"]) for b in batch)
    pad = processor.tokenizer.pad_token_id
    labels = torch.full((len(batch), longest), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
    # -100 is ignored by the loss; the decoder start token is added by the model.
    return {"input_features": features, "labels": labels}


train_set = Utterances(manifest[manifest.split == "train"], processor)
dev_set = Utterances(manifest[manifest.split == "dev"].sample(
    n=min(600, (manifest.split == "dev").sum()), random_state=0), processor)
print(f"{len(train_set)} training, {len(dev_set)} dev (subsampled, for speed)")"""),

md("""## 4. Metrics

WER is the standard number and it is not the number this project cares about.
The engine compares words with tone marks and punctuation stripped, because a
recogniser's tone marks drift and a tonal error belongs in the tone report
rather than being counted as a wrong word. Both are reported."""),

code("""import unicodedata, re, evaluate

wer_metric = evaluate.load("wer")
_PUNCT = re.compile(r"[^\\w\\s\\u0300-\\u036f]", re.UNICODE)

def strip_tones(text):
    decomposed = unicodedata.normalize("NFD", text)
    # Keep the dot below (U+0323): it changes which vowel a letter is.
    kept = "".join(c for c in decomposed
                   if not (0x0300 <= ord(c) <= 0x0304 or ord(c) == 0x0309))
    return unicodedata.normalize("NFC", kept)

def engine_key(text):
    return strip_tones(_PUNCT.sub("", str(text))).lower().strip()

def compute_metrics(prediction):
    ids = prediction.predictions
    labels = prediction.label_ids.copy()
    labels[labels == -100] = processor.tokenizer.pad_token_id
    predicted = processor.batch_decode(ids, skip_special_tokens=True)
    truth = processor.batch_decode(labels, skip_special_tokens=True)

    raw = 100 * wer_metric.compute(predictions=predicted, references=truth)
    folded = 100 * wer_metric.compute(
        predictions=[engine_key(t) for t in predicted],
        references=[engine_key(t) for t in truth])
    exact = 100 * np.mean([engine_key(p) == engine_key(t)
                           for p, t in zip(predicted, truth)])
    return {"wer": raw, "wer_folded": folded, "phrase_exact": exact}"""),

md("""## 5. Train

Batch size and precision are set for a Kaggle T4. `fp16` on, gradient
checkpointing on, and a small per-device batch with accumulation — whisper-small
plus 30-second mels fills 16 GB faster than the parameter count suggests."""),

code("""OUT = Path("/kaggle/working/whisper-yoruba-asr")

args = Seq2SeqTrainingArguments(
    output_dir=str(OUT),
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=2,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer_folded",
    greater_is_better=False,
    report_to=[],
    dataloader_num_workers=2,
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=train_set, eval_dataset=dev_set,
    data_collator=collate, compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
trainer.train()"""),

md("""## 6. The evaluation that decides whether to ship it

Held-out **speakers**, and scored both ways. The conditioned column is what the
engine actually runs, so it is the one that decides.

A fine-tune that improves the unconditioned column and not the conditioned one
is a better recogniser that changes nothing for a learner."""),

code("""from torch.utils.data import DataLoader

test = manifest[manifest.split == "test"].sample(
    n=min(300, (manifest.split == "test").sum()), random_state=0)
model.eval().cuda() if torch.cuda.is_available() else model.eval()
device = next(model.parameters()).device

def transcribe(row, condition_on_prompt):
    samples, _ = sf.read(str(AUDIO / f"{row.utterance}.flac"), dtype="float32")
    feats = processor.feature_extractor(
        samples, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    kwargs = {"max_new_tokens": 128}
    if condition_on_prompt:
        kwargs["prompt_ids"] = processor.get_prompt_ids(
            str(row.text), return_tensors="pt").to(device)
    with torch.inference_mode():
        ids = model.generate(feats, **kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if condition_on_prompt:
        head = text.lstrip()
        if head[:len(str(row.text))].lower() == str(row.text).lower():
            text = head[len(str(row.text)):]
    return text.strip()

for condition in (False, True):
    hypotheses, references = [], []
    for _, row in test.iterrows():
        hypotheses.append(transcribe(row, condition))
        references.append(str(row.text))
    folded = 100 * wer_metric.compute(
        predictions=[engine_key(h) for h in hypotheses],
        references=[engine_key(r) for r in references])
    exact = 100 * np.mean([engine_key(h) == engine_key(r)
                           for h, r in zip(hypotheses, references)])
    label = "conditioned on the prompt" if condition else "no context"
    print(f"  {label:<28}WER {folded:5.1f}%   phrases exactly right {exact:5.1f}%")

print("\\nShipped checkpoint, on the engine's own 60-reading set, for reference:")
print("  no context                  82% of words right, 48% of phrases")
print("  conditioned on the prompt   97% of words right, 88% of phrases")"""),

md("""## 7. Export

Save the full model — encoder *and* decoder, plus the tokenizer and feature
extractor. The engine loads `WhisperForConditionalGeneration` from a directory
and reads `config.json` to decide what it is holding; an encoder-only export
would load but transcribe nothing.

Then add it to `models/MANIFEST.json` as an optional artifact and upload the
tarball to the release, so `scripts/fetch_models.py --include-optional` can find
it."""),

code("""model.save_pretrained(OUT)
processor.save_pretrained(OUT)

import subprocess, hashlib
archive = "/kaggle/working/whisper-yoruba-asr.tar.gz"
subprocess.run(["tar", "-czf", archive, "-C", str(OUT.parent), OUT.name], check=True)

digest = hashlib.sha256()
with open(archive, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
print(f"{archive}\\n  bytes  {os.path.getsize(archive)}\\n  sha256 {digest.hexdigest()}")
print("\\nPut those two values in models/MANIFEST.json under optional_artifacts.")"""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "finetune_whisper_asr_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
