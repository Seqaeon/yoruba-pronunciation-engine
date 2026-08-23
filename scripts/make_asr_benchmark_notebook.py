"""Generate notebooks/benchmark_asr_kaggle.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(text): return {"cell_type": "markdown", "metadata": {},
                      "source": text.splitlines(True)}
def code(text): return {"cell_type": "code", "execution_count": None,
                        "metadata": {}, "outputs": [], "source": text.splitlines(True)}


CELLS = [
md("""# Is a bigger Yoruba Whisper worth it?

The engine transcribes with `LyngualLabs/whisper-small-yoruba` — 244M parameters,
the smallest Whisper — and conditions the decoder on the prompt, because a
practice app always knows what the learner was asked to say.

**That combination is the number to beat, and it is high.** On the 60 readings in
this dataset:

| decode | words right | phrases entirely right | sub-dots kept |
|---|---|---|---|
| greedy, no context | 82% | 48% | 97% |
| beam 5, no context | 86% | 55% | — |
| **greedy + prompt context** | **97%** | **88%** | **98%** |
| NCAIR1/Yoruba-ASR, no context | 91% | 68% | 94% |

A larger model has to clear **97% / 88% / 98%**, not 82%. And most large Yoruba
fine-tunes cannot use prompt context at all — NCAIR1 parrots it back, returning
the prompt verbatim and nothing else about half the time — so they give up more
than size gains.

## What this measures, and why it is not the full corpus

Three things, on the same audio every figure above came from:

1. **Words right**, compared with punctuation and tone marks folded away, after
   the transcript is re-spaced onto the prompt's word boundaries. Yoruba is
   written with spaces the speech does not have, so a recogniser that splits
   `ọmọ` differently is not wrong about the sounds.
2. **Sub-dots kept** — `ọ` is /ɔ/ and `o` is /o/, a distinction learners fail on
   and recognisers drop. This is the single most useful figure for this app.
3. **Seconds per utterance**, which decides whether a model can be deployed at
   all. The current one takes about 1.0 s on 16 CPU cores; a 1.55B model scales
   to roughly 6.6 s, and a 4-vCPU droplet is several times worse again.

96 readings is enough to separate 97% from 90% and not enough to separate 97%
from 96%. If two models come out within a point, they are tied as far as this
can tell."""),

code("""import os, re, csv, time, unicodedata, difflib
from pathlib import Path

import numpy as np, soundfile as sf, torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

PACK = Path("/kaggle/input/yoruba-asr-benchmark")     # adjust to your dataset slug
if not PACK.exists():
    PACK = Path("data/asr_benchmark")                  # running locally
rows = list(csv.DictReader(open(PACK / "manifest.tsv", encoding="utf-8"), delimiter="\\t"))
print(f"{len(rows)} readings, {sum(int(r['sub_dots']) for r in rows)} sub-dots")
print("device:", "cuda" if torch.cuda.is_available() else "cpu")"""),

md("""## Access

`NCAIR1/Yoruba-ASR` is gated: request access on its model page first. The token
is read from a Kaggle secret or an environment variable and never written into
the notebook, because a notebook is a thing people share."""),

code("""HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")     # Add-ons -> Secrets
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("token found; gated repositories are available")
else:
    print("no token; gated repositories will be skipped with a clear error")"""),

md("""## Scoring

Self-contained, so the notebook does not need the engine installed. These are the
same rules the engine applies, and the comments say why each exists — scoring a
recogniser on raw string equality would report a model as broken for writing
`bí` where the corpus wrote `bíi`."""),

code("""COMBINING = re.compile(r"[\\u0300-\\u0309]")     # tone marks, not the dot below

def strip_tones(text):
    \"\"\"Tone marks off, sub-dots kept.

    A recogniser's tone marks drift -- this project measured `àbùbùtàn` for
    `àbùbùtán` -- and tone is read from pitch, not from the transcript. Sub-dots
    stay because they change which vowel a letter is.
    \"\"\"
    return unicodedata.normalize("NFC", COMBINING.sub("", unicodedata.normalize("NFD", text)))

def collapse(word):
    \"\"\"`bíi` and `bí` are the same word; only a repeated vowel differs.\"\"\"
    out = []
    for ch in word:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)

def key(word):
    word = "".join(c for c in unicodedata.normalize("NFC", word)
                   if not unicodedata.category(c).startswith("P"))
    return collapse(strip_tones(word)).lower()

def dots(text):
    return unicodedata.normalize("NFD", text).count("\\u0323")

def resegment(prompt, heard):
    \"\"\"Put the transcript back on the prompt's word boundaries.

    Yoruba is written with spaces the speech does not have, so `Ẹ káàbọ̀` comes
    back as `ekaabo` or `e ka bo` -- all fair readings of one continuous gesture.
    Measured on this corpus, Whisper agreed with the reference word count on 22
    of 40 utterances; imposing the prompt's boundaries took that to 38. Without
    this, a model is scored for the writing system rather than for what it heard.
    \"\"\"
    words = prompt.split()
    if len(words) < 2 or not heard.strip():
        return heard

    def fold(text):
        d = unicodedata.normalize("NFD", text)
        keys, index = [], []
        for i, ch in enumerate(d):
            if unicodedata.combining(ch) and ch != "\\u0323": continue
            if ch.isspace() or unicodedata.category(ch).startswith("P"): continue
            keys.append(ch.lower()); index.append(i)
        return "".join(keys), index, d

    pk, _, _ = fold(prompt)
    hk, hidx, htext = fold(heard)
    if not pk or not hk:
        return heard
    bounds, run = [], 0
    for w in words:
        run += len(fold(w)[0]); bounds.append(run)

    mapped = [0] * (len(pk) + 1)
    for _tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, pk, hk).get_opcodes():
        span = i2 - i1
        for i in range(i1, i2):
            mapped[i] = j1 + round(((i - i1) / span if span else 0) * (j2 - j1))
        mapped[i2] = j2
    mapped[len(pk)] = len(hk)

    def cut(pos):
        if pos >= len(hidx): return len(htext)
        at = hidx[pos]
        while at < len(htext) and unicodedata.combining(htext[at]): at += 1
        return at

    out, prev = [], 0
    for b in bounds:
        pos = mapped[min(b, len(pk))]
        piece = "".join(unicodedata.normalize("NFC", htext[cut(prev):cut(pos)]).split())
        if piece: out.append(piece)
        prev = pos
    return " ".join(out) if out else heard"""),

md("""## Candidates

Every one of these is a Colab-scale upload with single-digit downloads and no
published WER. Treat a win here as a reason to look closer, not as a result."""),

code("""CANDIDATES = {
    # the baseline, for a like-for-like number on this hardware
    "whisper-small-yoruba (shipped)": "LyngualLabs/whisper-small-yoruba",
    "whisper-medium-yoruba":          "neoform-ai/whisper-medium-yoruba",
    "large-v3-yoruba-v3":             "devalade/whisper-large-v3-yoruba-v3",
    "Largev2-Decodis_Comb_FT":        "RafatK/Whisper_Largev2-Yoruba-Decodis_Comb_FT",
    "NCAIR1/Yoruba-ASR":              "NCAIR1/Yoruba-ASR",   # gated; needs HF_TOKEN
}
CONTEXTS = [False, True]      # without and with prompt conditioning"""),

code("""def transcribe(model, processor, wav, prompt, use_context, device):
    samples, rate = sf.read(str(wav), dtype="float32")
    assert rate == 16000, f"{wav} is {rate} Hz; the pack should be 16 kHz"
    # Features must match the model's dtype. Loading in fp16 and feeding fp32
    # raises "Input type (float) and bias type (c10::Half) should be the same",
    # which is what this notebook did on every candidate the first time it ran.
    feats = processor.feature_extractor(
        samples, sampling_rate=16000,
        return_tensors="pt").input_features.to(device=device, dtype=model.dtype)
    kwargs = {"max_new_tokens": 128}
    if use_context:
        kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        ids = model.generate(feats, **kwargs)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    if use_context:
        head = text.lstrip()
        if head[:len(prompt)].lower() == prompt.lower():
            text = head[len(prompt):]
        # The decoder sometimes returns the context and stops. Left alone that
        # is an empty transcript scored as total failure; it is really the model
        # declining to transcribe, which is worth seeing as its own number.
        if not text.strip():
            return "", True
    return text.strip(), False


def load(model_id, device):
    \"\"\"Load a checkpoint, tolerating an incomplete upload.

    Several of these are Colab exports missing `preprocessor_config.json`, so
    the feature extractor cannot be built from the repository. Whisper's mel
    front end is fixed per model size, so borrowing it from the base checkpoint
    is exact rather than approximate -- and better than skipping the candidate.
    \"\"\"
    token = os.environ.get("HF_TOKEN") or None
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, token=token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    try:
        processor = WhisperProcessor.from_pretrained(model_id, token=token)
    except Exception:
        base = {32: "openai/whisper-large-v2", 24: "openai/whisper-medium",
                12: "openai/whisper-small"}[model.config.encoder_layers]
        print(f"   no preprocessor in the repo; using {base}'s")
        processor = WhisperProcessor.from_pretrained(base)
    return model.to(device).eval(), processor


def evaluate(name, model_id, device):
    model, processor = load(model_id, device)
    params = sum(p.numel() for p in model.parameters())

    out = []
    for use_context in CONTEXTS:
        ok = total = whole = phrases = 0
        kept = want_dots = parroted = 0
        elapsed = 0.0
        for row in rows:
            wav = PACK / "audio" / f"{row['utterance']}.flac"
            prompt = row["text"]
            start = time.perf_counter()
            heard, empty = transcribe(model, processor, wav, prompt, use_context, device)
            elapsed += time.perf_counter() - start
            parroted += int(empty)

            fixed = resegment(prompt, heard)
            p = [key(w) for w in prompt.split()]
            h = [key(w) for w in fixed.split()]
            hits = (sum(1 for a, b in zip(p, h) if a == b) if len(h) == len(p)
                    else sum(1 for a in p if a in h))
            if "words" in row["groups"]:
                ok += hits; total += len(p)
                whole += 1 if hits == len(p) else 0
                phrases += 1
            want_dots += dots(prompt)
            kept += min(dots(prompt), dots(fixed))

        out.append({
            "model": name, "params": params, "context": use_context,
            "per_word": ok / max(total, 1),
            "whole_phrase": whole / max(phrases, 1),
            "sub_dots": kept / max(want_dots, 1),
            "seconds": elapsed / len(rows),
            "parroted": parroted / len(rows),
        })
    del model; torch.cuda.empty_cache()
    return out"""),

code("""device = "cuda" if torch.cuda.is_available() else "cpu"
results = []
for name, model_id in CANDIDATES.items():
    print(f"\\n== {name}", flush=True)
    try:
        results += evaluate(name, model_id, device)
    except Exception as exc:
        print(f"   failed: {type(exc).__name__}: {exc}")

print(f"\\n{'model':<32}{'params':>8}{'ctx':>5}{'per-word':>10}"
      f"{'phrase':>9}{'sub-dots':>10}{'s/utt':>8}{'parroted':>10}")
for r in sorted(results, key=lambda r: -r["per_word"]):
    print(f"{r['model']:<32}{r['params']/1e6:>7.0f}M{str(r['context']):>5}"
          f"{r['per_word']:>9.0%}{r['whole_phrase']:>9.0%}{r['sub_dots']:>10.0%}"
          f"{r['seconds']:>8.2f}{r['parroted']:>10.0%}")

print("\\nThe bar, from the shipped model with prompt context:")
print("  97% per-word   88% whole phrase   98% sub-dots   ~1.0 s on 16 CPU cores")"""),

md("""## Reading the result

**A win on per-word alone is not a reason to switch.** Check three things:

- **`parroted`** — if it is above a few percent, the model cannot use prompt
  context, and its unconditioned score is its real ceiling. NCAIR1 returns the
  prompt and nothing else on about half of inputs.
- **`sub-dots`** — the app's learners fail on `ọ` versus `o`, and a model that
  drops the distinction is worse for teaching regardless of word accuracy.
- **`s/utt` on a GPU is not what you will deploy.** A 1.55B model measured at
  0.4 s on a T4 runs about 6.6 s on 16 CPU cores and several times worse on a
  4-vCPU droplet. Multiply, then decide whether the app can wait.

For context on where accuracy is actually lost: transcription is at 97% and tone
detection at 79%. A recogniser upgrade competes for effort with the tone model,
and the tone model is nineteen points from perfect."""),
]


def main() -> int:
    notebook = {
        "cells": CELLS,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "benchmark_asr_kaggle.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
