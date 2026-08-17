# Installation Guide

## Overview

Two pieces install separately:

1. **The `yoruba_engine` package** — pure Python, installs with pip into any
   virtualenv.
2. **Montreal Forced Aligner** — ships compiled Kaldi and OpenFST binaries, so it
   needs conda. It is not pip-installable.

The engine drives MFA as a **subprocess**, never as an import. That is what lets
a plain virtualenv run the whole pipeline while MFA lives in its own conda
environment — no environment juggling, and no need to activate conda.

Total install: about 8 GB. MFA accounts for 1.65 GB with a CPU-only Kaldi
(5.34 GB if conda resolves the CUDA build -- pin it, see step 2), the
speech model for vowel identity is 1.26 GB, and the trained artifacts are 64 MB.

## 1. System prerequisites

### Linux (Debian/Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential ffmpeg libsndfile1 git curl
```

`ffmpeg` is only required if your audio is not already WAV.

### macOS

```bash
brew install ffmpeg libsndfile
```

### Windows

Use WSL2 with Ubuntu and follow the Linux instructions.

## 2. Install Montreal Forced Aligner

micromamba is a single static binary that does not modify your shell startup
files, which makes it the least invasive way to get conda packages. Conda,
Mamba, or Miniforge all work equally well if you already have one.

From the project root:

```bash
# Install micromamba (skip if you already have conda/mamba)
mkdir -p ~/.local/bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xj -C /tmp bin/micromamba
mv /tmp/bin/micromamba ~/.local/bin/micromamba
chmod +x ~/.local/bin/micromamba

# Create the aligner environment inside the project
export MAMBA_ROOT_PREFIX=~/.local/share/mamba
~/.local/bin/micromamba create -y -p .mfa-env \
  -c conda-forge python=3.11 montreal-forced-aligner=3.4.1 "kaldi=*=cpu*"
```

macOS: replace `linux-64` with `osx-64` (Intel) or `osx-arm64` (Apple silicon).

Verify:

```bash
.mfa-env/bin/mfa version      # expect 3.x
```

### Why the Kaldi build is pinned

`"kaldi=*=cpu*"` is not decoration. Unpinned, conda resolves per machine: a box
with CUDA present gets a GPU build whose every Kaldi binary links
`libcuda.so.1` — from the NVIDIA *driver*, not from the environment — so those
binaries will not even load on a CPU-only server:

```
error while loading shared libraries: libcuda.so.1: cannot open shared object file
```

It is also 5.34 GB against 1.65 GB, and 3.68 GB of that is CUDA math libraries
(`libmagma` alone is 1.26 GB) that forced alignment never meaningfully uses.

Pin MFA's version too. `models/gop_reference.json` and
`models/word_duration_stats.json` are measurements of one specific aligner's
behaviour — per-phone likelihood means and 5th-percentile durations. A different
Kaldi build shifts boundaries, both files become quietly wrong, and nothing
raises an error; the feedback just degrades.

### How the engine finds MFA

`EngineConfig` resolves the `mfa` binary in this order:

1. `$MFA_BIN`, if set
2. `.mfa-env/bin/mfa` under the project root
3. whatever `mfa` is on `PATH`

Installing to `.mfa-env/` as above means it is found with no further
configuration. If you install elsewhere, set `MFA_BIN` to the absolute path:

```bash
export MFA_BIN=/path/to/your/env/bin/mfa
```

The engine also **prepends that binary's directory to `PATH`** when it invokes
MFA. This is required, not a nicety: MFA shells out to OpenFST tools such as
`fstcompile` that live beside it, and without the directory on `PATH` it fails
with `ThirdpartyError: Could not find 'fstcompile'` even when called by absolute
path.

## 3. Install the engine

Any Python 3.10+ virtualenv works. Create one, or use an existing environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install praat-parselmouth
```

Install **editable** (`-e`). The project-local MFA lookup and the default corpus
paths are derived from the package's location on disk; a non-editable install
resolves them into `site-packages` instead.

Optional extras:

```bash
pip install -e ".[train]"             # jupyter, matplotlib, pandas, librosa
pip install -e ".[tone-classifier]"   # scikit-learn, joblib
pip install -e ".[api]"               # FastAPI service layer
pip install -e ".[dev]"               # pytest, mypy, ruff
```

Parselmouth is strongly recommended. Without it the engine falls back to a
pure-Python pitch tracker that is accurate enough for tests but not for
production.

## 3b. Speech model for vowel identity

```bash
mkdir -p models/lyngual-yoruba
wget -c -O models/lyngual-yoruba/model.safetensors \
  https://huggingface.co/LyngualLabs/mms-300m-yoruba-english/resolve/main/model.safetensors

python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
  models/lyngual-yoruba/model.safetensors
# expect 766a4e6c9c98fbb59bb4a2850c0ab0677a1197969564f28aab9b50815368aeb3
```

1.26 GB. The config files it needs (`config.json`, `vocab.json`, and four more,
about 5 KB total) are already committed under `models/lyngual-yoruba/`.

Use `wget -c`, not the `huggingface_hub` downloader, if you need to resume an
interrupted transfer. HuggingFace **preallocates the blob to its full final
size**, so a partial download looks complete to anything reading the file
length -- `ls`, `stat`, and `curl -C -` all resume from the wrong offset and
produce a corrupt file that passes a size check and fails its hash. `du -k`
(allocated blocks) is the only honest progress measure on a partial HF blob.

This step is optional. Without it the engine logs a warning and disables
vowel-identity checking; alignment, omission detection, tone comparison and GOP
all still work. Which feature backend is used is read from the classifier's own
metadata, so a mismatch cannot happen silently.

## 4. Verify

```bash
python -m pytest                        # needs no audio, no MFA, no model
python -m yoruba_engine.cli health
```

`health` reports the acoustic model and tone classifier as missing until you
train them. That is expected on a fresh install — they are steps 7 and 8.

## 5. Prepare the corpus

Place your dataset under `data/` and stage it:

```bash
python scripts/prepare_dataset.py
```

This writes `data/corpus/line_index.tsv` and symlinks audio into
`data/corpus/wavs/`, costing no extra disk. See `data/corpus/README.md` for the
expected input layouts and audio requirements.

Sources whose transcripts lack tone marks are excluded by default; they can be
included for acoustic training only, with `--include-tts`.

## 6. Audit and build the dictionary

```bash
python -m yoruba_engine.cli audit
python -m yoruba_engine.cli dictionary
```

Read `output/dataset_report.txt` before going further. **If tone-mark coverage is
below 50%, stop** — tone comparison cannot work on an undiacritized corpus, and
no amount of modelling will fix it.

Check that `output/g2p_failures.csv` is empty. If not, add the listed words to
`data/dictionary/overrides.tsv` and re-run.

## 7. Train the acoustic model

Required once, CPU only. There is no reliable pretrained Yoruba model.

```bash
jupyter notebook notebooks/train_mfa_yoruba.ipynb
```

Launch Jupyter from the virtualenv where you installed the engine.

Expect roughly **3–4 hours for 4 hours of audio on 15 cores**. Time scales with
corpus size and inversely with `MFA_JOBS`; the speaker-adaptive (SAT) stage
dominates.

## 8. Train the tone classifier

Optional but worth it: about 11 accuracy points over the threshold rules.
Takes about a minute once the acoustic model exists.

```bash
python scripts/train_tone_classifier.py
```

## 9. Confirm and try it

```bash
python -m yoruba_engine.cli health     # all entries available
python scripts/demo_assess.py          # end-to-end assessment on real audio
```

---

## Troubleshooting

**`ThirdpartyError: Could not find 'fstcompile'`**
MFA's `bin` directory is not on `PATH`. The engine handles this automatically; if
you are calling `mfa` yourself, run `export PATH="$PWD/.mfa-env/bin:$PATH"`.

**`mfa: command not found`**
Set `MFA_BIN` to the absolute path of the binary.

**Engine imports but cannot find the corpus**
It was installed non-editable. Reinstall with `pip install -e`.

**MFA hangs or runs out of memory while training**
Lower the job count: `export MFA_JOBS=2`. Each job holds its own feature buffers.

**Permission errors writing MFA temp files**
MFA writes to `~/Documents/MFA` by default. Set `MFA_ROOT_DIR` to redirect it.

**Parselmouth fails to install**
It needs a C++ toolchain (`build-essential`). The engine falls back to a slower,
less accurate pitch tracker if it is unavailable.

**Notebook cannot see engine changes**
Restart the kernel. Python caches imported modules.

## Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MFA_BIN` | Path to the `mfa` executable | `.mfa-env/bin/mfa`, else `PATH` |
| `MFA_JOBS` | Parallel jobs for alignment and training | `4` |
| `MFA_ROOT_DIR` | Where MFA keeps temporary files | `~/Documents/MFA` |

## Reference: a known-good install

For comparison, the development machine runs MFA 3.4.1 in `.mfa-env/`, the engine
installed editable into a Python 3.12 virtualenv, on a 16-core Linux box.
Training the acoustic model on 4.03 hours of audio took 3 h 23 min at
`MFA_JOBS=15`; the tone classifier took under a minute.
