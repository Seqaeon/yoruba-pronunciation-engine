# Yoruba Speech and Tone Comparison Engine

Compares a learner's spoken Yoruba against a native-speaker reference recording
and reports whether the words match, whether the tones match, and an overall
pronunciation score with actionable feedback.

Built for integration into a Yoruba language-learning application.

## What it does

```
Learner audio ─┐
               ├─→ Forced alignment ─→ Pitch extraction ─→ Tone detection ─┐
Reference audio┘                                                           │
                                                                           ▼
                                          Comparison ─→ Score + feedback report
```

Example output:

```
Word: òrí
Expected: Low-High
Detected: High-High
Tone mismatch on first vowel.
```

## Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| 1 | `audit` | Validates the corpus, extracts vocabulary, checks diacritic coverage |
| 2 | `g2p` | Generates the MFA pronunciation dictionary from Yoruba orthography |
| 3 | `alignment` | Runs Montreal Forced Aligner; word- and phone-level TextGrids |
| 4 | `pitch` | Extracts F0 contours over aligned vowels (Praat/Parselmouth) |
| 5 | `tone` | Classifies High/Mid/Low, removes downdrift, compares patterns |
| 6 | `compare` | Aligns word sequences, scores text and tone similarity |
| 7 | `report` | Renders text and JSON assessment reports |

## Current state

Trained and running end to end. Vowel identity uses a self-supervised speech
model; alignment, tone and duration remain MFA-based.

| | |
|---|---|
| Acoustic model | **37.9 h**, 106,441 utterances, 2,047 speaker groups |
| Alignment | **99.65%** of training utterances; **100%** of the 3,583-utterance local corpus |
| Dictionary | 9,092 entries, 31 phones, 0 G2P failures |
| Vowel identity | **87.3%** / macro F1 **0.811** on 31 held-out **studio** speakers |
| Tone detection | 73.0% held out, speaker-disjoint (chance 38.1%) |
| Omission detection | 95.9% caught at 4.6% false alarms |
| Substitution, vowel check | **94.0%** caught at 3.8% false alarms |
| Substitution, GOP | 87.9% caught at 2.0% false alarms |
| Naming what was said | 1.5% false-substitution rate |
| Tests | **183 passing** |

### The features were the ceiling, not the data

Vowel identity ran at 59.5% for most of this project's life on 39 hand-designed
measurements: three formants, three formant bandwidths, and 13 cepstral means
with their standard deviations. Replacing them with a mean-pooled layer of a
Yoruba-fine-tuned wav2vec2 model, changing nothing else:

| features | dims | accuracy | macro F1 | an | En | in | On | un |
|---|---|---|---|---|---|---|---|---|
| 39 hand-designed | 39 | 0.595 | 0.432 | 0.249 | 0.009 | 0.129 | 0.307 | 0.256 |
| **wav2vec2 layer 16** | 1024 | **0.873** | **0.811** | **0.791** | **0.714** | **0.696** | **0.813** | **0.718** |
| both concatenated | 1063 | 0.876 | 0.812 | 0.779 | 0.727 | 0.666 | 0.811 | 0.727 |

Nasal F1 improves 3–5×. All six minimal-pair contrasts now clear 91%:

```
un~an 95.8%   an~un 92.6%      (òkan / òkun)
o~ọ   92.9%   ọ~o   95.2%      (òkun / ọkun)
e~ẹ   91.6%   ẹ~e   94.2%
```

Two things worth taking from that table. **Concatenating the old features adds
+0.0006 macro F1** — noise — so Praat is no longer used for vowel identity at
all, only for pitch. And `ẹn`, which this README previously described as
unlearnable at F1 0.009 (364 tokens across seven corpora, 63% of them the single
word *yẹn*), reaches **0.714**. It was unlearnable with 39 numbers; the data was
always sufficient.

Layer 16 of 24, not the last — a CTC model's final layer is optimised to predict
characters, and that objective discards the detail separating `an` from `un`:

```
layer      4      8     12     16     20     24
accuracy 0.703  0.763  0.780  0.798  0.785  0.771
```

### Thresholds are calibrated, and re-calibrated when the model changes

Every operating point is measured from both error directions rather than chosen.
Improving the classifier made all of them better *and* changed where they belong:

| threshold | before | after | effect |
|---|---|---|---|
| `suspect_probability` / `weak_probability` | 0.02 / 0.15 | **0.20 / 0.30** | caught 66.3% → **94.0%** |
| decoder `DEFAULT_MARGIN` | 3.0 | **2.0** | false 3.2% → **1.5%** |
| `min_duration_fraction` | 0.5 | 0.6 | omissions 78.2% → 95.9% |

Two lessons learned the hard way, both encoded in the scripts:

**Calibrate on the audio you will judge.** The omission threshold was briefly
set to 0.8 from corpus speech and produced false "you did not say this" reports
on real learner recordings, where the aligner fits an unfamiliar voice less
tightly. See `scripts/calibrate_omissions.py`.

**Evaluate on the domain you will serve.** An earlier held-out set pooled studio
and crowdsourced speakers; because 94.5% of training audio is crowdsourced, that
inflated everything — the `un`/`an` contrast read 92.9% pooled and 69.7% on
studio speech alone. All figures above are studio-only.

### MFA stays

CTC forced alignment was evaluated as a replacement, which would have removed
1.65 GB of Kaldi and a ~35 s per-call startup. It aligns 200/200 utterances in
real Yoruba orthography with no romanization, and after trimming silence with a
15-line energy VAD its **word** boundaries agree with MFA to 40 ms median.

But nothing downstream reads words. On **phone** spans — what GOP, the classifier
and the duration model actually consume — the median vowel duration error is
**52% of a typical 97 ms vowel**. Not adequate. `scripts/ctc_align_compare.py`,
`ctc_align_trimmed.py` and `ctc_phone_precision.py` retain the measurements.

## Quick start

```bash
# 1. Install (see "Installation" below, or INSTALL.md for detail)
micromamba create -y -p .mfa-env -c conda-forge \
    python=3.11 montreal-forced-aligner "kaldi=*=cpu*"
pip install -e . -r requirements.txt

# 2. Fetch the trained artifacts (64 MB, verified against models/MANIFEST.json)
python scripts/fetch_models.py

# 3. Fetch the speech model used for vowel identity (1.26 GB)
mkdir -p models/lyngual-yoruba && wget -c -O models/lyngual-yoruba/model.safetensors \
    https://huggingface.co/LyngualLabs/mms-300m-yoruba-english/resolve/main/model.safetensors

# 4. Check everything resolved
python -m yoruba_engine.cli health

# 5. Assess a recording
python -m yoruba_engine.cli assess \
    --reference data/corpus/wavs/yof_00295_00099759266.wav \
    --learner uploads/learner.wav \
    --text "Àbùbùtán bíi omi òkun."
```

## Installation

Full detail and troubleshooting live in [INSTALL.md](INSTALL.md). The short
version, and the three things that actually go wrong:

### 1. Montreal Forced Aligner (conda only)

```bash
micromamba create -y -p .mfa-env -c conda-forge \
    python=3.11 montreal-forced-aligner=3.4.1 "kaldi=*=cpu*"
```

MFA is on PyPI but **`kalpy`, its Kaldi binding, is not** — and the engine shells
out to Kaldi executables (`gmm-align-compiled`, `fstcompile`,
`compute-mfcc-feats`) that pip cannot provide. There is no `requirements.txt`
route to a working aligner.

`"kaldi=*=cpu*"` is not optional. Without the pin, conda resolves per machine: a
box with CUDA gets a GPU build whose binaries link `libcuda.so.1` from the NVIDIA
*driver*, so they will not even load on a CPU-only server. The CPU build is also
~1.4 GB against ~5.5 GB for the CUDA one.

### 2. The engine itself

```bash
python -m venv venv && source venv/bin/activate
pip install -e . -r requirements.txt
```

Plain pip. `praat-parselmouth`, `scikit-learn`, `soundfile`, `numpy`, `joblib`.

### 3. The speech model for vowel identity

```bash
mkdir -p models/lyngual-yoruba
wget -c -O models/lyngual-yoruba/model.safetensors \
  https://huggingface.co/LyngualLabs/mms-300m-yoruba-english/resolve/main/model.safetensors
```

1.26 GB, `Wav2Vec2ForCTC`, 300M parameters. Its 97-token vocabulary carries every
distinction the engine reasons about — `ẹ ọ ṣ`, all eight tone-marked vowels, and
the syllabic nasals `ń ǹ`. A romanized model would have folded those away before
the audio was ever seen.

Inference runs at about **4.7× realtime on CPU**, so no GPU is needed to serve —
0.64 s against forced alignment's ~35 s is invisible. Resident memory is roughly
2.5 GB, which is what sets the 8 GB recommendation below.

Without it the engine still runs: omission detection, tone comparison and GOP all
work, and `VowelVerifier` logs a warning and disables vowel-identity checking
rather than failing. Which backend is used is read from the classifier's own
metadata, never guessed.

The Apache-2.0 tag on that repository sits on weights derived from
`facebook/mms-300m`, which Meta licenses CC-BY-NC-4.0. That tag is the uploader's
assertion, not an audit — verify before shipping commercially.

### 4. Trained artifacts

```bash
python scripts/fetch_models.py          # download and verify
python scripts/fetch_models.py --check  # verify only
```

64 MB of model files are too large for git and live on a GitHub Release.
`models/MANIFEST.json` carries their sha256 sums. If the repository is private,
run `gh auth login` first — the fetcher uses `gh` because private release assets
return 404 over anonymous HTTP.

**Why the dictionary is committed but the models are not:** MFA does not report a
phone-set mismatch. Align a dictionary whose phones differ from the model's and
it exits zero, writing TextGrids in which every segment carries the same
meaningless label. The dictionary is 342 KB, so it ships with the code and the
fetcher refuses to proceed if its checksum has changed.

### Retraining changes the calibration

`models/gop_reference.json`, `models/word_duration_stats.json` and
`models/word_frequency.json` are measurements of **this** acoustic model. Train
your own and they are silently wrong — no error, just worse feedback. Regenerate:

```bash
python scripts/measure_alignment_stats.py     # duration thresholds
python scripts/build_gop_reference.py         # per-phone likelihood means
python scripts/train_tone_classifier.py       # tone, on the new boundaries
python scripts/build_word_frequency.py        # decoder prior
```

Changing the **classifier** invalidates a different set — the probability
thresholds, because a sharper model produces differently-scaled distributions:

```bash
python scripts/calibrate_suspect.py    # suspect / weak probabilities
python scripts/calibrate_decoder.py    # decoder margin
python scripts/calibrate_omissions.py  # duration fraction
```

Both calibration scripts score through `VowelVerifier` rather than rebuilding
feature rows, so they follow whatever backend the classifier declares. They used
to hard-code the 39-measurement layout and broke the moment the model changed —
a calibration script measuring a different path than production is worse than no
calibration.

## The datasets

`scripts/prepare_dataset.py` merges the raw downloads into `data/corpus/`,
symlinking audio so it costs no extra disk.

**`yo_ng_female` + `yo_ng_male`** — OpenSLR layout, 48 kHz mono WAV, **fully
diacritized**. Staged by default; this is the tone reference.

**`Yoruba-TTS-Dataset`** — 1,949 MP3 clips with **no tone marks**. Excluded by
default. It can contribute audio to acoustic training via `--include-tts`, but
only with tone-stripped phones, and it can never serve as a tone reference.

### Non-speech annotations

The transcripts mark non-speech events inline — `[snap]` (session clapper) 700x,
`[breath]` 395x, `[external]` 192x, plus `[abrupt]` and `[hesitation]`. All
collapse to one `[noise]` token mapped to MFA's `spn` phone.

They are deliberately not deleted: an unmodelled snap gets absorbed into the
neighbouring vowel, and that vowel's F0 — the measurement tone detection reads —
is then wrong. They are excluded from the vocabulary and dropped before scoring,
so learners are never penalized for not reproducing a reference speaker's
breathing.

## Running an assessment

```bash
python scripts/demo_assess.py                    # picks a real pair from the corpus
python scripts/demo_assess.py --no-classifier    # compare against the threshold rules

python scripts/demo_assess.py \
    --reference data/corpus/wavs/yof_00295_00099759266.wav \
    --learner   my_attempt.wav \
    --text      "Àbùbùtán bíi omi òkun."
```

Learner audio must be mono WAV:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 my_attempt.wav
```

Reports are written to `output/reports/` as both text and JSON. Each assessment
runs the aligner twice, so expect 30–60 seconds.

### Getting the learner's transcript

The brief asks for a **learner transcript** and a **text similarity** score, which
means the engine has to work out what the learner actually said. There are two
ways to do that, and they are very different in deployment cost.

**Constrained detection (default, 43 MB).** In prompted practice you already know
the target sentence, so open-vocabulary recognition solves a harder problem than
you have. Forced alignment must place every prompt word somewhere in the audio,
so a word the learner skipped gets crushed into a sliver. Words far shorter than
their length implies are reported as not detected.

The threshold scales with word length, because duration does — one phone runs
about 120 ms on this corpus, eight phones about 780 ms. A flat threshold would
either reject common short words like `ti` and the syllabic nasal `ń`, or wave
through a long word crushed to a tenth of its normal length.

Measured over 10,158 genuine corpus words: **0.09% false rejections** (9 words).
On a learner recording of a completely different sentence it flagged 2 of 4
prompt words as missing, dropping text similarity to 50%.

It catches **omissions, not substitutions**: a learner saying a different word of
similar length in the same slot still passes. That is the price of not running a
recognizer.

Regenerate the duration statistics after retraining the acoustic model:

```bash
python scripts/measure_alignment_stats.py
```

**Open ASR (`MMSTranscriptProvider`, ~4 GB).** Needed only for free speech, where
the learner is not repeating a known prompt. Meta's MMS has a Yoruba adapter, so
no training is required, but the model is loaded once per worker and costs about
4 GB of RAM or VRAM. It emits **undiacritized** text, so it can answer "which
words" but never supplies tone — tone is always measured acoustically.

```python
from yoruba_engine.asr.providers import MMSTranscriptProvider
from yoruba_engine.asr.restore import DiacriticRestorer
from yoruba_engine.pipeline import AssessmentPipeline

restorer = DiacriticRestorer.from_word_frequency(Path("output/word_frequency.csv"))
pipeline = AssessmentPipeline(
    transcript_provider=MMSTranscriptProvider(restorer=restorer)
)
```

Diacritics are restored only so the aligner can find the words in the
pronunciation dictionary; the restored spelling is never used as tone evidence.

### Verifying the learner said the prompt

Forced alignment cannot fail: given any audio and any text it returns an
alignment. In prompted mode the learner's transcript *is* the target text, so
without a check a learner who said something completely different would be
compared against the target and score 100%.

The engine therefore measures **speech coverage** — the fraction of the recording
the aligned words account for. Below `ScoringConfig.min_speech_coverage` (0.35)
the attempt is marked unverified, the overall score is zeroed, and the report
says so instead of congratulating the learner.

This is a partial safeguard, not a solved problem. Measured on this corpus,
correct alignments cover 40–92% of the audio and mismatched ones 22–53%; the
distributions overlap, so the check caught about half of deliberately mismatched
utterances with under 1% false rejections. A stronger check would use acoustic
likelihood, which MFA does not expose through its CLI.

## Library use

```python
from pathlib import Path
from yoruba_engine import AssessmentPipeline

pipeline = AssessmentPipeline()

result = pipeline.assess(
    reference_audio=Path("reference.wav"),
    learner_audio=Path("learner.wav"),
    target_text="òrí",
)

print(f"Score: {result.overall_score:.0%}")
for error in result.tone_errors:
    print(error.describe())
```

## Model training

**One training step is required: the MFA acoustic model.** There is no reliable
pretrained Yoruba model, and multilingual substitutes produce vowel boundaries
too imprecise for tone measurement.

It is not deep learning. MFA's trainer is a GMM-HMM (Kaldi) pipeline that runs on
CPU in hours, and trains on your own audio plus transcripts with no extra
labelling. `notebooks/train_mfa_yoruba.ipynb` walks through it end to end,
including tone-detector validation and threshold tuning.

Nothing else in the pipeline needs training. G2P, pitch extraction, tone
classification, and scoring are all rule-based or signal processing.

### What about ASR?

The engine does **not** need speech recognition, by design. In prompted practice
— the app shows a word, the learner repeats it — the target text is already
known, so forced alignment verifies the attempt directly.

Free-form speech would need Yoruba ASR. That path is stubbed out in
`yoruba_engine.asr.providers.ASRTranscriptProvider`; subclass it and implement
`_recognize()`. Be aware that most Yoruba ASR emits undiacritized text, which is
useless for tone comparison unless the model was fine-tuned on diacritized
transcripts.

## Requirements that will sink the project if unmet

**Transcripts must carry tone marks.** Yoruba tone is written with diacritics
(`òrí`, not `ori`). Against an undiacritized corpus, every vowel reads as Mid and
the engine reports perfect tonal accuracy for every learner regardless of what
they said. Stage 1 measures this and refuses to proceed below 50% coverage.

**Sub-dot characters must be preserved.** `ẹ`, `ọ`, and `ṣ` are distinct
phonemes, not styled variants of `e`, `o`, `s`.

## Configuration

Everything tunable lives in `yoruba_engine/config.py`:

- `PitchConfig` — F0 floor/ceiling, frame step, vowel edge trimming
- `ToneConfig` — H/M/L semitone thresholds, confidence floor
- `ScoringConfig` — text/tone weights, pass threshold
- `MFAConfig` — model and dictionary paths, parallelism, beam widths

`ToneConfig` defaults are tuned against this corpus by grid search (section 9 of
the training notebook). Re-run the sweep if you change corpus or recording setup.

### Tone detection accuracy

Measured on **held-out speakers** (9 of 36 excluded from training entirely — a
random split would let the model memorize individual voices, since features are
normalized per speaker):

| Method | Accuracy |
|---|---|
| Majority class | 36.8% |
| Threshold rules | 63.7% |
| Logistic regression | 64.3% |
| **Gradient boosting (shipped)** | **73.0%** |

Per-tone recall for the shipped model: High 80%, Mid 74%, Low 68%.

`AssessmentPipeline` loads `models/tone_classifier.joblib` automatically when
present and falls back to the threshold rules otherwise, so the rules remain a
working default with no model file and no scikit-learn.

```python
AssessmentPipeline()                            # classifier if present
AssessmentPipeline(use_tone_classifier=False)   # force the rules
AssessmentPipeline(use_tone_classifier=True)    # require it
```

Retrain with `python scripts/train_tone_classifier.py` whenever you retrain the
acoustic model or change `tone/features.py`.

The remaining error is not all model error: orthographic tone is *lexical* tone,
while the microphone records tone after downstep, sandhi and coarticulation have
applied, so the labels and the signal genuinely disagree in places. Raise
`ToneConfig.min_confidence` if you would rather say nothing than mislead a
learner — with the classifier, confidence is a real predicted probability.

## Tests

```bash
/home/seqaeon/Downloads/venv/bin/python -m pytest
```

94 tests.

The suite covers orthography and Unicode handling, G2P, tone classification and
downdrift correction, and word alignment and scoring. It needs no audio, no MFA,
and no trained model.

## Project layout

```
scripts/prepare_dataset.py         # merge raw downloads into data/corpus/
scripts/train_tone_classifier.py   # train the tone classifier from aligned audio
src/yoruba_engine/
├── config.py           # all tunable parameters
├── datatypes.py        # shared dataclasses
├── exceptions.py       # typed exception hierarchy
├── pipeline.py         # end-to-end orchestration (the FastAPI seam)
├── cli.py              # command-line interface
├── audit/              # stage 1
├── g2p/                # stage 2
├── alignment/          # stage 3
├── pitch/              # stage 4
├── tone/               # stage 5
├── compare/            # stage 6
├── report/             # stage 7
└── asr/                # transcript providers (ASR placeholder)

notebooks/train_mfa_yoruba.ipynb   # acoustic model training
data/corpus/                       # your dataset (placeholder provided)
data/dictionary/                   # generated dict + manual overrides
models/                            # trained acoustic model
```

## API integration

`AssessmentPipeline` is the intended seam for a FastAPI layer: construct it once
at startup, call `assess()` per request. `ReportGenerator.to_dict()` produces the
response shape. `pipeline.health_check()` backs a `/health` endpoint.

All exceptions derive from `YorubaEngineError` with typed subclasses, so handlers
can map them to status codes without parsing messages.
