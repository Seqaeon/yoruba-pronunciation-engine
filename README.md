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

MFA 3.4.1 is installed in `.mfa-env/`, the engine is installed editable into
`/home/seqaeon/Downloads/venv`, and the corpus is staged. Everything up to
training runs; the acoustic model has not been trained yet.

| | |
|---|---|
| Corpus | 3,583 utterances, 4.03 h, 36 speakers |
| Tone-mark coverage | 100% |
| Dictionary | 3,413 entries, 62 phones, 0 G2P failures |
| Acoustic model | trained; 99.5% of utterances aligned |
| Tone detection | 74.7% held out, speaker-disjoint (chance 36.8%) |
| Tests | 121 passing |

## Quick start

```bash
# Install: see INSTALL.md (MFA needs conda; the engine is a plain pip install)

# 1. Stage the downloaded datasets into data/corpus/
python scripts/prepare_dataset.py

# 2. Audit
python -m yoruba_engine.cli audit

# 3. Build the pronunciation dictionary
python -m yoruba_engine.cli dictionary

# 4. Train the acoustic model (one-time, 1-2 h, CPU only)
jupyter notebook notebooks/train_mfa_yoruba.ipynb

# 5. Assess a recording
python -m yoruba_engine.cli assess \
    --reference data/corpus/wavs/yof_00295_00318039388.wav \
    --learner uploads/learner.wav \
    --text "òrí"
```

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
| **Gradient boosting (shipped)** | **74.7%** |

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
