# Datasets

Every speech corpus and model this engine was built from, what each one
contributes, and where it came from. Clip counts and durations are measured
from the copies on disk, not quoted from the dataset cards, so they reflect
what actually reached training after filtering and deduplication.

## Speech corpora

### OpenSLR SLR86 — Crowdsourced high-quality Yoruba speech

<https://openslr.org/86/> ·
[`yo_ng_female.zip`](https://openslr.org/resources/86/yo_ng_female.zip) ·
[`yo_ng_male.zip`](https://openslr.org/resources/86/yo_ng_male.zip)

| | clips | hours |
|---|---:|---:|
| female | 1,892 | 2.06 |
| male | 1,691 | 1.97 |
| **total** | **3,583** | **4.03** |

Licence: **CC BY-SA 4.0** — attribution required, and derivative corpora must
carry the same licence.

The backbone of the project. It is fully diacritized, which is why it does
double duty: the orthography can be treated as ground truth for tone, so it is
the only corpus that can measure whether a detected tone is *correct* rather
than merely *confident*. Used for:

* the MFA acoustic model (`data/mfa_corpus`, 3,583 utterances)
* the GOP per-phone reference distributions
* pitch and tone calibration (`scripts/calibrate_pitch.py`)
* alignment statistics and the omission thresholds

### Tundragoon/IroyinSpeech

<https://huggingface.co/datasets/Tundragoon/IroyinSpeech>

3,061 clips · 5.01 h · the largest single contributor

A Common Voice export of Yoruba news-domain speech. Read rather than
spontaneous, and recorded close-mic, which is what made it worth ingesting:
the vowel classifier was trained on crowdsourced audio and tested on studio
audio, and that domain gap was costing more accuracy than any amount of extra
crowdsourced data recovered.

### At-Tawheed/yoruba-tts-ljspeech

<https://huggingface.co/datasets/At-Tawheed/yoruba-tts-ljspeech>

2,069 clips · 1.62 h

Single-speaker TTS-grade recordings in the LJSpeech layout. Clean and
consistent, but one voice, so it broadens the *recording conditions* the
classifier has seen without broadening the speaker population.

### thisniyi/yoruba-tts-dataset-cv_and_openslr

<https://huggingface.co/datasets/thisniyi/yoruba-tts-dataset-cv_and_openslr>

334 clips kept · 0.56 h

The name is a warning: this set is itself assembled from Common Voice **and
OpenSLR**, so most of it duplicates SLR86 and IroyinSpeech above. Ingestion
deduplicates on transcript, which is why only 334 of several thousand clips
survived. Included because the residue is genuinely new, but it is the
smallest contributor by an order of magnitude.

### Common Voice Scripted Speech 26.0 — Yoruba

<https://mozilladatacollective.com/datasets/cmqi6qutc00gmo507myna6k5s>

448 clips kept · 0.74 h · from 3,514 validated · **CC0-1.0**

Fully diacritized — 3,514 of 3,514 validated sentences carry tone marks and
sub-dots — and public domain, so it carries no distribution conditions at all.

The yield is small because IroyinSpeech is a subset of an earlier Common Voice
release: 3,062 clips are byte-identical between them. What remains is 448
recordings and 14 speakers the earlier export predates. Ingesting it forced the
deduplication rule to be corrected, which was worth more than the audio — see
`scripts/ingest_studio.py`. Under the old transcript-only rule this contributed
116 clips; keying on speaker as well recovered the other 332.

### Combined studio corpus

`data/studio_corpus` — **5,912 clips · 7.93 h** across the four sources above.
With SLR86 that is **9,495 clips · 11.96 h** of aligned Yoruba speech.

## Downloaded but not used

**Yoruba-TTS-Dataset** — 1,954 mp3 clips, ~5.8 h. Two independent blockers.

Not one of its 1,975 transcripts carries a diacritic: they are in missionary
orthography, so `yóò yí orúkọ` is written `yoo yi oruko`. That collapses `ẹ`/`e`
and `ọ`/`o`, which are exactly the contrasts the vowel classifier is weakest on,
and leaves no tone information at all. Diacritic restoration could recover some
of it, at the cost of injecting label noise into the most fragile classes.

Its licence is NOODL-1.0, research-only, forbidding commercial use and
redistribution — so it cannot ship in a commercial product regardless.

Available at
<https://mozilladatacollective.com/datasets/cmo1nlaah0071mk077mw0qhpv>; there is
an ingestion path in `scripts/prepare_dataset.py` that is deliberately not run.

(**Correction.** An earlier version of this file listed google/WaxalNLP and
michsethowusu/yoruba-speech-text-parallel here as evaluated and not ingested.
That was wrong -- see below. Both are in the shipped acoustic model and are the
bulk of it.)

## In the shipped acoustic model

These were trained on Kaggle and are not in the local `studio_corpus`, which is
why they are easy to overlook: `studio_corpus` feeds the *vowel classifier*,
where the studio-domain gap mattered, while these feed the *aligner*. Per
`models/yoruba_acoustic.manifest.json`, the shipped model is 37.93 hours over
2,047 speakers.

### michsethowusu/yoruba-speech-text-parallel

<https://huggingface.co/datasets/michsethowusu/yoruba-speech-text-parallel>

99,550 utterances - 24.0 h - the largest single contributor to the aligner

One- and two-word fragments averaging under a second (`ti kó`, `yóò sì`,
`ọmọ`), 16 kHz, 85.8% tone-marked. Too short to teach prosody or context, but
that volume of clean fragments is good phone coverage. Only shard 1 of 50
(32,941 rows) is held locally.

### google/WaxalNLP

<https://huggingface.co/datasets/google/WaxalNLP>

2,202 utterances - 6.30 h

Long read sentences, ~10 s each, at 48 kHz. Only **58.4% of lines carry a tone
mark**, and at that length an utterance with no mark anywhere is genuinely
undiacritized rather than legitimately all-mid. The shipped model was trained
with `require_tone_marks` disabled and so learned from the unmarked 42%; the
local ingest enforces the filter, which costs about that share of it.

### Local TTS

1,606 utterances - 4.16 h, staged on Kaggle from the same recording-tool
exports as `Yoruba-TTS-Dataset`.

## Worth pursuing

**YECS — Yoruba-English Code-Switching Corpus** (LyngualLabs),
<https://mozilladatacollective.com/datasets/cmo09pqp300gbnx07xcl42los>

120 hours, 99,930 validated pairs, 140 speakers, with *verified* tone marks and
word-level language tags — an order of magnitude beyond everything above
combined, from the same producer as the shipped feature extractor. Access is by
request.

Two conditions before building on it. It is code-switched, so Yoruba-only spans
have to be selected using the language tags. And NOODL-1.0 grants open use to
developers in developing nations while requiring benefit-sharing negotiation
with LyngualLabs for commercial entities in high-income countries, which needs
settling against the intended distribution before it becomes a dependency.

## Models

### LyngualLabs/mms-300m-yoruba-english

<https://huggingface.co/LyngualLabs/mms-300m-yoruba-english>

The feature extractor behind the vowel classifier, shipped in
`models/lyngual-yoruba`. Layer 16 hidden states, 1,024 dimensions.

This is the single change that moved the project furthest. The hand-built
feature set — 39 dimensions of formants, bandwidths and cepstra — reached
0.595 accuracy and 0.432 macro-F1. The same classifier on wav2vec2 features
reaches **0.873 / 0.811**. Concatenating the hand-built features on top added
0.0006, so Praat measurements were dropped from vowel identity entirely.

`En` is the clearest case: it scored 0.009 F1 on hand-built features and was
written off as unlearnable for want of data. On wav2vec2 features it reaches
**0.714 F1** from the same recordings. It was a feature-set limit, not a data
limit — worth remembering before buying more audio to fix an accuracy problem.

### facebook/mms-300m

<https://huggingface.co/facebook/mms-300m>

The untuned multilingual baseline, kept for comparison. The Yoruba-adapted
model above separates vowels better, which is the reason it ships.

## Lexicon

`data/dictionary/yoruba.dict` and `studio.dict` are generated by the project's
own G2P from the corpus transcripts, not downloaded. `overrides.tsv` holds
hand-written corrections. Word frequencies for the decoder prior come from
209,265 tokens counted across every corpus on disk
(`scripts/build_word_frequency.py`).

## Before shipping

SLR86's CC BY-SA 4.0 obligations are the firm ones: attribute, and license
derivative corpora alike. The three HuggingFace corpora need their licences
confirmed on the dataset cards — Common Voice material is normally CC0, but
the redistributed exports do not all state terms, and that should be checked
against the client's distribution plans rather than assumed.
