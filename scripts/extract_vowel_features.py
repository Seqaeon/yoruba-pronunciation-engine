"""Measure vowel features from an aligned corpus into a cache file.

Writes the same four arrays the Kaggle notebook caches -- raw formant features,
raw cepstra, labels, speakers -- so caches from different corpora concatenate
directly. Measurement needs both the audio and the alignment and is the slow
step; everything downstream (normalization, splits, class weights, thresholds)
is arithmetic on the result and costs seconds.

    python scripts/extract_vowel_features.py \\
        --aligned output/studio_aligned --audio data/studio_corpus \\
        --out models/studio_features.npz
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
from yoruba_engine.segmental.classifier import combined_feature_names, vowel_features  # noqa: E402
from yoruba_engine.segmental.formants import FormantExtractor, normalize_vowel_space  # noqa: E402
from yoruba_engine.segmental.mfcc import MFCCExtractor  # noqa: E402

#: en and on are not Yoruba vowels; they appear only in alignments produced
#: before the grapheme-to-phoneme rules were corrected.
ARTEFACT = {"en": "e", "on": "o"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", help="Tag prepended to speaker ids, if any")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    audio_index = {p.stem: p for p in args.audio.rglob("*.wav")}
    textgrids = sorted(args.aligned.rglob("*.TextGrid"))
    print(f"{len(textgrids):,} TextGrids, {len(audio_index):,} recordings")

    formant_extractor, mfcc_extractor = FormantExtractor(), MFCCExtractor()
    formants_out, cepstra_out, labels_out, speakers_out = [], [], [], []
    processed = skipped = 0
    started = time.time()

    for number, textgrid in enumerate(textgrids, start=1):
        audio = audio_index.get(textgrid.stem)
        if audio is None:
            skipped += 1
            continue
        try:
            alignment = alignment_from_textgrid(textgrid, utterance_id=textgrid.stem)
            formants = formant_extractor.analyse(audio)
            times, frames = mfcc_extractor.analyse(audio)
        except Exception:
            skipped += 1
            continue

        measured = []
        for phone in alignment.phones:
            if not is_vowel_phone(phone.label):
                continue
            f = formant_extractor.measure_segment(formants, phone.start_s, phone.end_s)
            c = mfcc_extractor.measure_segment(times, frames, phone.start_s, phone.end_s)
            if f and c:
                measured.append((phone.label, f, c))
        if not measured:
            continue

        # Formant normalization is per utterance and relative to the speaker's
        # own range, which is exactly what the verifier does at inference.
        speaker_f1, speaker_f2 = normalize_vowel_space([m[1] for m in measured])
        speaker = audio.parent.name
        if args.source:
            speaker = f"{args.source}_{speaker}"
        for label, f, c in measured:
            base = label.split("_")[0]
            formants_out.append(vowel_features(f, speaker_f1, speaker_f2))
            cepstra_out.append(list(c))
            labels_out.append(ARTEFACT.get(base, base))
            speakers_out.append(speaker)

        processed += 1
        if number % 250 == 0:
            rate = number / max(time.time() - started, 1e-9)
            remaining = (len(textgrids) - number) / max(rate, 1e-9) / 60
            print(f"  {number:,}/{len(textgrids):,}  {len(labels_out):,} vowels  "
                  f"~{remaining:.0f} min left", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        formants=np.asarray(formants_out, dtype="float32"),
        cepstra=np.asarray(cepstra_out, dtype="float32"),
        labels=np.asarray(labels_out),
        speakers=np.asarray(speakers_out),
        feature_names=np.asarray(list(combined_feature_names())),
    )
    from collections import Counter

    counts = Counter(labels_out)
    print(f"\n{len(labels_out):,} vowels from {processed:,} utterances "
          f"({skipped} skipped), {len(set(speakers_out)):,} speakers")
    print("class balance:", dict(sorted(counts.items())))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB) "
          f"in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
