"""Train the vowel classifier on wav2vec2 features, and decide what to keep.

Three arms on identical speakers and an identical split:

* the 39 hand-designed measurements (formants, bandwidths, cepstral means)
* wav2vec2 layer 16, mean-pooled over each vowel
* both concatenated

The third arm is the one that decides whether the Praat path stays in the engine
at all. If concatenating adds nothing over wav2vec2 alone, then formant and MFCC
extraction -- two Praat analyses per assessment -- can be dropped from the
inference path entirely.

Evaluation is studio-only and speaker-disjoint, for the reason established
earlier: pooling crowdsourced speakers into the held-out set inflates every
number, because 94.5% of training audio is crowdsourced.

    python scripts/train_vowel_wav2vec.py
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.segmental.classifier import VowelClassifier, build_estimator  # noqa: E402

#: Speaker prefixes that are real people with fully diacritized transcripts.
STUDIO = ("yof_", "yom_", "iroyin_")
NASAL = ["an", "En", "in", "On", "un"]
CONTRASTS = [("un", "an"), ("an", "un"), ("o", "O"), ("O", "o"), ("e", "E"), ("E", "e")]


def is_studio(speaker: str) -> bool:
    return speaker.startswith(STUDIO) and "_unk" not in speaker


def held_out(speaker: str, percent: int = 25) -> bool:
    return int(hashlib.md5(speaker.encode("utf-8")).hexdigest()[:8], 16) % 100 < percent


def load_hand(paths):
    formants, cepstra, labels, speakers = [], [], [], []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        formants.append(d["formants"].astype(np.float32))
        cepstra.append(d["cepstra"].astype(np.float32))
        labels.append(d["labels"].astype(str))
        speakers.append(d["speakers"].astype(str))
    return (np.hstack([np.vstack(formants), np.vstack(cepstra)]),
            np.concatenate(labels), np.concatenate(speakers))


def report(name, X, y, labels_index, label_set, train, test):
    from sklearn.metrics import accuracy_score, f1_score

    model = build_estimator(balanced=True).fit(X[train], y[train])
    predicted = model.predict(X[test])
    proba = model.predict_proba(X[test])
    truth = y[test]
    accuracy = accuracy_score(truth, predicted)
    macro = f1_score(truth, predicted, average="macro", zero_division=0)
    per_class = f1_score(truth, predicted, average=None,
                         labels=list(range(len(label_set))), zero_division=0)
    cells = []
    for expected, competitor in CONTRASTS:
        mask = truth == labels_index[expected]
        if not mask.any():
            cells.append(f"{'-':>7}")
            continue
        ratio = np.log(np.clip(proba[mask][:, labels_index[expected]], 1e-9, 1)
                       / np.clip(proba[mask][:, labels_index[competitor]], 1e-9, 1))
        cells.append(f"{(ratio > 0).mean():>7.1%}")
    print(f"{name:<26}{X.shape[1]:>6}{accuracy:>8.3f}{macro:>8.3f}"
          + "".join(cells) + "  "
          + "".join(f"{per_class[labels_index[n]]:>7.3f}" for n in NASAL), flush=True)
    return model, accuracy, macro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--wav2vec", type=Path,
                        default=PROJECT_ROOT / "models" / "w2v_layer16.npz")
    parser.add_argument("--hand", nargs="+", default=["models/slr86_bw.npz",
                                                      "models/studio_bw.npz"])
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "vowel_classifier.joblib")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    w = np.load(args.wav2vec, allow_pickle=True)
    layer = next(k for k in w.files if k.startswith("layer_"))
    Xw = w[layer].astype(np.float32)
    lw, sw = w["labels"].astype(str), w["speakers"].astype(str)
    print(f"wav2vec  : {Xw.shape[0]:,} vowels x {Xw.shape[1]} dims ({layer})")

    Xh, lh, sh = load_hand(args.hand)
    print(f"hand     : {Xh.shape[0]:,} vowels x {Xh.shape[1]} dims")

    # The two caches were produced by separate passes and can differ by a few
    # utterances (the wav2vec pass skips clips under 0.1 s). Align them on the
    # (speaker, label, position-within-speaker) triple so the concatenated arm
    # is row-for-row honest rather than approximately aligned.
    def keyed(labels, speakers):
        seen: Counter = Counter()
        keys = []
        for label, speaker in zip(labels, speakers):
            seen[(speaker, label)] += 1
            keys.append((speaker, label, seen[(speaker, label)]))
        return keys

    kw, kh = keyed(lw, sw), keyed(lh, sh)
    position_h = {key: i for i, key in enumerate(kh)}
    pairs = [(i, position_h[key]) for i, key in enumerate(kw) if key in position_h]
    print(f"matched  : {len(pairs):,} rows present in both\n")
    wi = np.array([p[0] for p in pairs]); hi = np.array([p[1] for p in pairs])

    labels, speakers = lw[wi], sw[wi]
    label_set = sorted(set(labels))
    index = {l: i for i, l in enumerate(label_set)}
    y = np.array([index[l] for l in labels])
    hold = np.fromiter((is_studio(s) and held_out(s) for s in speakers),
                       bool, len(speakers))
    train = ~hold
    print(f"held out {hold.sum():,} vowels from "
          f"{len({s for s in speakers[hold]})} studio speakers")
    print(f"sources  : {dict(Counter(s.split('_')[0] for s in set(speakers[hold])))}\n")

    print(f"{'features':<26}{'dims':>6}{'acc':>8}{'macro':>8}"
          + "".join(f"{a + '~' + b:>7}" for a, b in CONTRASTS)
          + "  " + "".join(f"{n:>7}" for n in NASAL))

    arms = {
        "39 hand-designed": Xh[hi],
        "wav2vec2 layer 16": Xw[wi],
        "both concatenated": np.hstack([Xw[wi], Xh[hi]]),
    }
    results = {}
    for name, X in arms.items():
        results[name] = report(name, X, y, index, label_set, train, hold)

    # Prefer wav2vec2 alone unless concatenation earns its keep. Measured, the
    # two are 0.811 against 0.812 macro F1 -- noise. Keeping the 39 hand-designed
    # features would mean running two Praat analyses on every assessment forever
    # to buy nothing, so the tie goes to the simpler inference path.
    MEANINGFUL = 0.01
    solo, both = "wav2vec2 layer 16", "both concatenated"
    if both in results and solo in results:
        best = both if results[both][2] - results[solo][2] > MEANINGFUL else solo
        print(f"\nconcatenation gains {results[both][2] - results[solo][2]:+.4f} macro F1 "
              f"over wav2vec2 alone (threshold {MEANINGFUL}) -> keeping '{best}'")
    else:
        best = max(results, key=lambda k: results[k][2])
        print(f"\nbest by macro F1: {best}")
    if args.dry_run:
        print("--dry-run: nothing saved")
        return 0

    X = arms[best]
    final = build_estimator(balanced=True).fit(X, y)
    VowelClassifier(model=final, labels=label_set, metadata={
        "features_kind": best,
        "feature_dims": int(X.shape[1]),
        "wav2vec_model": "LyngualLabs/mms-300m-yoruba-english",
        "wav2vec_layer": int(layer.split("_")[1]),
        "training_vowels": int(len(y)),
        "speakers": len(set(speakers)),
        "holdout_accuracy": round(float(results[best][1]), 4),
        "macro_f1": round(float(results[best][2]), 4),
        "holdout_vowels": int(hold.sum()),
        "holdout_domain": "studio only (SLR86 + IroyinSpeech)",
        "class_balanced": True,
        "cepstral_normalization": "none",
        "audio_sample_rate": 16000,
    }).save(args.out)
    print(f"saved {args.out}")
    print("\nNOTE: verifier.py still builds 39-dim rows from Praat. It must be "
          "taught to\nrun wav2vec2 before this classifier can be used at "
          "inference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
