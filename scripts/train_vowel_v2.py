"""Retrain the vowel classifier on the v2 acoustic model's alignments.

The features are the same as before -- wav2vec2 layer 16, 1024 dimensions, no
normalization -- so this is not an attempt to improve the representation. What
changed is underneath: every vowel span comes from an acoustic model trained on
118 hours rather than 37.93, which cut degenerate sub-40 ms vowels on SLR86 from
15.1% to 5.3%. A vowel measured over the wrong 30 ms is a mislabelled training
row, so better boundaries should show up as a better classifier from identical
inputs.

**Speaker names changed and the held-out rule had to follow.** The corpus is now
assembled by ``build_training_corpus.py``, which prefixes each speaker with its
source (``slr86__yof_00295``, ``studio__iroyin_...``) so speakers from different
corpora cannot collide. The old predicate matched ``openslr_``/``iroyin_`` and
would now match nothing, silently producing an empty held-out set and a
meaningless score. Prefixes here are the new ones.

**Evaluation stays studio-only**, for the reason the earlier script records:
pooling crowdsourced speakers into the held-out set reported the ``un``/``an``
contrast at 92.9% when the honest studio figure was 69.7%. Learners record in
studio-ish conditions, so that is the only number worth acting on.

**Hand-designed features are gone entirely.** Concatenating the 39 Praat
measurements onto wav2vec2 was measured at +0.0006 macro F1, which does not pay
for running two Praat analyses on every assessment forever.

    python scripts/train_vowel_v2.py models/w2v_slices/*.npz
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

from yoruba_engine.segmental.classifier import (  # noqa: E402
    VowelClassifier, build_estimator,
)

#: Sources eligible for the held-out set: fully diacritized, with real speaker
#: labels. Pseudo-speaker buckets (``_unk``) are excluded -- they pool unrelated
#: voices, so holding one out does not hold a speaker out.
#:
#: YECS is included, and that is a deliberate change. The held-out set was
#: previously SLR86 and IroyinSpeech only, on the reasoning that learners record
#: in studio-ish conditions. The cost was that ``ẹn`` barely occurs in either, so
#: the class had 26 held-out examples and its F1 carried no information -- it
#: could be trained on 2,937 examples but never scored. A class you cannot
#: measure is a class you cannot tell you have improved.
#:
#: YECS is 99.5% consistent with the corpus spellings and, being recorded
#: through a device-agnostic web app, is arguably closer to what a learner's
#: laptop produces than SLR86's careful recordings are.
HOLDOUT_PREFIXES = ("slr86__", "studio__iroyin_", "studio__commonvoice_", "yecs__")

#: The narrower definition used before, kept so the headline figure stays
#: comparable across versions rather than silently changing meaning.
STUDIO_PREFIXES = ("slr86__", "studio__iroyin_")
NASAL = ["an", "En", "in", "On", "un"]
#: Contrasts that decide real words: òkun/ọkàn turns on the first pair,
#: òkun/ọkun on the vowel-quality pairs.
CONTRASTS = [("un", "an"), ("an", "un"), ("o", "O"), ("O", "o"), ("e", "E"), ("E", "e")]


def is_studio(speaker: str) -> bool:
    return speaker.startswith(STUDIO_PREFIXES) and "_unk" not in speaker


def is_eligible(speaker: str) -> bool:
    return speaker.startswith(HOLDOUT_PREFIXES) and "_unk" not in speaker


def source_of(speaker: str) -> str:
    """The corpus a speaker came from, for a per-source breakdown."""
    if speaker.startswith("slr86__"):
        return "slr86"
    if speaker.startswith("yecs__"):
        return "yecs"
    for name in ("iroyin", "commonvoice", "ljspeech", "thisniyi", "waxal",
                 "michsethowusu"):
        if speaker.startswith(f"studio__{name}"):
            return name
    return "other"


def held_out(speaker: str, percent: int = 25) -> bool:
    """Stable per speaker, so runs stay comparable as the corpus grows."""
    return int(hashlib.md5(speaker.encode("utf-8")).hexdigest()[:8], 16) % 100 < percent


def load(paths: list[Path]):
    rows, labels, speakers = [], [], []
    for path in sorted(paths):
        data = np.load(path, allow_pickle=True)
        layer = next(k for k in data.files if k.startswith("layer_"))
        rows.append(data[layer].astype(np.float32))
        labels.append(data["labels"].astype(str))
        speakers.append(data["speakers"].astype(str))
        print(f"  {path.name}: {len(labels[-1]):,} vowels", flush=True)
    return (np.vstack(rows), np.concatenate(labels), np.concatenate(speakers),
            layer)


def report(y_true, predicted, proba, label_set, position) -> tuple[float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_true, predicted)
    macro = f1_score(y_true, predicted, average="macro", zero_division=0)
    per_class = f1_score(y_true, predicted, average=None,
                         labels=list(range(len(label_set))), zero_division=0)

    print(f"\naccuracy {accuracy:.3f}   macro F1 {macro:.3f}\n")
    print("minimal-pair contrasts (share where the right vowel outscores its rival)")
    for expected, competitor in CONTRASTS:
        mask = y_true == position[expected]
        if not mask.any():
            print(f"  {expected} vs {competitor}: no held-out examples")
            continue
        ratio = np.log(np.clip(proba[mask][:, position[expected]], 1e-9, 1) /
                       np.clip(proba[mask][:, position[competitor]], 1e-9, 1))
        print(f"  {expected:>3} vs {competitor:<3} {(ratio > 0).mean():>7.1%}"
              f"   (n={int(mask.sum()):,})")

    print("\nper class")
    counts = Counter(y_true.tolist())
    for name in label_set:
        index = position[name]
        print(f"  {name:<4}{per_class[index]:>7.3f}   support {counts.get(index, 0):>7,}")
    return float(accuracy), float(macro)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "vowel_classifier.joblib")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"loading {len(args.caches)} cache file(s)")
    X, labels, speakers, layer = load(args.caches)
    print(f"\n{X.shape[0]:,} vowels x {X.shape[1]} dims from "
          f"{len(set(speakers)):,} speakers ({layer})")

    label_set = sorted(set(labels))
    position = {name: index for index, name in enumerate(label_set)}
    y = np.fromiter((position[name] for name in labels), dtype=np.int64,
                    count=len(labels))

    hold = np.fromiter((is_eligible(s) and held_out(s) for s in speakers),
                       dtype=bool, count=len(speakers))
    studio_hold = np.fromiter((is_studio(s) and held_out(s) for s in speakers),
                              dtype=bool, count=len(speakers))
    train = ~hold
    print(f"held out: {hold.sum():,} vowels from "
          f"{len(set(speakers[hold])):,} speakers")
    by_source = Counter(source_of(s) for s in speakers[hold])
    print("   " + "  ".join(f"{k}={v:,}" for k, v in by_source.most_common()))
    if not hold.any():
        print("\nEMPTY HELD-OUT SET -- the speaker prefixes do not match "
              "HOLDOUT_PREFIXES.\nSpeakers look like: "
              + ", ".join(sorted(set(speakers))[:3]))
        return 1

    print("\nfitting on the training split...", flush=True)
    model = build_estimator(balanced=True).fit(X[train], y[train])
    proba = model.predict_proba(X[hold])
    predicted = proba.argmax(axis=1)
    accuracy, macro = report(y[hold], predicted, proba, label_set, position)

    # The narrower studio-only figure, so the headline stays comparable with
    # versions measured before YECS speakers were eligible.
    inner = studio_hold[hold]
    if inner.any():
        from sklearn.metrics import accuracy_score, f1_score
        s_acc = accuracy_score(y[hold][inner], predicted[inner])
        s_macro = f1_score(y[hold][inner], predicted[inner], average="macro",
                           zero_division=0)
        print(f"\nsame model on the old studio-only subset "
              f"({int(inner.sum()):,} vowels): "
              f"accuracy {s_acc:.3f}, macro F1 {s_macro:.3f}")
    else:
        s_acc = s_macro = float("nan")

    if args.dry_run:
        print("\n--dry-run: nothing saved")
        return 0

    print("\nrefitting on everything...", flush=True)
    final = build_estimator(balanced=True).fit(X, y)
    VowelClassifier(model=final, labels=label_set, metadata={
        "features_kind": "wav2vec2 layer 16",
        "feature_dims": int(X.shape[1]),
        "wav2vec_model": "LyngualLabs/mms-300m-yoruba-english",
        "wav2vec_layer": int(layer.split("_")[1]),
        "training_vowels": int(len(y)),
        "speakers": len(set(speakers)),
        "holdout_accuracy": round(accuracy, 4),
        "macro_f1": round(macro, 4),
        "holdout_vowels": int(hold.sum()),
        "holdout_domain": "SLR86 + IroyinSpeech + Common Voice + YECS",
        "holdout_studio_only_accuracy": round(float(s_acc), 4),
        "holdout_studio_only_macro_f1": round(float(s_macro), 4),
        "acoustic_model": "yoruba_acoustic_v2 (118 h, 2,801 speakers)",
        # The shipped model is refit on everything, held-out rows included,
        # because that is the model worth shipping. Scoring it on its own
        # held-out set afterwards therefore measures training accuracy, not
        # generalisation. Recording the speakers makes that detectable instead
        # of leaving a later comparison to produce a flattering number in
        # silence -- which it did once.
        "refit_on_all_data": True,
        "holdout_speakers": sorted({s for s in speakers[hold]}),
        "class_balanced": True,
        "cepstral_normalization": "none",
        "audio_sample_rate": 16000,
    }).save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
