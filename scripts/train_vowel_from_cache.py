"""Retrain the vowel classifier from cached measurements.

Takes one or more caches written by ``extract_vowel_features.py`` (or by the
Kaggle notebook, which writes the same arrays) and fits the classifier without
touching audio again.

**Evaluation is studio-only, and that is the point.** Pooling studio and
crowdsourced speakers in the held-out set is what previously reported the
``un``/``an`` contrast at 92.9% when the honest studio figure was 69.7%: the
crowdsourced held-out speakers are the same domain as the bulk of training, so
they flatter every number. Learners record in studio-ish conditions, so the only
figure worth acting on is the one measured on fully diacritized studio speech.

**No normalization anywhere.** Whatever is done here must be reproducible on a
single learner utterance, and neither per-utterance normalization (destructive
at 2-5 vowels) nor a corpus-wide constant (has to ship with the model, and
silently invalidates it when it does not) qualifies.

    python scripts/train_vowel_from_cache.py \\
        kaggle_model/vowel_features.npz models/studio_features.npz
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
    VowelClassifier, build_estimator, combined_feature_names,
)

#: Speaker-directory prefixes whose transcripts are fully diacritized AND whose
#: audio is studio-quality. These, and only these, are eligible for the
#: held-out set.
STUDIO_PREFIXES = ("openslr_", "iroyin_")
NASAL = ["an", "En", "in", "On", "un"]
#: Contrasts that decide real words: òkun/ọkàn turns on the first, òkun/ọkun on
#: the second.
CONTRASTS = [("un", "an"), ("an", "un"), ("o", "O"), ("O", "o"), ("e", "E"), ("E", "e")]


def is_studio(speaker: str) -> bool:
    return speaker.startswith(STUDIO_PREFIXES) and "_unk" not in speaker


def in_holdout(speaker: str, percent: int = 25) -> bool:
    """Stable per speaker, so runs stay comparable as the corpus grows."""
    return int(hashlib.md5(speaker.encode("utf-8")).hexdigest()[:8], 16) % 100 < percent


def load(paths: list[Path]):
    formants, cepstra, labels, speakers = [], [], [], []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        formants.append(data["formants"].astype(np.float64))
        cepstra.append(data["cepstra"].astype(np.float64))
        labels.append(data["labels"].astype(str))
        speakers.append(data["speakers"].astype(str))
        print(f"  {path.name}: {len(labels[-1]):,} vowels, "
              f"{len(set(speakers[-1])):,} speakers")
    return (np.vstack(formants), np.vstack(cepstra),
            np.concatenate(labels), np.concatenate(speakers))


def report(name, y_true, predicted, proba, label_set, position):
    from sklearn.metrics import accuracy_score, f1_score

    accuracy = accuracy_score(y_true, predicted)
    macro = f1_score(y_true, predicted, average="macro", zero_division=0)
    per_class = f1_set = f1_score(y_true, predicted, average=None,
                                  labels=list(range(len(label_set))), zero_division=0)
    cells = []
    for expected, competitor in CONTRASTS:
        mask = y_true == position[expected]
        if not mask.any():
            cells.append("     -")
            continue
        ratio = np.log(np.clip(proba[mask][:, position[expected]], 1e-9, 1) /
                       np.clip(proba[mask][:, position[competitor]], 1e-9, 1))
        cells.append(f"{(ratio > 0).mean():>6.1%}")
    print(f"{name:<24}{accuracy:>7.3f}{macro:>8.3f}" + "".join(cells)
          + "  " + "".join(f"{per_class[position[n]]:>7.3f}" for n in NASAL))
    return accuracy, macro, f1_set


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "vowel_classifier.joblib")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading caches:")
    formants, cepstra, labels, speakers = load(args.caches)
    X = np.hstack([formants, cepstra])          # raw, no normalization
    label_set = sorted(set(labels))
    position = {label: i for i, label in enumerate(label_set)}
    y = np.array([position[l] for l in labels])

    studio = np.fromiter((is_studio(s) for s in speakers), bool, len(speakers))
    holdout = np.fromiter(
        (is_studio(s) and in_holdout(s) for s in speakers), bool, len(speakers))
    train = ~holdout
    print(f"\ntotal {len(y):,} vowels, {len(set(speakers)):,} speakers")
    print(f"studio {studio.sum():,} vowels from "
          f"{len({s for s in speakers if is_studio(s)}):,} speakers")
    print(f"held out {holdout.sum():,} vowels from "
          f"{len({s for s in speakers[holdout]}):,} studio speakers")
    print(f"held-out sources: {dict(Counter(s.split('_')[0] for s in set(speakers[holdout])))}")

    header = ("model", "acc", "macro") + tuple(f"{a}~{b}" for a, b in CONTRASTS)
    print(f"\n{'model':<24}{'acc':>7}{'macro':>8}"
          + "".join(f"{a + '~' + b:>6}" for a, b in CONTRASTS)
          + "  " + "".join(f"{n:>7}" for n in NASAL))

    # Baseline: the shipped classifier, on this same studio held-out set.
    try:
        shipped = VowelClassifier.load(args.out)
        if list(shipped.labels) == label_set:
            proba = shipped.model.predict_proba(X[holdout])
            report("shipped", y[holdout], shipped.model.predict(X[holdout]),
                   proba, label_set, position)
    except Exception as exc:
        print(f"(no comparable shipped model: {exc})")

    model = build_estimator(balanced=True).fit(X[train], y[train])
    proba = model.predict_proba(X[holdout])
    accuracy, macro, _ = report("retrained (this run)", y[holdout],
                                model.predict(X[holdout]), proba, label_set, position)

    print("\ncontrast columns are the share of held-out vowels on the correct side;"
          "\nthe last five are nasal F1")

    if args.dry_run:
        print("\n--dry-run: nothing saved")
        return 0

    final = build_estimator(balanced=True).fit(X, y)
    VowelClassifier(
        model=final, labels=label_set,
        metadata={
            "training_vowels": int(len(y)),
            "speakers": len(set(speakers)),
            "studio_vowels": int(studio.sum()),
            "holdout_accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro), 4),
            "holdout_vowels": int(holdout.sum()),
            "holdout_domain": "studio only (openslr + iroyin)",
            "class_balanced": True,
            "uses_mfcc": True,
            "cepstral_normalization": "none",
            "audio_sample_rate": 16000,
            "caches": [p.name for p in args.caches],
            "features": list(combined_feature_names()),
        },
    ).save(args.out)
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
