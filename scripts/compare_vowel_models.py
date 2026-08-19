"""Score two vowel classifiers on one identical held-out set.

Comparing a new model's held-out figure against a number recorded in an old
manifest compares two different held-out sets as much as two models: the corpus
grew, the speaker prefixes changed, and the hash that picks the held-out quarter
sees a different population. The only honest comparison runs both models over
the same rows.

Both consume wav2vec2 layer 16 at 1024 dimensions, so they are directly
interchangeable here; what differs is the alignment that decided where each
vowel began and ended.

**A shipped classifier is refit on all data, held-out rows included**, because
that is the model worth shipping. Scoring one on those rows measures how well it
memorised them. Doing exactly that once reported the new model at 0.935 against
the old at 0.897, when the honest split-fit figures were 0.895 and 0.873 -- a
gap of 0.04 invented entirely by the contamination. Any model whose metadata
says ``refit_on_all_data`` is therefore reported with a warning, and rows from
speakers it lists in ``holdout_speakers`` are the only ones it may be scored on
at all.

    python scripts/compare_vowel_models.py models/w2v_slices/*.npz \\
        --models models/vowel_classifier.v1.joblib models/vowel_classifier.joblib
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.segmental.classifier import VowelClassifier  # noqa: E402

STUDIO_PREFIXES = ("slr86__", "studio__iroyin_")
CONTRASTS = [("un", "an"), ("an", "un"), ("o", "O"), ("O", "o"), ("e", "E"), ("E", "e")]
NASAL = ["an", "En", "in", "On", "un"]


def is_studio(speaker: str) -> bool:
    return speaker.startswith(STUDIO_PREFIXES) and "_unk" not in speaker


def held_out(speaker: str, percent: int = 25) -> bool:
    return int(hashlib.md5(speaker.encode("utf-8")).hexdigest()[:8], 16) % 100 < percent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    rows, labels, speakers = [], [], []
    for path in sorted(args.caches):
        data = np.load(path, allow_pickle=True)
        layer = next(k for k in data.files if k.startswith("layer_"))
        rows.append(data[layer].astype(np.float32))
        labels.append(data["labels"].astype(str))
        speakers.append(data["speakers"].astype(str))
    X = np.vstack(rows)
    labels = np.concatenate(labels)
    speakers = np.concatenate(speakers)

    hold = np.fromiter((is_studio(s) and held_out(s) for s in speakers),
                       dtype=bool, count=len(speakers))
    Xh, yh = X[hold], labels[hold]
    print(f"held out: {hold.sum():,} vowels from "
          f"{len(set(speakers[hold])):,} studio speakers\n")

    from sklearn.metrics import accuracy_score, f1_score

    header = f"{'model':<34}{'acc':>7}{'macroF1':>9}"
    for a, b in CONTRASTS:
        header += f"{a + '/' + b:>9}"
    print(header)

    for path in args.models:
        classifier = VowelClassifier.load(path)
        meta = classifier.metadata
        if meta.get("refit_on_all_data") and not meta.get("holdout_speakers"):
            print(f"  !! {path.name} was refit on all data and does not record "
                  "its held-out speakers.\n     Its score below is training "
                  "accuracy, not generalisation. Use its manifest figure.")
        elif meta.get("refit_on_all_data"):
            print(f"  note: {path.name} was refit on all data; scoring only the "
                  f"{len(meta['holdout_speakers'])} speakers it held out during "
                  "evaluation.")
        names = list(classifier.labels)
        position = {name: index for index, name in enumerate(names)}
        keep = np.fromiter((name in position for name in yh), dtype=bool,
                           count=len(yh))
        truth = np.fromiter((position[name] for name in yh[keep]),
                            dtype=np.int64, count=int(keep.sum()))
        proba = classifier.model.predict_proba(Xh[keep])
        predicted = proba.argmax(axis=1)

        line = (f"{path.name:<34}{accuracy_score(truth, predicted):>7.3f}"
                f"{f1_score(truth, predicted, average='macro', zero_division=0):>9.3f}")
        for expected, competitor in CONTRASTS:
            if expected not in position or competitor not in position:
                line += f"{'-':>9}"
                continue
            mask = truth == position[expected]
            if not mask.any():
                line += f"{'-':>9}"
                continue
            ratio = np.log(np.clip(proba[mask][:, position[expected]], 1e-9, 1) /
                           np.clip(proba[mask][:, position[competitor]], 1e-9, 1))
            line += f"{(ratio > 0).mean():>8.1%} "
        print(line)

        per_class = f1_score(truth, predicted, average=None,
                             labels=list(range(len(names))), zero_division=0)
        print("      nasals  " + "  ".join(
            f"{n}={per_class[position[n]]:.3f}" for n in NASAL if n in position))
    return 0


if __name__ == "__main__":
    sys.exit(main())
