"""Train the tone classifier from aligned corpus audio.

Ground truth comes free from Yoruba orthography: every diacritized transcript is
a labelled example, so no hand-annotation is required. The script reads the
TextGrids produced by acoustic model training, extracts pitch features, and fits
a classifier against the tones written in the text.

Prerequisites:
    1. The acoustic model has been trained (notebooks/train_mfa_yoruba.ipynb),
       leaving TextGrids in output/aligned/.
    2. scikit-learn and joblib are installed.

Usage::

    python scripts/train_tone_classifier.py
    python scripts/train_tone_classifier.py --kind logreg --limit 1000
    python scripts/train_tone_classifier.py --dry-run     # evaluate, save nothing

**On honest evaluation.** The split is by *speaker*, not by utterance. Splitting
randomly would put the same speaker on both sides, and because features are
normalized per speaker the model would partly memorize individual voices --
inflating the score against learners it has never heard.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import (
    alignment_from_textgrid,
    mark_syllabic_nasals,
)  # noqa: E402
from yoruba_engine.audit.dataset_audit import DatasetAuditor  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.datatypes import Tone  # noqa: E402
from yoruba_engine.g2p.orthography import is_noise_token, tone_pattern  # noqa: E402
from yoruba_engine.logging_utils import configure_logging, get_logger  # noqa: E402
from yoruba_engine.pitch.extractor import PitchExtractor  # noqa: E402
from yoruba_engine.tone.analyzer import ToneAnalyzer  # noqa: E402
from yoruba_engine.tone.classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    TONE_ORDER,
    TONE_TO_INDEX,
    ToneClassifier,
    build_estimator,
    check_feature_count,
)
from yoruba_engine.tone.features import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    measurable,
)

logger = get_logger("train_tone")


def collect(config: EngineConfig, aligned_dir: Path, limit: int | None,
            corpus_dir: Path | None = None):
    """Build the training set from aligned utterances.

    Returns:
        ``(rows, labels, speakers)`` -- one entry per labelled vowel.

    Only vowels whose word's written tone pattern has the same length as its
    aligned vowel count are used. A mismatch means the aligner and the
    orthography disagree about syllable count, so the label cannot be trusted to
    belong to that vowel.
    """
    if corpus_dir is not None:
        # An MFA-format corpus: paired .wav and .lab, which is what the 118 h
        # training corpus is. The auditor reads config.corpus_dir instead, which
        # is SLR86 -- 36 speakers -- so this classifier was fitted on 49,531
        # vowels from 36 voices while the acoustic model and vowel classifier
        # were retrained on 2,801. Pointing --aligned at the larger set was not
        # enough on its own: nothing matched, because the transcripts came from
        # somewhere else.
        from types import SimpleNamespace  # noqa: PLC0415

        by_id = {}
        for lab in corpus_dir.rglob("*.lab"):
            wav = lab.with_suffix(".wav")
            if not wav.exists():
                continue
            text = lab.read_text(encoding="utf-8").strip()
            if text:
                # The speaker is the directory, which is how the corpus is laid
                # out and what the held-out split must be drawn on: this
                # classifier is asked to read a voice it has never heard.
                by_id[lab.stem] = SimpleNamespace(
                    utterance_id=lab.stem, audio_path=wav, text=text,
                    speaker_id=lab.parent.name)
        logger.info("Read %d transcripts from %s", len(by_id), corpus_dir)
    else:
        report = DatasetAuditor(config).audit()
        by_id = {utterance.utterance_id: utterance for utterance in report.matched}

    textgrids = sorted(aligned_dir.rglob("*.TextGrid"))
    if not textgrids:
        raise SystemExit(
            f"No TextGrids in {aligned_dir}. Train the acoustic model first "
            "(notebooks/train_mfa_yoruba.ipynb)."
        )

    random.Random(0).shuffle(textgrids)
    if limit:
        textgrids = textgrids[:limit]

    extractor = PitchExtractor(config.pitch)
    # Declination removal and three-point sampling must happen before feature
    # extraction, exactly as at inference time.
    analyzer = ToneAnalyzer(config.tone)

    rows: list[list[float]] = []
    labels: list[int] = []
    speakers: list[str] = []
    skipped = mismatched = 0

    for index, textgrid in enumerate(textgrids, start=1):
        utterance = by_id.get(textgrid.stem)
        if utterance is None:
            skipped += 1
            continue

        try:
            alignment = alignment_from_textgrid(textgrid, utterance_id=utterance.utterance_id)
            # Syllabic nasals carry tone. Without this they read as consonants,
            # and every word containing one is dropped for a syllable-count
            # mismatch -- which would make this an unfair baseline for the
            # sequence model as much as a weaker classifier.
            mark_syllabic_nasals(alignment, utterance.text)
            words, _ = extractor.extract(utterance.audio_path, alignment)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop training
            logger.debug("Skipping %s: %s", textgrid.name, exc)
            skipped += 1
            continue

        analyzer.analyse(words)

        gold: dict[int, Tone] = {}
        position = 0
        for word in words:
            if is_noise_token(word.word):
                continue
            written = tone_pattern(word.word)
            if len(written) != len(word.vowels):
                mismatched += 1
                position += len(word.vowels)
                continue
            for offset, tone in enumerate(written):
                gold[position + offset] = tone
            position += len(word.vowels)

        vowels = [vowel for word in words for vowel in word.vowels]
        measured = measurable(vowels)
        if not measured:
            continue

        features = extract_features(vowels)
        check_feature_count(features)

        # extract_features emits rows for measurable vowels only, so map back
        # through their index in the full vowel list to find the gold label.
        measured_positions = [i for i, v in enumerate(vowels) if v.mean_semitones is not None]
        for row, original_index in zip(features, measured_positions):
            tone = gold.get(original_index)
            if tone is None or tone is Tone.UNKNOWN:
                continue
            rows.append(row)
            labels.append(TONE_TO_INDEX[tone])
            speakers.append(utterance.speaker_id)

        if index % 250 == 0:
            logger.info("Processed %d/%d utterances, %d vowels", index, len(textgrids), len(rows))

    logger.info(
        "Collected %d labelled vowels from %d utterances (%d skipped, %d "
        "syllable-count mismatches)",
        len(rows), len(textgrids), skipped, mismatched,
    )
    return rows, labels, speakers


def speaker_split(speakers: list[str], holdout_fraction: float, seed: int = 0):
    """Split indices by speaker so no voice appears on both sides."""
    unique = sorted(set(speakers))
    random.Random(seed).shuffle(unique)
    holdout_count = max(1, int(len(unique) * holdout_fraction))
    holdout_speakers = set(unique[:holdout_count])

    train_idx = [i for i, s in enumerate(speakers) if s not in holdout_speakers]
    test_idx = [i for i, s in enumerate(speakers) if s in holdout_speakers]
    return train_idx, test_idx, holdout_speakers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--corpus-dir", type=Path,
                        help="MFA-format corpus of paired .wav/.lab to take "
                             "transcripts from, instead of the audited SLR86 "
                             "corpus. Use data/train_corpus with "
                             "--aligned output/train_aligned to fit on all "
                             "2,801 speakers rather than SLR86's 36.")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--kind", choices=("gbm", "logreg"), default="gbm")
    parser.add_argument("--limit", type=int, help="Use only the first N utterances")
    parser.add_argument("--holdout", type=float, default=0.25,
                        help="Fraction of speakers held out for evaluation")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without saving")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        # These log once per utterance, which buries the training output.
        for noisy in ("yoruba_engine.pitch", "yoruba_engine.tone", "yoruba_engine.audit"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    config = EngineConfig()
    rows, labels, speakers = collect(config, args.aligned, args.limit, args.corpus_dir)
    if len(rows) < 200:
        raise SystemExit(f"Only {len(rows)} labelled vowels; too few to train.")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)

    train_idx, test_idx, holdout_speakers = speaker_split(speakers, args.holdout)
    logger.info(
        "Split by speaker: %d train / %d held-out vowels across %d held-out speakers",
        len(train_idx), len(test_idx), len(holdout_speakers),
    )

    counts = Counter(y.tolist())
    majority = max(counts.values()) / len(y)
    print(f"\nClass balance: " + ", ".join(
        f"{TONE_ORDER[k].value}={v} ({v / len(y) * 100:.1f}%)" for k, v in sorted(counts.items())
    ))
    print(f"Majority-class baseline: {majority * 100:.1f}%")

    model = build_estimator(args.kind)
    model.fit(X[train_idx], y[train_idx])

    predicted = model.predict(X[test_idx])
    accuracy = accuracy_score(y[test_idx], predicted)

    print(f"\nHeld-out accuracy ({args.kind}): {accuracy * 100:.1f}%")
    print("\n" + classification_report(
        y[test_idx], predicted,
        labels=list(range(len(TONE_ORDER))),
        target_names=[t.value for t in TONE_ORDER],
        digits=3, zero_division=0,
    ))
    print("Confusion matrix (rows = expected, columns = predicted):")
    print("        " + "".join(f"{t.value:>8}" for t in TONE_ORDER))
    for row_index, row in enumerate(confusion_matrix(
        y[test_idx], predicted, labels=list(range(len(TONE_ORDER)))
    )):
        print(f"{TONE_ORDER[row_index].value:>7} " + "".join(f"{v:>8}" for v in row))

    if args.dry_run:
        print("\nDry run; model not saved.")
        return 0

    # Permutation importance on the held-out speakers: how much accuracy is lost
    # when each feature is shuffled. HistGradientBoosting exposes no built-in
    # importances, and this measure answers the more useful question anyway.
    from sklearn.inspection import permutation_importance

    logger.info("Computing permutation importance on the held-out speakers")
    perm = permutation_importance(
        model, X[test_idx], y[test_idx], n_repeats=5, random_state=0, n_jobs=-1,
    )
    importances = sorted(
        zip(FEATURE_NAMES, perm.importances_mean), key=lambda pair: -pair[1]
    )

    print("\nPermutation importance (accuracy lost when shuffled):")
    for name, value in importances[:10]:
        print(f"  {name:<22} {value:+.4f}")

    # Refit on everything for the shipped model: the split existed to measure
    # honestly, and the deployed model should use all available data.
    final = build_estimator(args.kind)
    final.fit(X, y)

    classifier = ToneClassifier(
        model=final,
        metadata={
            "kind": args.kind,
            "training_vowels": int(len(y)),
            "speakers": len(set(speakers)),
            "holdout_accuracy": round(float(accuracy), 4),
            "holdout_speakers": sorted(holdout_speakers),
            "majority_baseline": round(float(majority), 4),
            "feature_importances": [(n, round(float(v), 5)) for n, v in importances],
        },
    )
    path = classifier.save(args.out)

    print(f"\nSaved to {path}")
    print("Use it with:  ToneAnalyzer(classifier=ToneClassifier.load())")
    return 0


if __name__ == "__main__":
    sys.exit(main())
