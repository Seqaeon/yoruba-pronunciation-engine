"""Train the vowel-identity classifier from aligned corpus audio.

This is what lets the engine catch *substitutions* -- a learner saying ``òkan``
when the prompt was ``òkun``. Forced alignment cannot detect that by itself: the
word has the right shape and duration, so it aligns without complaint.

Labels come free from the corpus alignment: each vowel segment already carries
the phone the speaker intended.

Evaluation splits by **speaker**, because formants scale with vocal tract length
and features are normalized per speaker. A random split would let the model
memorize individual voices and report an accuracy it cannot reproduce on a
learner it has never heard.

Usage::

    python scripts/train_vowel_classifier.py
    python scripts/train_vowel_classifier.py --limit 500 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.audit.dataset_audit import DatasetAuditor  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.logging_utils import configure_logging, get_logger  # noqa: E402
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402
from yoruba_engine.segmental.classifier import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    FORMANT_FEATURE_NAMES,
    VowelClassifier,
    build_estimator,
    combined_feature_names,
    vowel_features,
)
from yoruba_engine.segmental.formants import FormantExtractor, normalize_vowel_space  # noqa: E402
from yoruba_engine.segmental.mfcc import (  # noqa: E402
    MFCCExtractor,
    cepstral_mean_variance,
    normalize_vector,
)

logger = get_logger("train_vowel")

#: Standard Yoruba has 12 vowels: 7 oral (a e ẹ i o ọ u) and 5 nasal
#: (an ẹn in ọn un). They are modelled as separate classes by default, because
#: nasality is contrastive -- collapsing them would make the classifier unable
#: to tell ``ku`` from ``kun``.
#:
#: ``--merge-nasals`` folds each nasal onto its oral counterpart, which trains
#: more reliably on a small corpus at the cost of that contrast. Note that
#: ``en`` and ``on`` are not Yoruba vowels; they appear only in alignments
#: produced before the grapheme-to-phoneme rules were corrected.
NASAL_TO_ORAL = {"an": "a", "En": "E", "in": "i", "On": "O", "un": "u"}

#: ``en`` and ``on`` are not Yoruba vowels. They appear only in alignments
#: produced before the grapheme-to-phoneme rules were corrected, and are always
#: folded onto their oral counterparts -- keeping them would create classes with
#: a handful of examples that the classifier can only learn to ignore.
ARTEFACT_TO_ORAL = {"en": "e", "on": "o"}


def resolve_audio(textgrid: Path, by_id: dict, audio_roots: list[Path],
                  audio_index: dict[str, Path]) -> tuple[Path, str] | None:
    """Find the recording a TextGrid came from, and who spoke it.

    Two layouts are supported: the project corpus, indexed by utterance id, and
    a plain MFA corpus where audio sits in a per-speaker directory. Mined
    TextGrids are collected into one flat folder, so the speaker cannot be read
    from the TextGrid's own path -- it has to come from where the *audio* lives.
    That matters because the cepstral normalization is per speaker; treating
    thousands of clips as one speaker would normalize away the very differences
    the classifier needs.
    """
    utterance = by_id.get(textgrid.stem)
    if utterance is not None:
        return utterance.audio_path, utterance.speaker_id

    found = audio_index.get(textgrid.stem)
    if found is not None:
        return found, found.parent.name

    for root in audio_roots:
        candidate = root / textgrid.parent.name / f"{textgrid.stem}.wav"
        if candidate.exists():
            return candidate, candidate.parent.name
    return None


def build_audio_index(audio_roots: list[Path]) -> dict[str, Path]:
    """Map utterance id to audio path across the given roots, once."""
    index: dict[str, Path] = {}
    for root in audio_roots:
        for wav in root.rglob("*.wav"):
            index.setdefault(wav.stem, wav)
    if index:
        logger.info("Indexed %d audio files across %d roots", len(index), len(audio_roots))
    return index


def collect(config: EngineConfig, aligned_dirs: list[Path], limit: int | None,
            merge_nasals: bool, use_mfcc: bool = True,
            audio_roots: list[Path] | None = None):
    """Extract features and vowel labels from the aligned corpus.

    With ``use_mfcc`` the formant features are joined by cepstral coefficients,
    which is what makes nasality visible: formants describe tongue position and
    are nearly identical for a vowel and its nasal counterpart, so a
    formant-only model cannot represent the ``an``/``a`` contrast at all.

    MFCCs are normalized per speaker (mean subtracted, variance scaled) before
    being returned, because the low-order coefficients otherwise encode vocal
    tract length and channel response rather than which vowel was spoken.
    """
    report = DatasetAuditor(config).audit()
    by_id = {u.utterance_id: u for u in report.matched}

    textgrids: list[Path] = []
    for directory in aligned_dirs:
        found = sorted(directory.rglob("*.TextGrid"))
        logger.info("%s: %d TextGrids", directory, len(found))
        textgrids.extend(found)
    if not textgrids:
        raise SystemExit(f"No TextGrids in {aligned_dirs}. Train the acoustic model first.")
    audio_roots = audio_roots or []
    audio_index = build_audio_index(audio_roots)

    random.Random(0).shuffle(textgrids)
    if limit:
        textgrids = textgrids[:limit]

    extractor = FormantExtractor()
    mfcc_extractor = MFCCExtractor() if use_mfcc else None

    rows: list[list[float]] = []
    labels: list[str] = []
    speakers: list[str] = []
    skipped = 0

    # MFCC normalization is per speaker, so rows are grouped until every
    # utterance of a speaker has been read.
    pending_by_speaker: dict[str, list[tuple[list[float], list[float], str]]] = {}

    for index, textgrid in enumerate(textgrids, start=1):
        resolved = resolve_audio(textgrid, by_id, audio_roots, audio_index)
        if resolved is None:
            skipped += 1
            continue
        audio_path, speaker_id = resolved

        try:
            alignment = alignment_from_textgrid(textgrid)
            formants = extractor.analyse(audio_path)
        except Exception:  # noqa: BLE001 - one bad file must not stop training
            skipped += 1
            continue

        if mfcc_extractor is not None:
            try:
                mfcc_times, mfcc_frames = mfcc_extractor.analyse(audio_path)
            except Exception:
                skipped += 1
                continue

        vowels = [p for p in alignment.phones if is_vowel_phone(p.label)]
        measurements = []
        for phone in vowels:
            formant_values = extractor.measure_segment(formants, phone.start_s, phone.end_s)
            cepstral_values = (
                mfcc_extractor.measure_segment(mfcc_times, mfcc_frames,
                                               phone.start_s, phone.end_s)
                if mfcc_extractor is not None else []
            )
            measurements.append((phone.label, formant_values, cepstral_values))

        usable = [f for _, f, _ in measurements if f]
        if not usable:
            continue
        speaker_f1, speaker_f2 = normalize_vowel_space(usable)

        for label, formant_values, cepstral_values in measurements:
            if not formant_values:
                continue
            if mfcc_extractor is not None and not cepstral_values:
                continue
            # Strip any tone suffix: this classifier is about vowel identity,
            # which is orthogonal to tone.
            base = label.split("_")[0]
            base = ARTEFACT_TO_ORAL.get(base, base)
            if merge_nasals:
                base = NASAL_TO_ORAL.get(base, base)

            formant_row = vowel_features(formant_values, speaker_f1, speaker_f2)
            pending_by_speaker.setdefault(speaker_id, []).append(
                (formant_row, list(cepstral_values), base)
            )

        if index % 250 == 0:
            logger.info("Processed %d/%d utterances, %d vowels", index, len(textgrids), len(rows))

    # Normalize each speaker's cepstral features against their own statistics,
    # then flatten into the training matrix.
    for speaker_id, entries in pending_by_speaker.items():
        if mfcc_extractor is not None:
            means, deviations = cepstral_mean_variance([c for _, c, _ in entries])
        for formant_row, cepstral_row, base in entries:
            if mfcc_extractor is not None:
                cepstral_row = normalize_vector(cepstral_row, means, deviations)
            rows.append(formant_row + cepstral_row)
            labels.append(base)
            speakers.append(speaker_id)

    logger.info("Collected %d vowels from %d utterances (%d skipped), %d features each",
                len(rows), len(textgrids), skipped, len(rows[0]) if rows else 0)
    return rows, labels, speakers


def speaker_split(speakers: list[str], holdout_fraction: float, seed: int = 0):
    """Split indices by speaker so no voice appears on both sides."""
    unique = sorted(set(speakers))
    random.Random(seed).shuffle(unique)
    holdout = set(unique[: max(1, int(len(unique) * holdout_fraction))])
    train = [i for i, s in enumerate(speakers) if s not in holdout]
    test = [i for i, s in enumerate(speakers) if s in holdout]
    return train, test, holdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, action="append",
                        help="Directory of TextGrids; repeat to combine sources")
    parser.add_argument("--audio-root", type=Path, action="append",
                        help="Where to find audio for TextGrids not in the corpus index")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, help="Use only the first N utterances")
    parser.add_argument("--holdout", type=float, default=0.25)
    parser.add_argument("--merge-nasals", action="store_true",
                        help="Fold nasal vowels onto their oral counterparts "
                             "(loses the nasal contrast; only for small corpora)")
    parser.add_argument("--balanced", action="store_true",
                        help="Weight classes inversely to frequency, so the rare "
                             "nasal vowels are not simply ignored")
    parser.add_argument("--no-mfcc", action="store_true",
                        help="Formants only, for comparison against the MFCC model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        for noisy in ("yoruba_engine.audit", "yoruba_engine.formants"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    config = EngineConfig()
    aligned_dirs = args.aligned or [PROJECT_ROOT / "output" / "aligned"]
    rows, labels, speakers = collect(
        config, aligned_dirs, args.limit, merge_nasals=args.merge_nasals,
        use_mfcc=not args.no_mfcc, audio_roots=args.audio_root or [],
    )
    if len(rows) < 500:
        raise SystemExit(f"Only {len(rows)} vowels collected; too few to train.")

    label_set = sorted(set(labels))
    index_of = {label: i for i, label in enumerate(label_set)}

    X = np.asarray(rows, dtype=float)
    y = np.asarray([index_of[label] for label in labels], dtype=int)

    train_idx, test_idx, holdout_speakers = speaker_split(speakers, args.holdout)
    logger.info("Split by speaker: %d train / %d held-out vowels across %d speakers",
                len(train_idx), len(test_idx), len(holdout_speakers))

    counts = Counter(labels)
    majority = max(counts.values()) / len(labels)
    print("\nClass balance: " + ", ".join(
        f"{label}={counts[label]}" for label in label_set))
    print(f"Majority-class baseline: {majority * 100:.1f}%")

    model = build_estimator(balanced=args.balanced)
    model.fit(X[train_idx], y[train_idx])
    predicted = model.predict(X[test_idx])
    accuracy = accuracy_score(y[test_idx], predicted)

    print(f"\nHeld-out vowel identity accuracy: {accuracy * 100:.1f}%\n")
    print(classification_report(
        y[test_idx], predicted, labels=list(range(len(label_set))),
        target_names=label_set, digits=3, zero_division=0,
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("      " + "".join(f"{label:>6}" for label in label_set))
    for row_index, row in enumerate(
        confusion_matrix(y[test_idx], predicted, labels=list(range(len(label_set))))
    ):
        print(f"{label_set[row_index]:>5} " + "".join(f"{value:>6}" for value in row))

    if args.dry_run:
        print("\nDry run; model not saved.")
        return 0

    final = build_estimator(balanced=args.balanced)
    final.fit(X, y)
    classifier = VowelClassifier(
        model=final,
        labels=label_set,
        metadata={
            "training_vowels": int(len(y)),
            "speakers": len(set(speakers)),
            "holdout_accuracy": round(float(accuracy), 4),
            "majority_baseline": round(float(majority), 4),
            "nasals_merged": args.merge_nasals,
            "features": list(
                FORMANT_FEATURE_NAMES if args.no_mfcc else combined_feature_names()
            ),
            "uses_mfcc": not args.no_mfcc,
            "class_balanced": args.balanced,
        },
    )
    path = classifier.save(args.out)
    print(f"\nSaved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
