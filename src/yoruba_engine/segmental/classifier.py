"""Trained vowel-identity classifier.

Answers "which vowel did the learner actually produce?" from formant
measurements, so the engine can detect substitutions -- saying ``òkan`` when the
prompt was ``òkun``. Forced alignment cannot catch that on its own: the word has
the right number of syllables and a plausible duration, so it aligns happily.

As with tone, the training labels are free. The corpus is aligned against its own
transcripts, so every vowel segment already carries the phone label the speaker
intended, and the classifier learns the mapping from formants to that label.

Train with::

    python scripts/train_vowel_classifier.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..exceptions import ToneAnalysisError
from ..logging_utils import get_logger

logger = get_logger("segmental.classifier")

#: Where the training script writes the model.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "vowel_classifier.joblib"

#: Formant feature order produced by :func:`vowel_features`.
FORMANT_FEATURE_NAMES: tuple[str, ...] = (
    "f1", "f2", "f3", "f1_rel", "f2_rel", "f2_minus_f1", "f1_over_f2",
    "b1", "b2", "b3", "b1_over_f1", "b2_over_f2", "b1_over_b2",
)

#: Kept for backwards compatibility with models trained on formants alone.
FEATURE_NAMES = FORMANT_FEATURE_NAMES


def mfcc_feature_names(coefficients: int = 12) -> tuple[str, ...]:
    """Names for the MFCC block: mean then standard deviation of c0..cN."""
    return tuple(
        [f"mfcc{i}_mean" for i in range(coefficients + 1)]
        + [f"mfcc{i}_std" for i in range(coefficients + 1)]
    )


def combined_feature_names(coefficients: int = 12) -> tuple[str, ...]:
    """Full feature order when formants and MFCCs are used together."""
    return FORMANT_FEATURE_NAMES + mfcc_feature_names(coefficients)


def vowel_features(
    measurement: dict[str, float],
    speaker_f1: float,
    speaker_f2: float,
) -> list[float]:
    """Build the feature row for one vowel.

    Includes both absolute formants and values relative to the speaker's own
    vowel space, because absolute frequencies vary with vocal tract length while
    the *relationships* between vowels are stable across speakers.

    Formant **bandwidths** are included because nasality is largely a bandwidth
    phenomenon and the earlier feature set could not see it. Coupling the nasal
    cavity adds a pole-zero pair near F1 and damps the resonance, so a nasal
    vowel shows a markedly wider B1 than its oral counterpart at a similar F1.
    Centre frequencies alone cannot express that, which is why ``an``/``a`` and
    ``ọn``/``ọ`` stayed weak however much data was added -- the distinction was
    not in the features at all.
    """
    f1 = measurement.get("f1", 0.0)
    f2 = measurement.get("f2", 0.0)
    f3 = measurement.get("f3", 0.0)
    b1 = measurement.get("b1", 0.0)
    b2 = measurement.get("b2", 0.0)
    b3 = measurement.get("b3", 0.0)
    return [
        f1,
        f2,
        f3,
        f1 / speaker_f1 if speaker_f1 else 0.0,
        f2 / speaker_f2 if speaker_f2 else 0.0,
        f2 - f1,
        f1 / f2 if f2 else 0.0,
        b1,
        b2,
        b3,
        # Damping relative to the resonance it belongs to: a wide B1 under a low
        # F1 is the classic nasal signature, and the ratio states it directly
        # rather than making the model infer it from two raw numbers.
        b1 / f1 if f1 else 0.0,
        b2 / f2 if f2 else 0.0,
        b1 / b2 if b2 else 0.0,
    ]


class VowelClassifier:
    """Wraps a fitted estimator mapping formants to vowel identity."""

    def __init__(self, model: Any, labels: list[str], metadata: dict[str, Any] | None = None) -> None:
        self.model = model
        self.labels = labels
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: Path | None = None) -> VowelClassifier:
        """Load a trained classifier.

        Raises:
            ToneAnalysisError: If the model file is missing or unreadable.
        """
        model_path = path or DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise ToneAnalysisError(
                f"No vowel classifier at {model_path}. Train one with:\n"
                "  python scripts/train_vowel_classifier.py"
            )
        try:
            import joblib  # noqa: PLC0415

            payload = joblib.load(model_path)
        except Exception as exc:
            raise ToneAnalysisError(f"Could not load {model_path}: {exc}") from exc

        metadata = payload.get("metadata", {})
        accuracy = metadata.get("holdout_accuracy")
        logger.info(
            "Loaded vowel classifier from %s (%d classes, held-out accuracy %s)",
            model_path, len(payload["labels"]),
            f"{accuracy:.1%}" if accuracy else "unknown",
        )
        return cls(model=payload["model"], labels=payload["labels"], metadata=metadata)

    def save(self, path: Path | None = None) -> Path:
        """Persist model, label set, and provenance."""
        model_path = path or DEFAULT_MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib  # noqa: PLC0415

        joblib.dump(
            {"model": self.model, "labels": self.labels, "metadata": self.metadata},
            model_path,
        )
        logger.info("Wrote vowel classifier to %s", model_path)
        return model_path

    def predict(self, rows: list[list[float]]) -> list[tuple[str, float]]:
        """Predict vowel identity for each feature row.

        Returns:
            ``(label, confidence)`` per row, where confidence is the predicted
            probability of the chosen class.
        """
        if not rows:
            return []

        probabilities = self.model.predict_proba(rows)
        results = []
        for distribution in probabilities:
            best = max(range(len(distribution)), key=distribution.__getitem__)
            results.append((self.labels[best], float(distribution[best])))
        return results


def build_estimator(random_state: int = 0, balanced: bool = False) -> Any:
    """Create the untrained estimator used for vowel identity.

    Args:
        random_state: Seed, for reproducible training.
        balanced: Weight classes inversely to their frequency. The nasal vowels
            are outnumbered by their oral counterparts from 1.6:1 (``un``) to
            58:1 (``ẹn``), and measured F1 tracks that ratio almost exactly --
            0.41 down to 0.00. Without weighting the model is simply rewarded
            for never predicting the rare class.
    """
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415
    except ImportError as exc:
        raise ToneAnalysisError(
            "scikit-learn is required for the vowel classifier:\n"
            "  pip install scikit-learn joblib"
        ) from exc

    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=6,
        l2_regularization=1.0, random_state=random_state,
        class_weight="balanced" if balanced else None,
    )
