"""Trained tone classifier.

An optional replacement for the threshold rules in
:class:`~yoruba_engine.tone.analyzer.ToneAnalyzer`. The rules stay the default
because they need no model file and no scikit-learn; this is opt-in.

Ground truth is free: Yoruba orthography encodes tone, so every diacritized
transcript in the corpus is a labelled example. No hand-annotation is needed.

Train one with::

    python scripts/train_tone_classifier.py

then use it::

    from yoruba_engine.tone.classifier import ToneClassifier
    from yoruba_engine.tone.analyzer import ToneAnalyzer

    analyzer = ToneAnalyzer(classifier=ToneClassifier.load())

scikit-learn is an optional dependency. Everything here raises a clear error
rather than failing obscurely when it is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT
from ..exceptions import ToneAnalysisError
from ..logging_utils import get_logger
from ..datatypes import Tone, VowelSegment
from .features import FEATURE_COUNT, FEATURE_NAMES, extract_features, measurable

logger = get_logger("tone.classifier")

#: Where ``scripts/train_tone_classifier.py`` writes the model by default.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tone_classifier.joblib"

#: Label encoding. Fixed so a model stays readable across versions.
TONE_ORDER: tuple[Tone, ...] = (Tone.HIGH, Tone.MID, Tone.LOW)
TONE_TO_INDEX = {tone: index for index, tone in enumerate(TONE_ORDER)}


def _require_sklearn() -> Any:
    """Import scikit-learn, or explain how to install it."""
    try:
        import sklearn  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ToneAnalysisError(
            "scikit-learn is required for the trained tone classifier.\n"
            "  pip install scikit-learn joblib\n"
            "Or use the default rule-based analyzer, which needs no model."
        ) from exc
    return sklearn


class ToneClassifier:
    """Wraps a fitted scikit-learn estimator over :mod:`.features`.

    Attributes:
        model: The fitted estimator.
        metadata: Provenance recorded at training time -- corpus size,
            cross-validated accuracy, feature names. Written alongside the model
            so a deployed classifier can always be traced back to its training
            run.
    """

    def __init__(self, model: Any, metadata: dict[str, Any] | None = None) -> None:
        self.model = model
        self.metadata: dict[str, Any] = metadata or {}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> ToneClassifier:
        """Load a trained classifier from disk.

        Args:
            path: Model file; defaults to :data:`DEFAULT_MODEL_PATH`.

        Raises:
            ToneAnalysisError: If the file is missing or unreadable.
        """
        model_path = path or DEFAULT_MODEL_PATH
        if not model_path.exists():
            raise ToneAnalysisError(
                f"No tone classifier at {model_path}. Train one with:\n"
                "  python scripts/train_tone_classifier.py"
            )

        _require_sklearn()
        try:
            import joblib  # noqa: PLC0415

            payload = joblib.load(model_path)
        except Exception as exc:
            raise ToneAnalysisError(f"Could not load {model_path}: {exc}") from exc

        metadata = payload.get("metadata", {})
        accuracy = metadata.get("holdout_accuracy")
        logger.info(
            "Loaded tone classifier from %s (%s, trained on %s vowels, held-out accuracy %s)",
            model_path,
            metadata.get("kind", "unknown"),
            metadata.get("training_vowels", "?"),
            f"{accuracy:.1%}" if accuracy else "unknown",
        )
        return cls(model=payload["model"], metadata=metadata)

    def save(self, path: Path | None = None) -> Path:
        """Persist the model and its metadata."""
        model_path = path or DEFAULT_MODEL_PATH
        model_path.parent.mkdir(parents=True, exist_ok=True)

        import joblib  # noqa: PLC0415

        joblib.dump({"model": self.model, "metadata": self.metadata}, model_path)
        logger.info("Wrote tone classifier to %s", model_path)
        return model_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, vowels: list[VowelSegment]) -> None:
        """Classify an utterance's vowels in place.

        Sets ``detected_tone`` and ``confidence`` on every measurable vowel.
        Unmeasurable vowels are left as :attr:`Tone.UNKNOWN` with zero
        confidence -- the same contract the rule-based path honours, so a caller
        never has to know which classifier produced a result.

        Args:
            vowels: The utterance's vowels in time order.
        """
        rows = extract_features(vowels)
        if not rows:
            return

        targets = measurable(vowels)
        probabilities = self.model.predict_proba(rows)

        for vowel, distribution in zip(targets, probabilities):
            best = max(range(len(distribution)), key=distribution.__getitem__)
            vowel.detected_tone = TONE_ORDER[best]
            # Confidence is the predicted probability, which is directly
            # comparable across tones -- unlike the rule path's distance-from-
            # threshold heuristic, and better suited to suppressing shaky
            # feedback via ToneConfig.min_confidence.
            vowel.confidence = float(distribution[best])

    def feature_importances(self) -> list[tuple[str, float]]:
        """Feature names paired with their importances, most important first.

        Gradient boosting histograms expose no ``feature_importances_``
        attribute, so the training script computes permutation importance on the
        held-out speakers and stores it in :attr:`metadata`. That is the more
        meaningful measure anyway: it reports how much held-out accuracy drops
        when a feature is shuffled, rather than how often the tree split on it.

        Returns an empty list when neither source is available.
        """
        stored = self.metadata.get("feature_importances")
        if stored:
            return [(name, float(value)) for name, value in stored]

        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return []
        return sorted(zip(FEATURE_NAMES, importances), key=lambda pair: -pair[1])


def build_estimator(kind: str = "gbm", random_state: int = 0) -> Any:
    """Create an untrained estimator.

    Args:
        kind: ``"gbm"`` for histogram gradient boosting (best accuracy here), or
            ``"logreg"`` for logistic regression (smaller, more interpretable,
            about two points worse on this corpus).
        random_state: Seed, so training runs are reproducible.

    Raises:
        ToneAnalysisError: If ``kind`` is unknown or scikit-learn is missing.
    """
    _require_sklearn()

    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            l2_regularization=1.0,
            random_state=random_state,
        )

    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=random_state),
        )

    raise ToneAnalysisError(f"Unknown classifier kind {kind!r}; use 'gbm' or 'logreg'")


def check_feature_count(rows: list[list[float]]) -> None:
    """Guard against a feature-extraction change that a stale model cannot serve.

    Raises:
        ToneAnalysisError: If a row does not have :data:`FEATURE_COUNT` values.
    """
    for row in rows:
        if len(row) != FEATURE_COUNT:
            raise ToneAnalysisError(
                f"Feature extraction produced {len(row)} values, expected "
                f"{FEATURE_COUNT}. Retrain the classifier after changing features."
            )
