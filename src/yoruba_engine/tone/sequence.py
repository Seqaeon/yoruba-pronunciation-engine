"""Tone read from the whole utterance at once.

The gradient-boosted tree in :mod:`~yoruba_engine.tone.classifier` decides each
tone-bearing unit on its own, from 14 features. It reaches 69.6% on held-out
speakers, and the finding that motivated replacing it is that eight times more
training data moved that to 70.2%. It was not short of examples.

Yoruba tone is relative. F0 declines across an utterance, so a High late in a
phrase can sit below a Low early in it, and the tree sees that only through
three fixed lag features -- which its own permutation importance ranks second,
third and fourth. A model that reads the sequence learns declination, neighbour
interaction and phrase position together:

    gradient-boosted tree          69.6%
    BiGRU over the utterance       79.0%

Same features, same corpus, same kind of speaker-disjoint split. The gain is
representational, not informational.

**The interface is unchanged.** :class:`~yoruba_engine.tone.analyzer.ToneAnalyzer`
already hands its classifier the whole utterance's units in time order, because
the tree's features needed neighbours too. So this is a drop-in: same
``predict(vowels)``, same contract of setting ``detected_tone`` and
``confidence`` on measurable units and leaving the rest ``UNKNOWN``.

**It is not a bigger model, it is a differently shaped one.** Two GRU layers of
96 units, about 250k parameters, running on 14 numbers per unit. Inference is
one forward pass over a handful of units, so it costs microseconds against the
0.5 s the aligner spends and the 0.3 s Whisper spends.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..datatypes import VowelSegment
from .classifier import TONE_ORDER
from .features import extract_features, measurable

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

#: Where the trained sequence model lives.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "tone_sequence.pt"


class ToneSequenceClassifier:
    """A BiGRU over an utterance's tone-bearing units.

    Interface-compatible with :class:`~yoruba_engine.tone.classifier.ToneClassifier`,
    so which one the analyzer holds is a loading decision rather than a code
    path.
    """

    def __init__(self, net, mean, sd, metadata: dict, device: str = "cpu") -> None:
        self.net = net
        self.mean = mean
        self.sd = sd
        self.metadata = metadata
        self.device = device

    @property
    def available(self) -> bool:
        return self.net is not None

    @classmethod
    def load(cls, path: Path | None = None, device: str | None = None):
        """Load the trained model, or return ``None`` if it is not there.

        Returning ``None`` rather than raising lets the caller fall back to the
        tree, which is a worse model but a working one -- and makes the absence
        visible in a log rather than as a crash on the first assessment.
        """
        model_path = Path(path or DEFAULT_MODEL_PATH)
        if not model_path.exists():
            logger.info("No tone sequence model at %s", model_path)
            return None
        try:
            import numpy as np  # noqa: PLC0415
            import torch  # noqa: PLC0415
            import torch.nn as nn  # noqa: PLC0415
        except ImportError:
            logger.info("torch is not installed; the tone sequence model needs it")
            return None

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        hidden = int(checkpoint["hidden"])
        features = int(checkpoint["features"])
        net = nn.ModuleDict({
            "project": nn.Sequential(nn.Linear(features, hidden), nn.GELU()),
            "rnn": nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2),
            "head": nn.Linear(hidden * 2, len(TONE_ORDER)),
        })
        net.load_state_dict(checkpoint["state_dict"])
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(resolved).eval()

        metadata = {k: v for k, v in checkpoint.items()
                    if k not in {"state_dict", "mean", "sd"}}
        logger.info("Loaded tone sequence model from %s (held-out accuracy %.3f)",
                    model_path, metadata.get("holdout_accuracy", float("nan")))
        return cls(net, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["sd"]),
                   metadata, resolved)

    def predict(self, vowels: list[VowelSegment]) -> None:
        """Classify an utterance's tone-bearing units in place.

        Sets ``detected_tone`` and ``confidence`` on every measurable unit and
        leaves the rest ``UNKNOWN`` with zero confidence -- the same contract the
        tree and the rule path honour, so a caller never has to know which
        produced a result.

        The whole utterance goes through in one pass, which is the point: each
        unit's prediction is conditioned on the ones around it.
        """
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        rows = extract_features(vowels)
        if not rows:
            return
        targets = measurable(vowels)

        # Standardised with the training split's statistics, which are stored
        # with the model. Recomputing them per utterance would normalise a
        # learner against themselves and erase the register information the
        # model was trained to read.
        matrix = (np.asarray(rows, dtype=np.float32) - self.mean) / self.sd
        batch = torch.tensor(matrix, dtype=torch.float32,
                             device=self.device).unsqueeze(0)

        with torch.inference_mode():
            hidden = self.net["project"](batch)
            output, _ = self.net["rnn"](hidden)
            logits = self.net["head"](output)[0]
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        for vowel, distribution in zip(targets, probabilities):
            best = int(distribution.argmax())
            vowel.detected_tone = TONE_ORDER[best]
            vowel.confidence = float(distribution[best])
