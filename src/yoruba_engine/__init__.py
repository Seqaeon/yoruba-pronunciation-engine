"""Yoruba pronunciation assessment engine.

Compares a learner's spoken Yoruba against a native-speaker reference and scores
segmental accuracy, tonal accuracy, and overall pronunciation.

Pipeline stages:
    1. :mod:`yoruba_engine.audit` -- dataset audit and vocabulary extraction
    2. :mod:`yoruba_engine.g2p` -- pronunciation dictionary generation
    3. :mod:`yoruba_engine.alignment` -- Montreal Forced Aligner integration
    4. :mod:`yoruba_engine.pitch` -- pitch extraction over vowel segments
    5. :mod:`yoruba_engine.tone` -- H/M/L tone detection and comparison
    6. :mod:`yoruba_engine.compare` -- word alignment and scoring
    7. :mod:`yoruba_engine.report` -- report generation

Quick start::

    from pathlib import Path
    from yoruba_engine import AssessmentPipeline

    pipeline = AssessmentPipeline()
    result = pipeline.assess(
        reference_audio=Path("reference.wav"),
        learner_audio=Path("learner.wav"),
        target_text="òrí",
    )
    print(result.overall_score, result.feedback)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import DEFAULT_CONFIG, EngineConfig
from .datatypes import ComparisonResult, Tone, ToneError, UtteranceAnalysis, WordAnalysis
from .exceptions import YorubaEngineError
from .pipeline import AssessmentPipeline

__all__ = [
    "__version__",
    "AssessmentPipeline",
    "EngineConfig",
    "DEFAULT_CONFIG",
    "Tone",
    "ToneError",
    "WordAnalysis",
    "UtteranceAnalysis",
    "ComparisonResult",
    "YorubaEngineError",
]
