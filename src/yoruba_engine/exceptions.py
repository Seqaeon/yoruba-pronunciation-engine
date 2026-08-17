"""Exception hierarchy for the Yoruba pronunciation assessment engine.

Every failure mode the pipeline can hit deliberately raises one of these rather
than a bare ``Exception``, so that a FastAPI layer can map them onto HTTP status
codes without inspecting message strings.
"""

from __future__ import annotations


class YorubaEngineError(Exception):
    """Base class for every error raised by this package."""


class DatasetError(YorubaEngineError):
    """The corpus on disk is missing, malformed, or internally inconsistent."""


class TranscriptError(DatasetError):
    """A transcript could not be read, parsed, or normalized."""


class G2PError(YorubaEngineError):
    """A word could not be converted to a phoneme sequence."""


class AlignmentError(YorubaEngineError):
    """Montreal Forced Aligner failed, or produced unusable output."""


class MFANotAvailableError(AlignmentError):
    """The ``mfa`` executable or a required model is not installed."""


class PitchExtractionError(YorubaEngineError):
    """Pitch analysis failed for a segment (silence, noise, or too short)."""


class ToneAnalysisError(YorubaEngineError):
    """A tone label could not be assigned to a vowel segment."""


class ComparisonError(YorubaEngineError):
    """Reference and learner utterances could not be compared."""


class ASRNotConfiguredError(YorubaEngineError):
    """Free-form transcription was requested but no ASR backend is wired up."""
