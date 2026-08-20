"""Cache the reference recording's analysis between assessments.

Every assessment analyses two recordings: the learner's, which is different each
time, and the reference, which is not. The reference for a given prompt is a
fixed file with a fixed transcript, and analysing it costs the same forced
alignment as the learner's -- measured at 17.6 seconds of a 43-second
assessment, or 41% of the total, recomputed identically on every request.

Caching it roughly halves latency and per-request CPU, which matters more for
deployment than any instance size: the engine is CPU-bound in MFA, so halving
the alignment work halves what a server must provision.

**The key includes everything that could change the answer.** A cache that
returns a stale analysis after the acoustic model is retrained would be worse
than no cache: the calibration files, the tone classifier and the vowel spans
all belong to a specific aligner, and silently mixing generations is the class
of bug this project has spent the most time on. The key therefore covers the
audio, the transcript, the acoustic model, the dictionary, and the pitch
settings that decide where vowels begin.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Bumped when the cached object's shape changes in a way older entries cannot
#: satisfy. Entries carrying a different version are ignored, not upgraded.
FORMAT_VERSION = 1


def _fingerprint(path: Path) -> str:
    """Identify a file cheaply, by size and modification time.

    Hashing a 92 MB acoustic model on every assessment would cost more than the
    cache saves. Size and mtime miss only a same-size edit that preserves the
    timestamp, which does not happen to a model file in practice.
    """
    try:
        stat = path.stat()
    except OSError:
        return f"missing:{path}"
    return f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"


@dataclass(frozen=True)
class ReferenceCache:
    """Stores analysed reference recordings on disk, keyed by what produced them."""

    directory: Path
    #: Everything outside the audio and transcript that changes the analysis.
    context: str

    @classmethod
    def build(cls, directory: Path, acoustic_model: Path, dictionary: Path,
              pitch_settings: str) -> ReferenceCache:
        context = "|".join((
            f"v{FORMAT_VERSION}",
            _fingerprint(acoustic_model),
            _fingerprint(dictionary),
            pitch_settings,
        ))
        return cls(directory=directory, context=context)

    def key(self, audio_path: Path, transcript: str) -> str:
        material = "|".join((
            self.context, _fingerprint(audio_path), transcript.strip(),
        ))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    def path_for(self, key: str) -> Path:
        return self.directory / f"{key}.joblib"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            import joblib  # noqa: PLC0415

            payload = joblib.load(path)
        except Exception as exc:
            # A cache that cannot be read is a cache miss, never an error. It
            # may have been written by another version, or truncated by a
            # process that died mid-write.
            logger.warning("Ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        if payload.get("version") != FORMAT_VERSION:
            return None
        return payload.get("analysis")

    def put(self, key: str, analysis: Any) -> None:
        try:
            import joblib  # noqa: PLC0415

            self.directory.mkdir(parents=True, exist_ok=True)
            # Write beside the target and rename, so a crash mid-write leaves no
            # half-file that a later run would read as valid.
            temporary = self.path_for(key).with_suffix(".partial")
            joblib.dump({"version": FORMAT_VERSION, "analysis": analysis}, temporary)
            temporary.replace(self.path_for(key))
        except Exception as exc:
            logger.warning("Could not cache reference analysis: %s", exc)
