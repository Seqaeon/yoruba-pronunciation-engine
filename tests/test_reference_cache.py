"""The reference cache must survive being used twice.

A name collision once made the second assessment from a single pipeline fail:
``_reference_cache`` held the cache object at construction and was then
overwritten with the reference ``UtteranceAnalysis`` for report rendering, so
the second call found the wrong type. Every test passed, because none of them
assessed twice from one pipeline.
"""

from __future__ import annotations

from pathlib import Path

from yoruba_engine.cache import ReferenceCache


def test_key_changes_with_every_input(tmp_path: Path) -> None:
    """Audio, transcript and context each have to move the key.

    A cache that ignores any of them returns an analysis produced by a
    different aligner, dictionary or recording -- worse than no cache, because
    the calibration files, tone classifier and vowel spans all belong to one
    generation of the model.
    """
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    other = tmp_path / "b.wav"
    other.write_bytes(b"y" * 200)

    cache = ReferenceCache.build(tmp_path, audio, audio, "pitch:1")
    base = cache.key(audio, "omi")

    assert cache.key(audio, "omi") == base, "the same inputs must hit"
    assert cache.key(audio, "òkun") != base, "transcript must matter"
    assert cache.key(other, "omi") != base, "audio must matter"

    moved = ReferenceCache.build(tmp_path, audio, audio, "pitch:2")
    assert moved.key(audio, "omi") != base, "pitch settings must matter"


def test_round_trip_and_miss(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    cache = ReferenceCache.build(tmp_path / "store", audio, audio, "pitch:1")

    key = cache.key(audio, "omi")
    assert cache.get(key) is None, "an empty cache must miss, not raise"

    cache.put(key, {"words": ["omi"]})
    assert cache.get(key) == {"words": ["omi"]}


def test_unreadable_entry_is_a_miss(tmp_path: Path) -> None:
    """A truncated or foreign entry must read as absent, never raise."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x" * 100)
    store = tmp_path / "store"
    cache = ReferenceCache.build(store, audio, audio, "pitch:1")
    key = cache.key(audio, "omi")

    store.mkdir(parents=True, exist_ok=True)
    cache.path_for(key).write_bytes(b"not a joblib file")
    assert cache.get(key) is None


def test_pipeline_keeps_its_cache_across_assessments() -> None:
    """The attribute holding the cache must not be reused for anything else.

    This is the collision that broke the second assessment, expressed as a
    check that does not need audio: the pipeline's cache attribute and the
    attribute it exposes the reference analysis under have to be distinct.
    """
    import inspect

    from yoruba_engine import pipeline as module

    source = inspect.getsource(module)
    assert "self._reference_store = self._build_reference_cache()" in source
    # The reference analysis is exposed under its own name, not the cache's.
    assert "self._reference_store = reference" not in source
