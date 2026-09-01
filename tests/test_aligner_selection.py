"""Tests for which aligner an assessment actually gets.

The pipeline picks the neural frame aligner when its head is present and MFA
when it is not, and the difference is 0.56 s against 13.79 s per utterance. The
choice is therefore worth a test on its own, for a reason the timing does not
convey: getting it wrong raises nothing. MFA returns real alignments and the
assessment comes back with ordinary-looking scores, twenty-five times slower,
announced in one log line. A bug that only makes software slow is one nobody
goes looking for.

The head is 2.2 MB and committed, so the neural aligner is what every clone
gets. What could take it away is not a missing file but a working directory:
``available`` was an existence check against a relative path, so a backend
started from anywhere but the repository root fell back to MFA. The app happens
to chdir first, which is the only reason this never showed up in production.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from yoruba_engine.alignment.mfa import MFAAligner
from yoruba_engine.alignment.neural import NeuralAligner
from yoruba_engine.config import EngineConfig
from yoruba_engine.pipeline import _default_aligner

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def committed(relative: str) -> bytes:
    """A tracked file's content as committed, not as it sits on disk."""
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "show", f"HEAD:{relative}"],
        capture_output=True)
    assert result.returncode == 0, f"{relative} is not committed"
    return result.stdout


class TestTheNeuralAlignerIsTheDefault:
    def test_it_is_chosen_from_the_repository_root(self, monkeypatch):
        monkeypatch.chdir(PROJECT_ROOT)
        assert isinstance(_default_aligner(EngineConfig()), NeuralAligner)

    def test_it_is_chosen_from_any_other_directory(self, tmp_path, monkeypatch):
        """The regression. A relative default made this depend on the cwd."""
        monkeypatch.chdir(tmp_path)
        chosen = _default_aligner(EngineConfig())
        assert isinstance(chosen, NeuralAligner), (
            "Started outside the repository, the pipeline fell back to MFA: "
            "13.8 s per utterance instead of 0.56 s, and nothing raises.")

    def test_the_head_it_finds_is_the_committed_one(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        chosen = _default_aligner(EngineConfig())
        assert chosen.model_path.resolve() == (
            PROJECT_ROOT / "models" / "frame_aligner.pt").resolve()

    def test_an_explicit_absolute_path_is_left_alone(self):
        named = PROJECT_ROOT / "models" / "frame_aligner.pt"
        assert NeuralAligner(model_path=named).model_path == named


class TestTheFallbackIsAudible:
    """If MFA is ever chosen, the log has to say so loudly."""

    def test_falling_back_to_mfa_warns(self, tmp_path, monkeypatch, caplog):
        # A head that is not there anywhere: absolute, so anchoring cannot
        # rescue it, which is the only way to reach the fallback on a complete
        # checkout.
        monkeypatch.setattr(
            "yoruba_engine.alignment.neural.NeuralAligner.available",
            property(lambda self: False))
        with caplog.at_level(logging.WARNING, logger="yoruba_engine.pipeline"):
            chosen = _default_aligner(EngineConfig())
        assert isinstance(chosen, MFAAligner)
        assert any(record.levelno >= logging.WARNING for record in caplog.records), (
            "A silent 25x slowdown. The fallback must warn, not inform.")

    def test_the_warning_says_what_it_costs(self, monkeypatch, caplog):
        monkeypatch.setattr(
            "yoruba_engine.alignment.neural.NeuralAligner.available",
            property(lambda self: False))
        with caplog.at_level(logging.WARNING, logger="yoruba_engine.pipeline"):
            _default_aligner(EngineConfig())
        assert "13.8" in caplog.text, (
            "The reader has to be told the size of what just happened.")


class TestGopReferenceFollowsTheAligner:
    """The two are not interchangeable, and picking the wrong pair is silent.

    MFA reports a GMM log-likelihood and the neural head reports a log-posterior.
    They are not on the same scale, so a reference built for one scores nonsense
    against the other -- confidently, and without raising.
    """

    def test_the_neural_reference_is_committed(self):
        # Asked of git, not of the disk. A development checkout can have these
        # files present and untracked, or tracked and locally deleted; what a
        # clone gets is the only thing that matters, and it is what the two
        # tests below compare.
        assert committed("models/gop_reference_neural.json")

    def test_both_references_are_committed_and_different(self):
        assert (committed("models/gop_reference_neural.json")
                != committed("models/gop_reference.json")), (
            "One reference has been copied over the other; the scales differ.")


@pytest.mark.parametrize("cwd_is_root", [True, False])
def test_the_choice_does_not_depend_on_the_working_directory(
        cwd_is_root, tmp_path, monkeypatch):
    """Stated once more as an invariant, because that is what it is."""
    monkeypatch.chdir(PROJECT_ROOT if cwd_is_root else tmp_path)
    assert type(_default_aligner(EngineConfig())) is NeuralAligner
