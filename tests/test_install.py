"""Tests for what a *clone* gets, rather than for what this working tree has.

Every other test here runs against the checkout it happens to find, which on a
development machine holds files that were never committed and paths that exist
on no other computer. That blind spot has now shipped three broken installs in a
row, each of which the whole suite passed straight through:

* ``models/frame_aligner.pt`` recorded the absolute path of the encoder it was
  trained against. It named the right directory here and nothing anywhere else,
  so transformers was handed
  ``\\home\\seqaeon\\...\\models\\finetuned_utterance`` on Windows.
* ``.gitignore`` excluded ``models/finetuned_utterance/`` entirely, so the
  release carried 858 MB of weights and the repository carried nothing to
  interpret them with. Missing ``config.json`` does not read as a missing file:
  transformers stops treating the argument as a path and reports a
  huggingface.co URL built out of the local path.
* ``models/MANIFEST.json`` described a vowel classifier that had never been
  uploaded, so ``fetch_models.py`` downloaded the previous build and refused its
  checksum, on every machine, forever.

None of those are runtime bugs, which is why runtime tests could not see them.
They are properties of the distribution: what is tracked, what is ignored, and
what the recorded metadata says. So these tests ask git and the committed bytes,
never the filesystem -- a file that is present but untracked must fail here, or
this suite reproduces the blind spot it exists to close.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: An absolute path that escaped into a committed file: a POSIX home or mount,
#: or a Windows drive letter with either separator.
#:
#: Both branches demand a further path segment, and the drive letter must not
#: follow another letter. Without the first the pattern matches noise in any
#: large binary; without the second it matches every URL, since "https:/" ends
#: in a letter, a colon and a slash. Both mistakes were made writing this, and
#: both make the guard useless in the opposite direction -- one by crying wolf
#: until someone deletes the test, the other by never firing at all.
ABSOLUTE_PATH = re.compile(
    rb"(?:/(?:home|Users|mnt|media)/[\w.+-]+/"
    rb"|(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:[\w .+-]+[\\/])+)")

#: Small files a checkpoint directory is useless without. Weights are fetched;
#: these have to be in the repository, because nothing downloads them.
CHECKPOINT_SIDECAR = "config.json"


def git(*args: str) -> str:
    """Run git in the repository and return its stdout."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), *args],
        capture_output=True, text=True, check=True).stdout


def tracked() -> list[str]:
    """Every path git would hand to a fresh clone."""
    return [line for line in git("ls-files").splitlines() if line]


def committed_bytes(relative: str) -> bytes:
    """The content of a tracked file as committed, not as it sits on disk."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "show", f"HEAD:{relative}"],
        capture_output=True, check=True).stdout


def is_ignored(relative: str) -> bool:
    """Whether .gitignore would keep this path out of a commit."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", relative]).returncode == 0


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(committed_bytes("models/MANIFEST.json"))


class TestNoMachineSpecificPaths:
    """No committed file may name a directory on one particular computer.

    Source is covered by the same rule as data here. A path in a script only
    wastes the reader's time; a path recorded inside a checkpoint is worse,
    because the loader believes it.
    """

    def test_no_absolute_paths_in_committed_model_artifacts(self):
        offenders = {}
        for relative in tracked():
            if not relative.startswith("models/"):
                continue
            found = ABSOLUTE_PATH.findall(committed_bytes(relative))
            if found:
                offenders[relative] = sorted({f.decode("utf-8", "replace") for f in found})
        assert not offenders, (
            "Committed model artifacts record absolute paths, which name the "
            f"right thing on exactly one machine: {offenders}")

    def test_no_absolute_paths_in_committed_scripts(self):
        offenders = {}
        for relative in tracked():
            if not (relative.startswith(("scripts/", "src/", "tests/"))
                    or relative.endswith(".md")):
                continue
            for number, line in enumerate(
                    committed_bytes(relative).splitlines(), start=1):
                if ABSOLUTE_PATH.search(line):
                    offenders.setdefault(relative, []).append(number)
        assert not offenders, (
            f"Absolute paths in committed sources: {offenders}")


class TestCheckpointsAreLoadable:
    """A fetched checkpoint must be loadable with what the clone already has.

    The weights come from a release and the configuration comes from git, so a
    checkpoint is only complete when both halves agree to exist. Only git is
    consulted: this exact bug survived on a machine where the file was sitting
    in the right directory, untracked.
    """

    def encoder_directories(self, manifest) -> set[str]:
        """Checkpoint directories the manifest expects a clone to end up with."""
        directories = set()
        for relative in manifest["artifacts"]:
            parent = str(Path(relative).parent)
            if parent.startswith("models/") and parent != "models":
                directories.add(parent)
        return directories

    def test_manifest_names_at_least_one_checkpoint_directory(self, manifest):
        # Guards the two tests below from passing vacuously if the manifest is
        # ever restructured.
        assert self.encoder_directories(manifest)

    def test_every_fetched_checkpoint_ships_its_config(self, manifest):
        files = set(tracked())
        missing = [
            f"{directory}/{CHECKPOINT_SIDECAR}"
            for directory in sorted(self.encoder_directories(manifest))
            if f"{directory}/{CHECKPOINT_SIDECAR}" not in files
        ]
        assert not missing, (
            "These checkpoints fetch weights but commit no config.json, so "
            "transformers reads the local path as a Hub repository id and "
            f"reports a huggingface.co URL for it: {missing}")

    def test_config_files_are_not_gitignored(self, manifest):
        ignored = [
            path for directory in sorted(self.encoder_directories(manifest))
            if is_ignored(path := f"{directory}/{CHECKPOINT_SIDECAR}")
        ]
        assert not ignored, (
            "A .gitignore rule excludes configuration a clone cannot work "
            f"without: {ignored}")

    def test_weights_stay_out_of_git(self, manifest):
        """The other half of the rule, so nobody fixes the above with a commit."""
        committed = [
            relative for relative in manifest["artifacts"]
            if relative in set(tracked())
        ]
        assert not committed, (
            f"Release artifacts committed to git: {committed}")


class TestRecordedMetadata:
    """What the trained artifacts say about themselves has to stay true."""

    def test_frame_aligner_records_a_relative_encoder(self):
        torch = pytest.importorskip("torch")

        checkpoint = torch.load(
            io.BytesIO(committed_bytes("models/frame_aligner.pt")),
            map_location="cpu", weights_only=False)
        recorded = str(checkpoint["encoder"])
        assert not (recorded.startswith("/")
                    or re.match(r"^[A-Za-z]:[\\/]", recorded)), (
            f"The frame aligner records the absolute encoder path {recorded!r}, "
            "which resolves on the machine that trained it and nowhere else.")
        assert f"{recorded}/{CHECKPOINT_SIDECAR}" in set(tracked()), (
            f"The frame aligner names the encoder {recorded!r}, whose "
            "config.json this repository does not ship.")

    def test_dictionary_matches_what_the_acoustic_model_was_trained_against(
            self, manifest):
        """The check fetch_models.py makes at install time, made at commit time.

        A regenerated dictionary does not raise at runtime: MFA emits
        meaningless alignments instead. Better to fail here.
        """
        expected = manifest["calibrated_against"]["dictionary_sha256"]
        actual = hashlib.sha256(
            committed_bytes("data/dictionary/yoruba.dict")).hexdigest()
        assert actual == expected, (
            "The committed dictionary is not the one the acoustic model was "
            "trained against.")

    def test_dictionary_is_pinned_to_lf(self):
        """CRLF on checkout changes the hash above and breaks a Windows install."""
        attributes = committed_bytes(".gitattributes").decode()
        assert re.search(r"^data/dictionary/\*\.dict\s+text\s+eol=lf",
                         attributes, re.MULTILINE), (
            "Without an eol=lf rule Windows rewrites the dictionary on "
            "checkout, and its sha256 stops matching the manifest.")


class TestManifestIsSelfConsistent:
    """The manifest is what an install trusts; nothing else validates it."""

    def test_every_artifact_declares_size_and_checksum(self, manifest):
        for relative, spec in manifest["artifacts"].items():
            assert isinstance(spec.get("bytes"), int), relative
            assert re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", "")), relative

    def test_colliding_basenames_declare_an_asset_name(self, manifest):
        """A release is one flat namespace; two model.safetensors cannot share it."""
        seen: dict[str, str] = {}
        for relative, spec in manifest["artifacts"].items():
            asset = spec.get("asset") or Path(relative).name
            assert asset not in seen, (
                f"{relative} and {seen[asset]} would both upload as {asset!r}; "
                "one of them needs an explicit 'asset' in the manifest.")
            seen[asset] = relative

    def test_acoustic_model_checksum_matches_its_artifact(self, manifest):
        assert (manifest["acoustic_model_sha256"]
                == manifest["artifacts"]["models/yoruba_acoustic.zip"]["sha256"]), (
            "The calibration files committed here describe the acoustic model "
            "named by acoustic_model_sha256; it has to be the one that is "
            "actually fetched.")


@pytest.mark.skipif(
    os.environ.get("YORUBA_CHECK_RELEASE") != "1",
    reason="Needs the network and an authenticated gh; set YORUBA_CHECK_RELEASE=1")
class TestReleaseMatchesManifest:
    """Whether the files a fresh install downloads are the ones described.

    Off by default because it needs the network and credentials, and the rest of
    this suite deliberately needs neither. It is the only test that can catch a
    manifest updated without the matching upload -- which shipped a release that
    could not complete setup on any machine, and stayed that way until someone
    tried a clean install.
    """

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def assets_for(tag: str) -> tuple[tuple[str, int], ...]:
        """Every asset on one release. Cached: artifacts share their tags."""
        repo = "Seqaeon/yoruba-pronunciation-engine"
        listing = json.loads(subprocess.run(
            ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
            capture_output=True, text=True, check=True).stdout)
        return tuple((asset["name"], asset["size"]) for asset in listing["assets"])

    def test_every_artifact_is_on_its_release_at_the_declared_size(self, manifest):
        default_tag = manifest["release_tag"]
        wrong = {}
        for relative, spec in {**manifest["artifacts"],
                               **manifest.get("optional_artifacts", {})}.items():
            asset = spec.get("asset") or Path(relative).name
            published = dict(self.assets_for(spec.get("tag") or default_tag))
            if published.get(asset) != spec["bytes"]:
                wrong[relative] = (f"manifest {spec['bytes']}, "
                                   f"release {published.get(asset, 'absent')}")
        assert not wrong, f"Release assets disagree with the manifest: {wrong}"


class TestTheGuardItself:
    """That ABSOLUTE_PATH still fires.

    A pattern this narrow is one careless edit away from matching nothing, and a
    guard that cannot fail is worse than no guard: it reports safety. The bad
    samples are the strings that actually shipped; the good ones are what
    previous versions of this pattern falsely accused, which is how the tests
    above got deleted-then-rewritten twice before landing.
    """

    BAD = [
        b"/home/seqaeon/Downloads/Yoruba_Learner/models/finetuned_utterance",
        b"PY=/home/seqaeon/Downloads/venv/bin/python",
        b"C:\\Users\\USER\\yoruba-pronunciation-engine\\models\\finetuned_utterance",
        b"/Users/someone/engine/models/finetuned_utterance",
        b"D:/builds/engine/models/",
    ]

    GOOD = [
        b"https://huggingface.co/LyngualLabs/mms-300m-yoruba-english",
        b"<https://openslr.org/86/>",
        b'"Learned vowel features need torch and transformers:\\n"',
        b"models/finetuned_utterance",
        b"data/dictionary/yoruba.dict",
        b"see INSTALL.md, section 3: models/",
    ]

    @pytest.mark.parametrize("sample", BAD)
    def test_absolute_paths_are_caught(self, sample):
        assert ABSOLUTE_PATH.search(sample), (
            f"{sample!r} shipped in this repository and must be caught")

    @pytest.mark.parametrize("sample", GOOD)
    def test_ordinary_text_is_not_accused(self, sample):
        assert not ABSOLUTE_PATH.search(sample), (
            f"{sample!r} is legitimate and must not be flagged")

    def test_it_catches_the_checkpoint_that_actually_shipped(self):
        """The historical blob, so the guard is measured against the real bug."""
        blob = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "show",
             "daeee58^:models/frame_aligner.pt"],
            capture_output=True)
        if blob.returncode != 0:
            pytest.skip("history not available (shallow clone)")
        assert ABSOLUTE_PATH.search(blob.stdout), (
            "The frame aligner that shipped an absolute encoder path is no "
            "longer detected; the pattern has been narrowed too far.")
