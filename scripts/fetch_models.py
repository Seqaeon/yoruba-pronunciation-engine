"""Fetch and verify the trained artifacts a clone does not carry.

The acoustic model and the two classifiers total ~64 MB, which is too much for
the repository, so they are attached to a GitHub Release and described by
``models/MANIFEST.json``.

Verification is not ceremony here. The calibration files that *are* committed --
``gop_reference.json`` (per-phone likelihood means), ``word_duration_stats.json``
(p5 duration thresholds) -- are measurements of one specific acoustic model's
behaviour. Pair them with a different model and every threshold is quietly
wrong: nothing raises, the feedback just degrades. So the manifest records the
acoustic model's checksum and this script refuses a mismatch rather than
proceeding into that failure.

    python scripts/fetch_models.py            # download whatever is missing
    python scripts/fetch_models.py --check    # verify only, download nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "models" / "MANIFEST.json"

#: Where release assets live. Override with --base-url for a mirror or a
#: local file:// path when the machine has no internet access.
DEFAULT_BASE = "https://github.com/{repo}/releases/download/{tag}/"
DEFAULT_REPO = "Seqaeon/yoruba-pronunciation-engine"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(path: Path, expected: dict) -> str:
    """Return 'ok', 'missing', 'wrong-size' or 'wrong-hash'."""
    if not path.exists():
        return "missing"
    if path.stat().st_size != expected["bytes"]:
        return "wrong-size"
    if digest(path) != expected["sha256"]:
        return "wrong-hash"
    return "ok"


#: Diagnosed once, on the first artifact that needs fetching, and remembered.
_GH_STATE: str | None = None


def gh_state() -> str:
    """Return 'ready', 'absent' or 'unauthenticated', explaining the first time.

    Worth the noise because the failure is otherwise unreadable. Without a
    usable ``gh`` every artifact falls through to the anonymous HTTP path and
    GitHub answers 404 for all of them, so a single missing credential presents
    as five broken URLs -- and the URLs are the one thing that is fine.
    """
    global _GH_STATE
    if _GH_STATE is not None:
        return _GH_STATE
    if shutil.which("gh") is None:
        _GH_STATE = "absent"
        print("\n! The GitHub CLI (gh) is not on PATH, so the download below can only\n"
              "  be attempted anonymously. Release assets on a private repository are\n"
              "  invisible to an anonymous request -- GitHub answers 404 rather than\n"
              "  401 -- so every artifact will fail with a 404 that is about your\n"
              "  credentials and not about the address it was fetched from.\n"
              "\n"
              "    Windows:  winget install GitHub.cli\n"
              "    macOS:    brew install gh\n"
              "    Linux:    see https://github.com/cli/cli#installation\n"
              "\n"
              "  Then:  gh auth login\n")
        return _GH_STATE
    probe = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if probe.returncode != 0:
        _GH_STATE = "unauthenticated"
        print("\n! The GitHub CLI is installed but not signed in. A private repository's\n"
              "  release assets answer 404 to an unauthenticated request, so the\n"
              "  downloads below will fail as though the files did not exist.\n"
              "\n"
              "  Sign in with:  gh auth login\n"
              "  You also need read access to the engine repository itself; the same\n"
              "  grant that lets you clone it is what releases the artifacts.\n")
        return _GH_STATE
    _GH_STATE = "ready"
    return _GH_STATE


def download_with_gh(repo: str, tag: str, destination: Path,
                     asset: str | None = None) -> bool:
    """Fetch a release asset through the GitHub CLI.

    Release assets on a **private** repository are not reachable over anonymous
    HTTP -- the download URL returns 404 rather than 401, which is why this is
    tried first rather than as a fallback. ``gh`` carries the user's existing
    credentials, so it works for private and public alike.

    Returns whether the asset was written.
    """
    if gh_state() != "ready":
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gh", "release", "download", tag, "--repo", repo,
         "--pattern", asset or destination.name,
         "--dir", str(destination.parent), "--clobber"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"    gh: {(result.stderr or result.stdout).strip()[:160]}")
        return False
    # gh saves the file under the release's name for it, which is not always the
    # name the artifact carries on disk -- two encoders both live at
    # model.safetensors locally but cannot share one name on the release.
    landed = destination.parent / (asset or destination.name)
    if landed != destination and landed.exists():
        landed.replace(destination)
    return destination.exists()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        with partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r    {written / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
    print()
    partial.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Verify without downloading")
    parser.add_argument("--include-optional", action="store_true",
                        help="Also fetch optional artifacts from HuggingFace. "
                             "Currently the Whisper transcriber, 894 MB, "
                             "needed only for --transcribe.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub owner/name")
    parser.add_argument("--tag", help="Release tag; defaults to the manifest's")
    parser.add_argument("--base-url", help="Override where assets are fetched from")
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tag = args.tag or manifest.get("release_tag", "v0.1.0")
    base = args.base_url or DEFAULT_BASE.format(repo=args.repo, tag=tag)
    if not base.endswith("/"):
        base += "/"

    statuses: dict[str, str] = {}
    print(f"{'artifact':<40}{'status':<13}size")
    for relative, expected in manifest["artifacts"].items():
        path = PROJECT_ROOT / relative
        state = verify(path, expected)
        if state != "ok" and not args.check:
            print(f"{relative:<40}{state:<13}fetching")
            # Release assets live in one flat namespace, so two artifacts whose
            # basename is model.safetensors -- the upstream encoder and the
            # fine-tuned one -- cannot both be uploaded under it. The manifest
            # names the asset explicitly where the basename would collide.
            asset = expected.get("asset") or Path(relative).name
            # An artifact unchanged since an earlier release stays on that
            # release rather than being re-uploaded. 1.75 GB of encoder and
            # Whisper weights did not change between v0.5.0 and v0.6.0, and
            # copying them forward would cost an upload for no difference.
            asset_tag = args.tag or expected.get("tag") or tag
            fetched = download_with_gh(args.repo, asset_tag, path, asset=asset)
            if not fetched:
                url = (args.base_url or DEFAULT_BASE.format(
                    repo=args.repo, tag=asset_tag)) + asset
                try:
                    download(url, path)
                    fetched = True
                except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                    print(f"  ! {url}\n    {exc}")
            if not fetched:
                statuses[relative] = state
                continue
            state = verify(path, expected)
            # A download that lands and then fails verification is not a
            # corrupted transfer; it is the release carrying a different build
            # from the one the manifest was written against. Say so, because
            # re-running the fetch will reproduce it forever.
            if state in ("wrong-size", "wrong-hash"):
                print(f"    ! {asset} on {asset_tag} is not the file the manifest "
                      f"describes.\n"
                      f"      The asset on the release is stale. Replace it with:\n"
                      f"        gh release upload {asset_tag} <path-to-{asset}> "
                      f"--repo {args.repo} --clobber")
        statuses[relative] = state
        print(f"{relative:<40}{state:<13}{expected['bytes'] / 1e6:.1f} MB")

    # Optional artifacts are archives: a Whisper checkpoint is a directory of
    # weights, tokenizer and config, and a release asset is a single file.
    for relative, spec in manifest.get("optional_artifacts", {}).items():
        path = PROJECT_ROOT / relative
        if (path / "config.json").exists():
            print(f"{relative:<40}{'ok':<13}{spec['bytes'] / 1e6:.1f} MB")
            continue
        if not args.include_optional:
            print(f"{relative:<40}{'optional':<13}--include-optional to fetch")
            continue
        if args.check:
            print(f"{relative:<40}{'missing':<13}{spec['bytes'] / 1e6:.1f} MB")
            continue

        print(f"{relative:<40}{'fetching':<13}{spec['bytes'] / 1e6:.1f} MB")
        archive = PROJECT_ROOT / "models" / spec["asset"]
        asset_tag = args.tag or spec.get("tag") or tag
        if not download_with_gh(args.repo, asset_tag, archive, asset=spec["asset"]):
            try:
                download((args.base_url or DEFAULT_BASE.format(
                    repo=args.repo, tag=asset_tag)) + spec["asset"], archive)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"  ! {exc}")
                continue
        if digest(archive) != spec["sha256"]:
            print(f"  ! {spec['asset']} does not match its recorded sha256")
            archive.unlink(missing_ok=True)
            continue
        import tarfile  # noqa: PLC0415

        with tarfile.open(archive) as bundle:
            bundle.extractall(PROJECT_ROOT / "models")
        archive.unlink(missing_ok=True)
        print(f"{relative:<40}{'ok':<13}extracted")

    # The dictionary is committed, so a mismatch means someone regenerated it --
    # and the acoustic model will not align against a changed phone set.
    dictionary = PROJECT_ROOT / "data" / "dictionary" / "yoruba.dict"
    expected_dictionary = manifest.get("calibrated_against", {}).get("dictionary_sha256")
    if expected_dictionary and dictionary.exists():
        if digest(dictionary) != expected_dictionary:
            print("\n! The pronunciation dictionary differs from the one the acoustic "
                  "model was trained against.\n"
                  "  MFA does not report a phone-set mismatch -- it emits meaningless "
                  "alignments.\n"
                  "  Restore the committed dictionary, or retrain the model against yours.")
            return 2

    failed = {k: v for k, v in statuses.items() if v != "ok"}
    if failed:
        print(f"\n{len(failed)} artifact(s) unresolved:")
        for relative, state in failed.items():
            print(f"  {state:<12}{relative}")
        if args.check:
            print("\nRun without --check to download them.")
        else:
            print(f"\nIf the release is not published, attach these files to tag "
                  f"{tag} on {args.repo}.\n"
                  "If the repository is private, authenticate first:  gh auth login")
        return 1

    print("\nAll artifacts present and verified.")
    print("Calibration files committed in models/ describe this acoustic model "
          "(sha256 " + str(manifest.get("acoustic_model_sha256"))[:12] + "...);")
    print("regenerate them with build_gop_reference.py and measure_alignment_stats.py "
          "if you retrain it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
