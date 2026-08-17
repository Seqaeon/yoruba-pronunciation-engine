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
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "models" / "MANIFEST.json"

#: Where release assets live. Override with --base-url for a mirror or a
#: local file:// path when the machine has no internet access.
DEFAULT_BASE = "https://github.com/{repo}/releases/download/{tag}/"
DEFAULT_REPO = "OWNER/yoruba-pronunciation-engine"


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
            url = base + Path(relative).name
            print(f"{relative:<40}{state:<13}fetching from {url}")
            try:
                download(url, path)
            except (urllib.error.URLError, urllib.error.HTTPError) as exc:
                print(f"  ! download failed: {exc}")
                statuses[relative] = state
                continue
            state = verify(path, expected)
        statuses[relative] = state
        print(f"{relative:<40}{state:<13}{expected['bytes'] / 1e6:.1f} MB")

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
            print(f"\nIf the release is not published yet, attach these files to "
                  f"tag {tag} on {args.repo}, or pass --base-url.")
        return 1

    print("\nAll artifacts present and verified.")
    print("Calibration files committed in models/ describe this acoustic model "
          "(sha256 " + str(manifest.get("acoustic_model_sha256"))[:12] + "...);")
    print("regenerate them with build_gop_reference.py and measure_alignment_stats.py "
          "if you retrain it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
