"""Merge the raw downloaded datasets into the engine's expected corpus layout.

The project ships two very different sources:

**OpenSLR-style** (``yo_ng_female``, ``yo_ng_male``)
    A flat directory of WAV files alongside a ``line_index.tsv`` of
    ``utterance_id<TAB>transcript``. Transcripts are fully diacritized, so this
    is the only source usable as a tone reference.

**Yoruba-TTS-Dataset**
    Per-session subdirectories of MP3 clips, each with a ``mapping.tsv`` whose
    columns are ``audio_filename, key, sentence, attempts``. The sentences carry
    **no tone marks**, which makes this source unusable for tone comparison. It
    can still contribute audio to acoustic model training, but only when the
    dictionary is built with tone-stripped phones -- otherwise every one of its
    vowels would be labelled Mid and would poison the tone-conditioned phone
    models. It is therefore opt-in via ``--include-tts``.

Output layout::

    data/corpus/
    ├── line_index.tsv     utterance_id <TAB> speaker_id <TAB> transcript
    └── wavs/              symlinks to (or converted copies of) the source audio

Audio is symlinked rather than copied, so the corpus costs no extra disk. MP3
sources are converted to WAV with ffmpeg, because MFA does not read MP3.

Usage::

    python scripts/prepare_dataset.py                    # OpenSLR only (default)
    python scripts/prepare_dataset.py --include-tts      # add the TTS clips
    python scripts/prepare_dataset.py --copy             # copy instead of symlink
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
WAVS_DIR = CORPUS_DIR / "wavs"
INDEX_PATH = CORPUS_DIR / "line_index.tsv"

#: Sources laid out as a flat WAV directory plus a line_index.tsv.
OPENSLR_SOURCES = ("yo_ng_female", "yo_ng_male")

#: Root of the per-session MP3 dataset.
TTS_ROOT = DATA_DIR / "Yoruba-TTS-Dataset"

_TONE_MARKS = ("́", "̀")  # combining acute, combining grave


@dataclass
class Entry:
    """One staged utterance."""

    utterance_id: str
    speaker_id: str
    transcript: str
    source_audio: Path
    needs_conversion: bool = False


def has_tone_marks(text: str) -> bool:
    """Whether the text carries any tone diacritic."""
    return any(mark in unicodedata.normalize("NFD", text) for mark in _TONE_MARKS)


def is_apple_double(path: Path) -> bool:
    """Whether this is a macOS AppleDouble sidecar (``._name``) rather than data.

    The archives were built on macOS, so every real file has a ``._`` twin that
    is metadata, not audio. Including them corrupts the corpus.
    """
    return path.name.startswith("._")


def collect_openslr(source_dir: Path) -> list[Entry]:
    """Read one OpenSLR-style directory.

    The speaker id is the utterance id minus its trailing numeric segment:
    ``yof_06136_00616155826`` belongs to speaker ``yof_06136``.
    """
    index = source_dir / "line_index.tsv"
    if not index.exists():
        print(f"  SKIP {source_dir.name}: no line_index.tsv", file=sys.stderr)
        return []

    entries: list[Entry] = []
    missing = 0

    with index.open(encoding="utf-8", newline="") as handle:
        for parts in csv.reader(handle, delimiter="\t"):
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) < 2:
                continue

            utterance_id, transcript = parts[0], parts[-1]
            audio = source_dir / f"{utterance_id}.wav"
            if not audio.exists() or is_apple_double(audio):
                missing += 1
                continue

            head = utterance_id.rsplit("_", 1)
            speaker = head[0] if len(head) == 2 and head[1].isdigit() else source_dir.name

            entries.append(
                Entry(
                    utterance_id=utterance_id,
                    speaker_id=speaker,
                    transcript=transcript,
                    source_audio=audio,
                )
            )

    toned = sum(has_tone_marks(e.transcript) for e in entries)
    coverage = toned / len(entries) * 100 if entries else 0.0
    print(
        f"  {source_dir.name}: {len(entries)} utterances, "
        f"{len({e.speaker_id for e in entries})} speakers, "
        f"{coverage:.1f}% tone-marked"
        + (f", {missing} missing audio" if missing else "")
    )
    return entries


def collect_tts(root: Path) -> list[Entry]:
    """Read the per-session MP3 dataset.

    Each subdirectory is treated as one speaker: the sessions were recorded
    separately, and MFA's speaker-adaptive training needs utterances grouped by
    whoever actually produced them.
    """
    if not root.exists():
        print(f"  SKIP {root.name}: not found", file=sys.stderr)
        return []

    entries: list[Entry] = []
    session_dirs = sorted(
        d for d in root.iterdir() if d.is_dir() and not is_apple_double(d)
    )

    for session in session_dirs:
        mapping = session / "mapping.tsv"
        if not mapping.exists():
            continue

        with mapping.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                filename = (row.get("audio_filename") or "").strip()
                sentence = (row.get("sentence") or "").strip()
                key = (row.get("key") or "").strip()
                if not filename or not sentence:
                    continue

                audio = session / filename
                if not audio.exists() or is_apple_double(audio):
                    continue

                # Keys are content hashes and collide across sessions when the
                # same prompt was recorded twice; prefix with the session index.
                entries.append(
                    Entry(
                        utterance_id=f"tts_{session.name[-13:]}_{key or Path(filename).stem}",
                        speaker_id=f"tts_{session.name}",
                        transcript=sentence,
                        source_audio=audio,
                        needs_conversion=audio.suffix.lower() != ".wav",
                    )
                )

    toned = sum(has_tone_marks(e.transcript) for e in entries)
    coverage = toned / len(entries) * 100 if entries else 0.0
    print(
        f"  {root.name}: {len(entries)} utterances, "
        f"{len(session_dirs)} sessions, {coverage:.1f}% tone-marked"
    )
    if entries and coverage < 50:
        print(
            "    WARNING: these transcripts are undiacritized. They can train the\n"
            "    acoustic model only with tone-stripped phones, and can never serve\n"
            "    as a tone reference."
        )
    return entries


def ensure_ffmpeg() -> str:
    """Locate ffmpeg, which is needed to convert MP3 sources."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found, and the TTS dataset is MP3. Install it with:\n"
            "  sudo apt install ffmpeg"
        )
    return ffmpeg


def stage(entries: list[Entry], copy: bool, force: bool) -> tuple[int, int]:
    """Materialize the audio into ``data/corpus/wavs``.

    Returns:
        A ``(linked, converted)`` count.
    """
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    needs_ffmpeg = any(e.needs_conversion for e in entries)
    ffmpeg = ensure_ffmpeg() if needs_ffmpeg else None

    linked = converted = 0
    for index, entry in enumerate(entries, start=1):
        target = WAVS_DIR / f"{entry.utterance_id}.wav"

        if target.exists() or target.is_symlink():
            if not force:
                continue
            target.unlink()

        if entry.needs_conversion:
            # MFA cannot read MP3. Downmix to 16 kHz mono, which is what MFA
            # resamples to internally anyway.
            result = subprocess.run(
                [ffmpeg, "-nostdin", "-loglevel", "error", "-y",
                 "-i", str(entry.source_audio),
                 "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                 str(target)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"    ffmpeg failed on {entry.source_audio.name}: "
                      f"{result.stderr.strip()[:120]}", file=sys.stderr)
                continue
            converted += 1
        elif copy:
            shutil.copy2(entry.source_audio, target)
            linked += 1
        else:
            target.symlink_to(entry.source_audio.resolve())
            linked += 1

        if index % 500 == 0:
            print(f"    staged {index}/{len(entries)}")

    return linked, converted


def write_index(entries: list[Entry]) -> None:
    """Write the merged three-column ``line_index.tsv``."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for entry in sorted(entries, key=lambda e: e.utterance_id):
            writer.writerow([entry.utterance_id, entry.speaker_id, entry.transcript])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--include-tts", action="store_true",
        help="Also stage Yoruba-TTS-Dataset (undiacritized; acoustic training only)",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="Copy audio instead of symlinking (uses ~1.3 GB more disk)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Restage files that are already present",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be staged without writing anything",
    )
    args = parser.parse_args(argv)

    print("Scanning sources:")
    entries: list[Entry] = []
    for name in OPENSLR_SOURCES:
        entries.extend(collect_openslr(DATA_DIR / name))

    if args.include_tts:
        entries.extend(collect_tts(TTS_ROOT))

    if not entries:
        print("\nNo utterances found. Check that the datasets are extracted under data/.",
              file=sys.stderr)
        return 1

    # Duplicate ids would silently overwrite each other during staging.
    ids = Counter(e.utterance_id for e in entries)
    duplicates = [i for i, count in ids.items() if count > 1]
    if duplicates:
        print(f"\nWARNING: {len(duplicates)} duplicate utterance ids; keeping the first "
              f"of each (e.g. {duplicates[0]})")
        seen: set[str] = set()
        deduped = []
        for entry in entries:
            if entry.utterance_id not in seen:
                seen.add(entry.utterance_id)
                deduped.append(entry)
        entries = deduped

    toned = sum(has_tone_marks(e.transcript) for e in entries)
    print(f"\nTotal: {len(entries)} utterances, "
          f"{len({e.speaker_id for e in entries})} speakers, "
          f"{toned / len(entries) * 100:.1f}% tone-marked")

    if args.dry_run:
        print("\nDry run; nothing written.")
        return 0

    print(f"\nStaging into {WAVS_DIR} ...")
    linked, converted = stage(entries, copy=args.copy, force=args.force)
    write_index(entries)

    print(f"\nStaged {linked} {'copies' if args.copy else 'symlinks'}"
          + (f" and {converted} MP3 conversions" if converted else ""))
    print(f"Wrote {INDEX_PATH}")
    print("\nNext: yoruba-engine audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
