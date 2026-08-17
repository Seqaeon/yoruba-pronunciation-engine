"""Run the full assessment pipeline on a pair of recordings and print the report.

This is the end-to-end smoke test for Stages 3-7: forced alignment, pitch
extraction, tone detection, comparison, scoring, and report generation.

Two modes:

**Demo mode** (no arguments) picks a sentence that two different speakers in the
corpus both recorded, and scores one against the other. Because the pair is
usually one female and one male speaker, it also exercises the speaker
normalization -- absolute pitch differs by roughly an octave, but the tonal
pattern should still match, so a correct pipeline scores it highly.

**Explicit mode** scores your own recording against a reference::

    python scripts/demo_assess.py \\
        --reference data/corpus/wavs/yof_00295_00099759266.wav \\
        --learner   my_attempt.wav \\
        --text      "Àbùbùtán bíi omi òkun."

Learner audio should be mono WAV. Convert anything else first::

    ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 my_attempt.wav

Each assessment runs the aligner twice (once per recording), so expect roughly
30-60 seconds.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import subprocess
import sys
import wave
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.exceptions import YorubaEngineError  # noqa: E402
from yoruba_engine.logging_utils import configure_logging, get_logger  # noqa: E402
from yoruba_engine.pipeline import AssessmentPipeline  # noqa: E402

logger = get_logger("demo")


def find_shared_sentence(config: EngineConfig, seed: int) -> tuple[Path, Path, str]:
    """Pick a sentence two different speakers both recorded.

    Returns:
        ``(reference_audio, learner_audio, transcript)``.

    Raises:
        SystemExit: If the corpus has no such sentence, or the audio is missing.
    """
    index = config.line_index
    if not index.exists():
        raise SystemExit(
            f"No corpus index at {index}. Run scripts/prepare_dataset.py first."
        )

    by_text: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                continue
            by_text[row[2].strip()].append((row[0], row[1]))

    candidates = [
        (text, entries) for text, entries in by_text.items()
        if len({speaker for _, speaker in entries}) > 1
    ]
    if not candidates:
        raise SystemExit("No sentence in this corpus was recorded by two speakers.")

    random.Random(seed).shuffle(candidates)

    for text, entries in candidates:
        # Prefer a cross-sex pair: it is the harder case and the one that proves
        # speaker normalization works.
        female = next((u for u, s in entries if s.startswith("yof")), None)
        male = next((u for u, s in entries if s.startswith("yom")), None)
        chosen = (female, male) if female and male else (entries[0][0], entries[1][0])

        reference = config.audio_dir / f"{chosen[0]}.wav"
        learner = config.audio_dir / f"{chosen[1]}.wav"
        if reference.exists() and learner.exists():
            return reference, learner, text

    raise SystemExit("Found shared sentences but their audio files are missing.")


def normalize_audio(path: Path, workdir: Path) -> Path:
    """Return a mono 16-bit WAV, converting the input if necessary.

    Phone and messaging apps hand out stereo, 24-bit, or compressed audio. MFA
    wants mono PCM WAV, and a mismatch here produces confusing downstream
    failures rather than a clear message, so convert up front.
    """
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                if handle.getnchannels() == 1 and handle.getsampwidth() == 2:
                    return path
        except wave.Error:
            pass  # not plain PCM; fall through to conversion

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            f"{path.name} is not mono 16-bit WAV and ffmpeg is not installed.\n"
            "Install it (sudo apt install ffmpeg) or convert the file yourself:\n"
            f"  ffmpeg -i '{path}' -ar 16000 -ac 1 -sample_fmt s16 converted.wav"
        )

    workdir.mkdir(parents=True, exist_ok=True)
    converted = workdir / f"{path.stem}_mono16.wav"
    result = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(converted)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg could not convert {path}:\n{result.stderr.strip()[:300]}")

    print(f"Converted {path.name} to mono 16 kHz WAV -> {converted}")
    return converted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, help="Native-speaker recording")
    parser.add_argument("--learner", type=Path, help="Learner recording")
    parser.add_argument("--text", help="Target text, with tone marks")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "output" / "reports")
    parser.add_argument("--seed", type=int, default=0, help="Which demo pair to pick")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Use the threshold rules instead of the trained classifier")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    if not args.verbose:
        for noisy in ("yoruba_engine.pitch", "yoruba_engine.tone", "yoruba_engine.mfa"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    config = EngineConfig()

    supplied = [args.reference, args.learner, args.text]
    if any(supplied) and not all(supplied):
        raise SystemExit("--reference, --learner and --text must be given together.")

    if all(supplied):
        reference, learner, text = args.reference, args.learner, args.text
        for path in (reference, learner):
            if not path.exists():
                raise SystemExit(f"Audio file not found: {path}")
        workdir = PROJECT_ROOT / "output" / "converted"
        reference = normalize_audio(reference, workdir)
        learner = normalize_audio(learner, workdir)
    else:
        reference, learner, text = find_shared_sentence(config, args.seed)
        print("Demo mode: scoring one speaker's recording against another's.\n")

    print(f"Reference : {reference.name}")
    print(f"Learner   : {learner.name}")
    print(f"Text      : {text}\n")
    print("Aligning and analysing (this runs the aligner twice, ~30-60 s) ...\n")

    pipeline = AssessmentPipeline(
        config, use_tone_classifier=False if args.no_classifier else None
    )
    detector = "threshold rules" if pipeline.tone_analyzer.classifier is None else "trained classifier"
    print(f"Tone detector: {detector}\n")

    try:
        result, paths = pipeline.assess_and_report(
            reference_audio=reference,
            learner_audio=learner,
            target_text=text,
            output_dir=args.out,
        )
    except YorubaEngineError as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(pipeline.reporter.to_text(
        result,
        reference=getattr(pipeline, "_reference_cache", None),
        learner=getattr(pipeline, "_learner_cache", None),
    ))

    for name, path in paths.items():
        print(f"Wrote {name:<5} report: {path}")

    return 0 if result.overall_score >= config.scoring.pass_threshold else 1


if __name__ == "__main__":
    sys.exit(main())
