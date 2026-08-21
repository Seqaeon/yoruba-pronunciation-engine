"""Command-line interface.

Subcommands map one-to-one onto pipeline stages so each can be run and inspected
independently::

    yoruba-engine audit
    yoruba-engine dictionary
    yoruba-engine prepare-corpus --out data/mfa_corpus
    yoruba-engine validate
    yoruba-engine assess --reference ref.wav --learner learner.wav --text "òrí"
    yoruba-engine health
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .audit.dataset_audit import DatasetAuditor
from .config import DEFAULT_CONFIG, EngineConfig
from .exceptions import YorubaEngineError
from .g2p.dictionary import PronunciationDictionary
from .logging_utils import configure_logging, get_logger
from .pipeline import AssessmentPipeline

logger = get_logger("cli")


def _config_from_args(args: argparse.Namespace) -> EngineConfig:
    """Apply CLI overrides onto the default configuration."""
    config = EngineConfig()
    if getattr(args, "corpus", None):
        config.corpus_dir = Path(args.corpus)
        config.line_index = config.corpus_dir / "line_index.tsv"
        config.audio_dir = config.corpus_dir / "wavs"
    if getattr(args, "output", None):
        config.output_dir = Path(args.output)
    return config


def cmd_audit(args: argparse.Namespace) -> int:
    """Stage 1: audit the corpus and write the four deliverables."""
    config = _config_from_args(args)
    auditor = DatasetAuditor(config)
    report = auditor.audit()
    paths = auditor.write_all(report)

    print(f"\nMatched {len(report.matched)} utterances "
          f"({report.total_duration_hours:.2f} h), "
          f"{len(report.word_frequency)} unique words.")
    print(f"Tone-mark coverage: {report.tone_mark_coverage * 100:.1f}%")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    if report.tone_mark_coverage < 0.5:
        print("\nBLOCKER: corpus is largely undiacritized; tone comparison is not viable.")
        return 2
    return 0


def cmd_dictionary(args: argparse.Namespace) -> int:
    """Stage 2: generate the MFA pronunciation dictionary."""
    config = _config_from_args(args)

    vocab_path = Path(args.vocabulary) if args.vocabulary else config.output_dir / "vocabulary.txt"
    if not vocab_path.exists():
        print(f"Vocabulary not found at {vocab_path}. Run `audit` first.", file=sys.stderr)
        return 1

    words = [w.strip() for w in vocab_path.read_text(encoding="utf-8").splitlines() if w.strip()]

    overrides_path = Path(args.overrides) if args.overrides else (
        config.mfa.dictionary.parent / "overrides.tsv"
    )
    dictionary = PronunciationDictionary.load_overrides(overrides_path)
    dictionary.build(words)

    if args.tone_stripped:
        dictionary.strip_tone_suffixes()
        logger.info("Wrote tone-stripped phones (%d in inventory)",
                    len(dictionary.phone_inventory()))

    output = Path(args.out) if args.out else config.mfa.dictionary
    dictionary.write(output)
    dictionary.write_failures(config.output_dir / "g2p_failures.csv")

    print(f"\nDictionary: {len(dictionary.entries)} entries -> {output}")
    print(f"Phone inventory: {len(dictionary.phone_inventory())} phones")
    if dictionary.failures:
        print(f"G2P failures: {len(dictionary.failures)} "
              f"(see {config.output_dir / 'g2p_failures.csv'})")
    return 0


def cmd_prepare_corpus(args: argparse.Namespace) -> int:
    """Lay the corpus out in the directory structure MFA expects."""
    config = _config_from_args(args)
    report = DatasetAuditor(config).audit()
    destination = Path(args.out)
    from .alignment.mfa import MFAAligner  # noqa: PLC0415

    MFAAligner.prepare_corpus(report.matched, destination, by_speaker=not args.flat)
    print(f"Prepared {len(report.matched)} utterances at {destination}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Stage 3: run MFA's own corpus validation."""
    config = _config_from_args(args)
    pipeline = AssessmentPipeline(config)
    corpus = Path(args.corpus_dir) if args.corpus_dir else config.corpus_dir
    print(pipeline.aligner.validate_corpus(corpus))
    return 0


def cmd_assess(args: argparse.Namespace) -> int:
    """Stages 3-7: assess a learner recording against a reference."""
    config = _config_from_args(args)

    # Forced alignment cannot report what a learner said, only how badly the
    # prompt fits, so substitutions have to be recovered by matching vowels
    # against a 9,092-entry dictionary. Whisper has no lexicon to be limited by:
    # it named the substituted word in all four test recordings, including ọkun,
    # which occurs zero times in 209,265 corpus tokens. It costs about 2.8 s on
    # four CPU threads, and its tone marks are unreliable -- tone still comes
    # from pitch, and vowel spans still from alignment.
    provider = None
    if getattr(args, "transcribe", False):
        from .asr.providers import WhisperTranscriptProvider  # noqa: PLC0415

        provider = WhisperTranscriptProvider()
    pipeline = AssessmentPipeline(config, transcript_provider=provider)

    result, paths = pipeline.assess_and_report(
        reference_audio=Path(args.reference),
        learner_audio=Path(args.learner),
        target_text=args.text,
        learner_text=args.learner_text,
    )

    print(pipeline.reporter.to_text(
        result,
        reference=getattr(pipeline, "_reference_cache", None),
        learner=getattr(pipeline, "_learner_cache", None),
    ))
    for name, path in paths.items():
        print(f"Wrote {name} report: {path}")

    return 0 if result.overall_score >= config.scoring.pass_threshold else 1


def cmd_health(args: argparse.Namespace) -> int:
    """Report dependency and asset availability."""
    config = _config_from_args(args)
    status = AssessmentPipeline(config).health_check()
    print(json.dumps(status, indent=2))
    ready = status["mfa"]["available"] and status["acoustic_model"]["exists"]  # type: ignore[index]
    return 0 if ready else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="yoruba-engine",
        description="Yoruba pronunciation assessment engine",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", type=Path, help="Also write logs to this file")
    parser.add_argument("--corpus", help="Corpus directory (contains line_index.tsv and wavs/)")
    parser.add_argument("--output", help="Output directory for reports and deliverables")

    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Stage 1: audit the dataset")
    audit.set_defaults(func=cmd_audit)

    dictionary = subparsers.add_parser("dictionary", help="Stage 2: build the MFA dictionary")
    dictionary.add_argument("--vocabulary", help="Path to vocabulary.txt")
    dictionary.add_argument("--overrides", help="Path to manual overrides TSV")
    dictionary.add_argument("--out", help="Where to write the dictionary")
    dictionary.add_argument(
        "--tone-stripped", action="store_true",
        help="Emit phones without _H/_M/_L suffixes. Must match how the acoustic "
             "model was trained -- a mismatch makes MFA produce nonsense silently.",
    )
    dictionary.set_defaults(func=cmd_dictionary)

    prepare = subparsers.add_parser("prepare-corpus", help="Lay out an MFA-format corpus")
    prepare.add_argument("--out", required=True, help="Destination directory")
    prepare.add_argument("--flat", action="store_true", help="Do not group by speaker")
    prepare.set_defaults(func=cmd_prepare_corpus)

    validate = subparsers.add_parser("validate", help="Stage 3: run MFA corpus validation")
    validate.add_argument("--corpus-dir", help="MFA-format corpus directory")
    validate.set_defaults(func=cmd_validate)

    assess = subparsers.add_parser("assess", help="Stages 3-7: score a learner recording")
    assess.add_argument(
        "--transcribe", action="store_true",
        help="Recognise what the learner actually said with Whisper, instead of "
             "assuming they read the prompt. Names substituted words the "
             "dictionary does not contain, at about 2.8 s per assessment.")
    assess.add_argument("--reference", required=True, help="Native-speaker audio")
    assess.add_argument("--learner", required=True, help="Learner audio")
    assess.add_argument("--text", required=True, help="Target text, with tone marks")
    assess.add_argument("--learner-text", help="Learner transcript, if known")
    assess.set_defaults(func=cmd_assess)

    health = subparsers.add_parser("health", help="Check dependencies and assets")
    health.set_defaults(func=cmd_health)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code. ``0`` success, ``1`` failure or below-threshold score,
        ``2`` a blocking dataset problem.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
    )

    try:
        return args.func(args)
    except YorubaEngineError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
