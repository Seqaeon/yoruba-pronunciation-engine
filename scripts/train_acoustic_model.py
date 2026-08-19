"""Train a Yoruba acoustic model on every staged corpus.

The shipped model was trained on Kaggle over 37.93 hours from 2,047 speakers:
michsethowusu (24.0 h), Waxal (6.30 h), local TTS (4.16 h) and OpenSLR (3.47 h).
Its weakness is not volume but admission policy -- it was built with
``require_tone_marks`` disabled, so it learned from Waxal's undiacritized lines,
which teaches the model that ``ọ`` and ``o`` are the same vowel.

Aligning YECS with it left 62% of the audio unlabelled and put the median vowel
at 30 ms against 80 ms on SLR86, so its coverage does not extend to this speech.

Training does not use the old model. MFA starts flat, initialising monophones
from the audio itself and iterating align/re-estimate through triphone,
LDA+MLLT and speaker-adaptive training, so a bad one-pass alignment from a
mismatched model cannot propagate into the result.

Whether this actually helps is an empirical question, answered afterwards by
re-running the vowel-duration comparison against SLR86. If YECS-trained
alignment does not bring the median vowel near 80 ms, the extra data is not
earning its place and the old model should stay.

**Progress is streamed, not captured.** MFA emits its progress as it goes, but
running it through ``subprocess.run(capture_output=True)`` holds every line in a
pipe until the process exits, so a run that takes hours writes nothing until it
is over -- and a machine that reboots mid-run leaves a log with three lines in
it and no indication of how far it got. Output is therefore read line by line
and written out immediately, and a monitor thread reports stage completion
independently by watching the temporary directory, which survives even if MFA's
own output format changes.

**The bar and the log go to different places.** A progress bar redraws itself
with a carriage return, which a terminal turns into an updating bar and a file
turns into one padded line per redraw. Writing both to the same log made it
unreadable, so the bar renders into a small status file that is overwritten in
place, while the log keeps plain timestamped lines:

    tail -f output/training.log        # what MFA is doing
    watch -n5 cat output/training_status.txt   # how far along it is

    MFA_JOBS=12 python scripts/train_acoustic_model.py
    MFA_JOBS=12 python scripts/train_acoustic_model.py --resume
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tqdm import tqdm  # noqa: E402

from yoruba_engine.alignment.mfa import MFAAligner  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402

logger = logging.getLogger("train")

#: The training stages MFA works through, in order. Each leaves a ``final.mdl``
#: in its own subdirectory of the temporary directory when it completes, which
#: is how progress is measured -- reading the filesystem rather than parsing
#: output means the bar stays honest even if MFA's logging changes.
#:
#: The parallel ``*_ali`` directories are alignment passes that reuse their
#: parent's model, so they are not counted separately.
STAGES = (
    "monophone",
    "triphone",
    "lda",
    "sat",
    "pronunciation_probabilities",
    "sat_2",
    "sat_3",
    "pronunciation_probabilities_2",
    "sat_4",
)

#: MFA reports within-stage progress as "monophone - Iteration 7 of 40".
ITERATION = re.compile(r"([A-Za-z][\w]*)\s*-\s*Iteration\s+(\d+)\s+of\s+(\d+)")


def stage_root(temporary_directory: Path, corpus: Path) -> Path:
    """Where MFA puts its per-stage directories."""
    return temporary_directory / corpus.name


def completed_stages(root: Path) -> list[str]:
    return [s for s in STAGES if (root / s / "final.mdl").exists()]


def current_stage(root: Path) -> str:
    """The stage most recently written to, as a fallback progress signal."""
    candidates = [(d.stat().st_mtime, d.name) for d in root.glob("*") if d.is_dir()]
    if not candidates:
        return "starting"
    return max(candidates)[1]


class StatusFile:
    """A file-like target that keeps only a progress bar's latest frame.

    tqdm redraws by emitting a carriage return and rewriting the line, which is
    right for a terminal and useless in a file. Each frame replaces the last, so
    the file always holds the current state and nothing else.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.latest = ""
        self.extra = ""
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        text = text.strip("\r\n")
        if text:
            with self.lock:
                self.latest = text
                self._flush()

    def note(self, text: str) -> None:
        with self.lock:
            self.extra = text
            self._flush()

    def _flush(self) -> None:
        body = self.latest + ("\n" + self.extra if self.extra else "")
        self.path.write_text(body + "\n", encoding="utf-8")

    def flush(self) -> None:
        return None


class Monitor(threading.Thread):
    """Reports progress from the filesystem, independent of MFA's output."""

    def __init__(self, root: Path, aligned: Path, bar: tqdm,
                 status: StatusFile, every: float = 60.0) -> None:
        super().__init__(daemon=True)
        self.root = root
        self.aligned = aligned
        self.bar = bar
        self.status = status
        self.every = every
        self.started = time.time()
        self.stop = threading.Event()
        self.seen: set[str] = set()

    def run(self) -> None:
        while not self.stop.wait(self.every):
            self.report()

    def report(self) -> None:
        elapsed = time.time() - self.started
        done = completed_stages(self.root) if self.root.exists() else []
        for name in done:
            if name not in self.seen:
                self.seen.add(name)
                print(f"{time.strftime('%H:%M:%S')} | [STAGE COMPLETE] {name} "
                      f"({len(self.seen)}/{len(STAGES)}) "
                      f"after {elapsed / 60:.0f} min", flush=True)
                self.bar.n = len(self.seen)
        self.bar.refresh()
        grids = sum(1 for _ in self.aligned.rglob("*.TextGrid")) \
            if self.aligned.exists() else 0
        where = current_stage(self.root) if self.root.exists() else "starting"
        self.status.note(
            f"elapsed {elapsed / 3600:5.2f} h | "
            f"stages done {len(self.seen)}/{len(STAGES)} | "
            f"working in {where} | TextGrids {grids:,}")
        self.bar.set_postfix_str(where)
        self.bar.refresh()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path,
                        default=PROJECT_ROOT / "data" / "train_corpus")
    parser.add_argument("--dictionary", type=Path,
                        default=PROJECT_ROOT / "data" / "dictionary" / "train.dict")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "models" / "yoruba_acoustic_v2.zip")
    parser.add_argument("--aligned", type=Path,
                        default=PROJECT_ROOT / "output" / "train_aligned",
                        help="where to write the alignments training produces")
    parser.add_argument("--temporary-directory", type=Path,
                        default=PROJECT_ROOT / "output" / "mfa_train_tmp")
    parser.add_argument("--status-file", type=Path,
                        default=PROJECT_ROOT / "output" / "training_status.txt")
    parser.add_argument("--resume", action="store_true",
                        help="keep the temporary directory and continue from the "
                             "last completed stage instead of starting over")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="pass -v to MFA (default on)")
    parser.add_argument("--heartbeat", type=float, default=60.0,
                        help="seconds between progress reports (default 60)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    config = EngineConfig()
    aligner = MFAAligner(config)
    # The acoustic model is what we are producing, so do not require one.
    aligner.check_available(require_acoustic_model=False)

    utterances = sum(1 for _ in args.corpus.rglob("*.lab"))
    speakers = sum(1 for p in args.corpus.iterdir() if p.is_dir())
    logger.info("Training on %d utterances from %d speakers", utterances, speakers)
    logger.info("MFA jobs: %d", config.mfa.num_jobs)

    root = stage_root(args.temporary_directory, args.corpus)
    if args.resume:
        already = completed_stages(root) if root.exists() else []
        logger.info("Resuming; %d stage(s) already complete: %s",
                    len(already), ", ".join(already) or "none")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.aligned.mkdir(parents=True, exist_ok=True)

    command = [
        aligner.mfa.executable, "train",
        str(args.corpus), str(args.dictionary), str(args.out),
        "--output_directory", str(args.aligned),
        "--num_jobs", str(config.mfa.num_jobs),
        "--temporary_directory", str(args.temporary_directory),
        "--no_clean" if args.resume else "--clean",
    ]
    if args.verbose:
        command.append("-v")
    logger.info("Running: %s", " ".join(command))

    status = StatusFile(args.status_file)
    bar = tqdm(total=len(STAGES), unit="stage", ascii=True, ncols=100,
               mininterval=1.0, dynamic_ncols=False, file=status, leave=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages "
                          "[{elapsed}<{remaining}] {postfix}")
    if args.resume and root.exists():
        bar.n = len(completed_stages(root))
        bar.refresh()

    monitor = Monitor(root, args.aligned, bar, status, args.heartbeat)
    monitor.start()

    started = time.time()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=aligner._environment(),
    )
    try:
        for line in process.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            print(f"{time.strftime('%H:%M:%S')} | {line}", flush=True)
            match = ITERATION.search(line)
            if match:
                stage, index, total = (match.group(1), int(match.group(2)),
                                       int(match.group(3)))
                bar.set_postfix_str(f"{stage} iter {index}/{total}")
                bar.refresh()
    finally:
        returncode = process.wait()
        monitor.stop.set()
        monitor.report()
        bar.close()

    elapsed = (time.time() - started) / 3600
    if returncode != 0:
        logger.error("mfa train failed (exit %d) after %.2f h", returncode, elapsed)
        return 1

    grids = len(list(args.aligned.rglob("*.TextGrid")))
    logger.info("Trained in %.2f h. Model: %s", elapsed, args.out)
    logger.info("%d TextGrids written to %s (%.1f%% of utterances aligned)",
                grids, args.aligned, 100 * grids / max(utterances, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
