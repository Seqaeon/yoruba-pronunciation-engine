"""Stage 3: Montreal Forced Aligner integration.

MFA is driven as a subprocess rather than through its Python API, because its
internal API changes between minor versions while the CLI contract is stable.

The key operation for the learning app is :meth:`MFAAligner.align_single`, which
aligns one learner recording against a *known* target text. This is forced
alignment used as verification: we already know what the learner was asked to
say, so no ASR is required. The alignment's own confidence tells us whether they
plausibly said it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..config import EngineConfig
from ..exceptions import AlignmentError, MFANotAvailableError
from ..logging_utils import get_logger
from ..datatypes import Alignment
from .textgrid import alignment_from_textgrid

logger = get_logger("mfa")


class MFAAligner:
    """Thin, reusable wrapper around the ``mfa`` command-line tool."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.mfa = config.mfa

    # ------------------------------------------------------------------
    # Environment checks
    # ------------------------------------------------------------------

    def check_available(self, require_acoustic_model: bool = True) -> None:
        """Verify the MFA executable and required assets are present.

        Args:
            require_acoustic_model: Set ``False`` for operations that run before
                a model exists, such as corpus validation prior to training.

        Raises:
            MFANotAvailableError: With actionable guidance on what is missing.
        """
        if shutil.which(self.mfa.executable) is None:
            raise MFANotAvailableError(
                f"'{self.mfa.executable}' not found. Install it with:\n"
                "  micromamba create -y -p .mfa-env -c conda-forge "
                "python=3.11 montreal-forced-aligner\n"
                "(see INSTALL.md), or set the MFA_BIN environment variable."
            )
        if not self.mfa.dictionary.exists():
            raise MFANotAvailableError(
                f"Pronunciation dictionary missing at {self.mfa.dictionary}. "
                "Run Stage 2 (`yoruba-engine dictionary`) first."
            )
        if require_acoustic_model and not self.mfa.acoustic_model.exists():
            raise MFANotAvailableError(
                f"Acoustic model missing at {self.mfa.acoustic_model}. "
                "There is no reliable pretrained Yoruba model; train one with "
                "notebooks/train_mfa_yoruba.ipynb."
            )
        if require_acoustic_model:
            self.check_phone_sets_match()

    def check_phone_sets_match(self) -> None:
        """Verify the dictionary's phones are known to the acoustic model.

        This is the one mismatch MFA will not report. Aligning a tone-suffixed
        dictionary (``a_H``) against a model trained on tone-stripped phones
        (``a``) exits zero and writes TextGrids in which every segment carries
        the same meaningless label -- observed in practice as a whole corpus
        aligned to the single phone ``m``.

        Raises:
            AlignmentError: If the dictionary uses phones the model never saw.
        """
        import re  # noqa: PLC0415
        import zipfile  # noqa: PLC0415

        try:
            with zipfile.ZipFile(self.mfa.acoustic_model) as archive:
                name = next(n for n in archive.namelist() if n.endswith("phones.txt"))
                model_phones = {
                    token for token in archive.read(name).decode("utf-8", "ignore").split()
                    if not token.isdigit()
                }
        except (zipfile.BadZipFile, StopIteration, OSError) as exc:
            logger.debug("Could not read phones from %s: %s", self.mfa.acoustic_model, exc)
            return          # not fatal; MFA will do its own checks

        numeric = re.compile(r"^-?\d+(\.\d+)?$")
        dictionary_phones: set[str] = set()
        with self.mfa.dictionary.open(encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                # MFA's probabilistic format inserts numeric columns before the
                # phones, so skip anything that parses as a number.
                for field in fields[1:]:
                    if not numeric.match(field.strip()):
                        dictionary_phones.update(field.split())

        unknown = dictionary_phones - model_phones
        if unknown:
            suffixed = {p for p in unknown if re.search(r"_[HML]$", p)}
            hint = (
                "\nThe dictionary uses tone-suffixed phones but the model was "
                "trained on tone-stripped ones. Rebuild it with:\n"
                "  yoruba-engine dictionary --tone-stripped"
                if suffixed else
                "\nRebuild the dictionary from the same vocabulary the model was "
                "trained on, or use the dictionary exported alongside the model."
            )
            raise AlignmentError(
                f"Dictionary and acoustic model disagree on the phone set: "
                f"{len(unknown)} phones are unknown to the model, e.g. "
                f"{sorted(unknown)[:6]}.{hint}"
            )

        logger.debug("Phone sets agree (%d dictionary phones)", len(dictionary_phones))

    def version(self) -> str:
        """Return the installed MFA version string."""
        result = self._run([self.mfa.executable, "version"], check=False)
        return result.stdout.strip() or "unknown"

    # ------------------------------------------------------------------
    # Corpus operations
    # ------------------------------------------------------------------

    def validate_corpus(self, corpus_dir: Path | None = None) -> str:
        """Run ``mfa validate`` and return its output.

        This surfaces out-of-vocabulary words and unreadable audio before a long
        alignment or training run wastes time on them.

        The acoustic model is passed only if it exists, so this can be run
        before training to check the corpus and dictionary on their own.
        """
        self.check_available(require_acoustic_model=False)
        corpus = corpus_dir or self.config.corpus_dir

        command = [
            self.mfa.executable, "validate",
            str(corpus), str(self.mfa.dictionary),
        ]
        if self.mfa.acoustic_model.exists():
            command.append(str(self.mfa.acoustic_model))
        command += ["--num_jobs", str(self.mfa.num_jobs), "--clean"]

        result = self._run(command, check=False)
        logger.info("MFA validation finished with code %d", result.returncode)
        return result.stdout + result.stderr

    def align_corpus(
        self, corpus_dir: Path, output_dir: Path, dictionary: Path | None = None,
        temporary_directory: Path | None = None,
    ) -> Path:
        """Align an entire corpus, producing one TextGrid per utterance.

        Args:
            corpus_dir: Directory of paired ``.wav``/``.lab`` files.
            output_dir: Where TextGrids are written.
            dictionary: Overrides the configured dictionary. Used by
                :meth:`align_single`, which generates one covering exactly the
                words of the request.
            temporary_directory: Where MFA should work. Passing this also
                disables its cleanup, so the per-phone acoustic scores survive
                for :mod:`yoruba_engine.alignment.gop` to read.

        Returns:
            ``output_dir``.

        Raises:
            AlignmentError: If MFA exits non-zero.
        """
        self.check_available()
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.mfa.executable, "align",
            str(corpus_dir), str(dictionary or self.mfa.dictionary),
            str(self.mfa.acoustic_model),
            str(output_dir),
            "--num_jobs", str(self.mfa.num_jobs),
            "--beam", str(self.mfa.beam),
            "--retry_beam", str(self.mfa.retry_beam),
        ]
        if temporary_directory is not None:
            # MFA deletes its working directory on --clean, and the per-phone
            # acoustic scores live there. Keeping it is the only way to read
            # them; the caller is responsible for removing it afterwards.
            command += ["--temporary_directory", str(temporary_directory), "--no_clean"]
        else:
            command.append("--clean")
        if self.mfa.single_speaker:
            command.append("--single_speaker")

        result = self._run(command, check=False)
        if result.returncode != 0:
            raise AlignmentError(
                f"mfa align failed (exit {result.returncode}).\n{result.stderr[-2000:]}"
            )

        produced = list(output_dir.rglob("*.TextGrid"))
        logger.info("Aligned corpus: %d TextGrids in %s", len(produced), output_dir)
        if not produced:
            raise AlignmentError(
                f"mfa align reported success but wrote no TextGrids to {output_dir}"
            )
        return output_dir

    def model_phone_set(self) -> set[str]:
        """The phones the acoustic model was trained on, read from its archive.

        Cached, because it is consulted on every alignment request.
        """
        if getattr(self, "_phone_set", None) is None:
            import zipfile  # noqa: PLC0415

            try:
                with zipfile.ZipFile(self.mfa.acoustic_model) as archive:
                    name = next(n for n in archive.namelist() if n.endswith("phones.txt"))
                    self._phone_set = {
                        line.split()[0]
                        for line in archive.read(name).decode("utf-8").splitlines()
                        if line.split()
                    }
            except Exception as exc:
                logger.warning("Could not read the model's phone set: %s", exc)
                self._phone_set = set()
        return self._phone_set

    def request_dictionary(self, transcript: str, destination: Path) -> Path:
        """Write a dictionary covering exactly the words of one transcript.

        MFA drops any utterance containing a word its dictionary does not
        define, so a prompt with one unlisted word yields no alignment at all
        rather than a partial one. The shipped dictionary is finite -- it covers
        49% of the word types in a held-out Yoruba corpus -- which makes that a
        routine failure rather than an edge case.

        Yoruba orthography is close enough to phonemic that the rules in
        :mod:`yoruba_engine.g2p` cover essentially the whole language (zero
        failures across the 3,413-word corpus vocabulary), so anything the
        shipped dictionary lacks is generated here instead.

        Entries already present win, which preserves hand-written overrides and
        every pronunciation variant. Their probability columns are dropped: MFA
        accepts a plain two-column dictionary, and pronunciation probabilities
        earn little on a single utterance.
        """
        from ..exceptions import G2PError  # noqa: PLC0415
        from ..g2p.dictionary import grapheme_to_phonemes  # noqa: PLC0415
        from ..g2p.orthography import normalize, tokenize  # noqa: PLC0415

        def pronunciation(rest: str) -> str:
            # A trained dictionary carries probability columns between the word
            # and its phones; the phones are always the last column.
            return rest.split("\t")[-1].strip()

        known: dict[str, list[str]] = {}
        if self.mfa.dictionary.exists():
            for line in self.mfa.dictionary.read_text(encoding="utf-8").splitlines():
                if "\t" not in line:
                    continue
                word, rest = line.split("\t", 1)
                known.setdefault(word, []).append(pronunciation(rest))

        # The model decides whether phones carry tone suffixes. Generating
        # `a_H` for a model trained on `a` does not fail loudly -- MFA aligns
        # against whatever it is given -- so match the model rather than assume.
        phones = self.model_phone_set()
        strip_tone = bool(phones) and not any("_" in p for p in phones)

        entries: dict[str, list[str]] = {}
        generated, failed = [], []
        for token in tokenize(transcript, keep_noise=True):
            word = normalize(token)
            if not word or word in entries:
                continue
            if word in known:
                entries[word] = known[word]
                continue
            try:
                symbols = [p.symbol for p in grapheme_to_phonemes(word)]
            except G2PError as exc:
                failed.append(word)
                logger.warning("No pronunciation for %r: %s", word, exc)
                continue
            if strip_tone:
                import re  # noqa: PLC0415

                symbols = [re.sub(r"_[HML]$", "", s) for s in symbols]
            unknown = {s for s in symbols if phones and s not in phones}
            if unknown:
                failed.append(word)
                logger.warning(
                    "Generated pronunciation for %r uses phones the model does "
                    "not know (%s); the aligner would ignore it",
                    word, ", ".join(sorted(unknown)),
                )
                continue
            entries[word] = [" ".join(symbols)]
            generated.append(word)

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for word in sorted(entries):
                for variant in entries[word]:
                    handle.write(f"{word}\t{variant}\n")

        if generated:
            logger.info("Generated %d pronunciation(s) not in the shipped "
                        "dictionary: %s", len(generated), ", ".join(generated[:8]))
        if failed:
            logger.warning("%d word(s) have no usable pronunciation; MFA will "
                           "drop this utterance: %s", len(failed), ", ".join(failed))
        return destination

    def align_single(
        self,
        audio_path: Path,
        transcript: str,
        utterance_id: str | None = None,
        keep_workdir: bool = False,
        collect_gop: bool = False,
    ) -> Alignment:
        """Align one recording against a known transcript.

        This is the per-request entry point for the learning app. It builds a
        one-file corpus in a temporary directory, since MFA is corpus-oriented.

        Args:
            audio_path: The recording to align.
            transcript: What the speaker was asked to say, with tone marks.
            utterance_id: Identifier for the result; defaults to the file stem.
            keep_workdir: Leave the temp corpus on disk for debugging.
            collect_gop: Also read the acoustic model's per-phone scores, which
                requires keeping MFA's working directory during the run. Costs
                nothing extra to align -- the scores are computed either way --
                so this is off only because reading them needs the artifacts.

        Returns:
            Word- and phone-level alignment.

        Raises:
            AlignmentError: If the audio is missing or MFA produces no TextGrid.
        """
        if not audio_path.exists():
            raise AlignmentError(f"Audio file not found: {audio_path}")

        self.check_available()
        utt_id = utterance_id or audio_path.stem
        workdir = Path(tempfile.mkdtemp(prefix="yoruba_align_"))

        try:
            corpus = workdir / "corpus"
            corpus.mkdir(parents=True)

            # MFA's simplest corpus layout: matching .wav and .lab basenames.
            staged_audio = corpus / f"{utt_id}{audio_path.suffix}"
            shutil.copy2(audio_path, staged_audio)
            (corpus / f"{utt_id}.lab").write_text(transcript, encoding="utf-8")

            aligned = workdir / "aligned"
            mfa_temporary = (workdir / "mfa") if collect_gop else None
            self.align_corpus(
                corpus, aligned,
                dictionary=self.request_dictionary(transcript, workdir / "request.dict"),
                temporary_directory=mfa_temporary,
            )

            textgrids = list(aligned.rglob(f"{utt_id}.TextGrid"))
            if not textgrids:
                textgrids = list(aligned.rglob("*.TextGrid"))
            if not textgrids:
                raise AlignmentError(
                    f"No TextGrid produced for {utt_id}. The utterance may have "
                    "failed to align -- check for out-of-vocabulary words."
                )

            alignment = alignment_from_textgrid(textgrids[0], utterance_id=utt_id)

            if mfa_temporary is not None:
                from .gop import read_scores  # noqa: PLC0415

                # MFA names its working subdirectory after the corpus directory.
                scores = read_scores(mfa_temporary, corpus.name)
                alignment.phone_scores = scores.get(utt_id) or next(
                    iter(scores.values()), [])
                if not alignment.phone_scores:
                    logger.info("No per-phone scores recovered for %s", utt_id)

            return alignment
        finally:
            if keep_workdir:
                logger.info("Left alignment workdir at %s", workdir)
            else:
                shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Corpus preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_corpus(
        utterances: list,
        destination: Path,
        by_speaker: bool = True,
    ) -> Path:
        """Lay out utterances in the directory structure MFA expects.

        MFA infers speaker identity from directory structure, which matters for
        speaker-adaptive training. With ``by_speaker=True`` each speaker gets a
        subdirectory.

        Args:
            utterances: :class:`~yoruba_engine.types.Utterance` objects.
            destination: Root of the corpus to create.
            by_speaker: Group files into per-speaker subdirectories.

        Returns:
            ``destination``.
        """
        destination.mkdir(parents=True, exist_ok=True)
        for utterance in utterances:
            target_dir = destination / utterance.speaker_id if by_speaker else destination
            target_dir.mkdir(parents=True, exist_ok=True)

            audio_target = target_dir / f"{utterance.utterance_id}{utterance.audio_path.suffix}"
            if not audio_target.exists():
                shutil.copy2(utterance.audio_path, audio_target)
            (target_dir / f"{utterance.utterance_id}.lab").write_text(
                utterance.transcript, encoding="utf-8"
            )

        logger.info("Prepared MFA corpus with %d utterances at %s", len(utterances), destination)
        return destination

    # ------------------------------------------------------------------

    def _environment(self) -> dict[str, str]:
        """Build the environment MFA needs.

        MFA shells out to OpenFST and Kaldi binaries (``fstcompile``,
        ``gmm-align-compiled``, ...) that live beside the ``mfa`` executable in
        its conda environment. Invoking ``mfa`` by absolute path is therefore not
        enough -- without its ``bin`` directory on ``PATH`` it fails with
        ``ThirdpartyError: Could not find 'fstcompile'``.

        Prepending the directory here means callers never have to activate the
        conda environment.
        """
        environment = os.environ.copy()

        executable = shutil.which(self.mfa.executable) or self.mfa.executable
        bin_dir = Path(executable).resolve().parent
        if bin_dir.is_dir():
            environment["PATH"] = f"{bin_dir}{os.pathsep}{environment.get('PATH', '')}"

        return environment

    def _run(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Execute a subprocess, logging the command and capturing output."""
        logger.debug("Running: %s", " ".join(command))
        try:
            return subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check,
                timeout=None,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise MFANotAvailableError(f"Could not execute {command[0]}: {exc}") from exc
        except subprocess.CalledProcessError as exc:
            raise AlignmentError(f"{command[0]} failed: {exc.stderr}") from exc
