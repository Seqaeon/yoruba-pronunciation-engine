"""Stage the studio-domain corpora into an MFA corpus for local alignment.

Why this exists: 94.5% of the vowel classifier's training audio is crowdsourced,
while learners record in studio-ish conditions. Measured by source on held-out
speakers, nasal F1 runs 0.65-0.85 on crowdsourced voices and 0.14-0.39 on studio
ones. More crowdsourced data does not close that; more studio data is the only
thing that lifts the ceiling.

Two rules are enforced here, and both matter more than volume:

**Diacritics are required.** An undiacritized line teaches the classifier that
``ọ`` is ``o``, which is precisely the contrast that fails on real attempts
(77.6% correct on SLR86 held-out, the weakest of any pair measured). A line
without tone marks is worse than no line at all for labelling vowels, however
good its audio.

**Transcripts are deduplicated.** ``thisniyi`` is explicitly ``cv_and_openslr``
and therefore contains SLR86, which is already the local corpus. Merging blind
would double-count utterances and pull the classifier further toward speakers it
already over-represents.

**Ingestion is incremental.** Transcripts already staged in the output corpus
are read back as part of the deduplication set, so re-running adds only what is
new and re-running with no new data is a no-op. This matters because the sources
overlap heavily: Common Voice 26.0 and IroyinSpeech share 3,062 byte-identical
clips, IroyinSpeech being a subset of an earlier Common Voice release.

    python scripts/ingest_studio.py --out data/studio_corpus
    python scripts/ingest_studio.py --only commonvoice
"""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.orthography import normalize, tokenize  # noqa: E402

#: Duration bounds for an accepted utterance.
#:
#: The floor is 0.20 s because michsethowusu is one- and two-word fragments with
#: a median of 0.26 s -- a 0.30 s floor rejected more than half of it. These are
#: about one syllable each and carry no context, so they earn their place on
#: phone coverage rather than on prosody.
#:
#: The ceiling is 30 s because Waxal's median utterance is 16.9 s, so a 15 s
#: ceiling took only 42% of it and threw away precisely the long, context-rich
#: read sentences that balance michsethowusu's fragments. 30 s keeps 76% of
#: Waxal and stays inside what MFA aligns comfortably; its longest clip is 580 s
#: and is still excluded.
MIN_DURATION_S, MAX_DURATION_S = 0.20, 30.0
TARGET_RATE = 16000
#: Clips with no speaker label are grouped this many to a pseudo-speaker, so
#: MFA's speaker-adaptive stage has something to estimate a transform from.
BUCKET_SIZE = 50


def has_tone_marks(text: str) -> bool:
    """Whether the line carries acute or grave marks at all."""
    return any(c in "̀́" for c in unicodedata.normalize("NFD", text or ""))


def decode_audio(data: bytes | None, path: Path | None):
    """Read audio from bytes or a file, falling back to ffmpeg for mp3.

    libsndfile handles mp3 only from 1.1 onward, and the IroyinSpeech clips are
    mp3, so the fallback is not optional on older systems.
    """
    try:
        if data is not None:
            return sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return sf.read(str(path), dtype="float32", always_2d=False)
    except Exception:
        pass
    command = ["ffmpeg", "-v", "quiet", "-i", "pipe:0" if data else str(path),
               "-f", "wav", "-ar", str(TARGET_RATE), "-ac", "1", "pipe:1"]
    result = subprocess.run(command, input=data, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise ValueError("unreadable audio")
    return sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=False)


class Stager:
    """Writes accepted utterances into an MFA-format corpus."""

    def __init__(self, root: Path, seen_text: set[str]) -> None:
        self.root = root
        self.seen_text = seen_text
        self.kept = Counter()
        self.hours = Counter()
        self.rejected = Counter()
        self.unlabelled = Counter()
        self.speakers: dict[str, set] = {}

    def add(self, source: str, speaker: str | None, utt_id: str, text: str,
            data: bytes | None = None, path: Path | None = None,
            per_speaker: bool = False) -> bool:
        text = (text or "").strip()
        if not text:
            self.rejected["empty"] += 1
            return False
        if not has_tone_marks(text):
            self.rejected["no diacritics"] += 1
            return False

        key = " ".join(tokenize(normalize(text), keep_noise=False))
        if not key:
            self.rejected["empty"] += 1
            return False
        # The transcript is the deduplication key because the sources
        # redistribute each other's recordings after re-encoding, which changes
        # every byte and defeats hashing the audio.
        #
        # It is the wrong key when a corpus deliberately has several speakers
        # read the same prompt. Common Voice does, and dropping those cost 332
        # of 448 genuinely new recordings on the 26.0 ingest -- new voices, new
        # formants, discarded for saying a sentence we already had. A vowel
        # classifier learns from acoustics, so a second speaker reading a known
        # sentence is new evidence, not a duplicate.
        #
        # `per_speaker` therefore keys on speaker as well, and is set only for
        # sources with trustworthy speaker IDs. Sources that anonymise speakers
        # keep the strict rule, since there a repeated transcript is more likely
        # to be the same recording arriving by a second route.
        if per_speaker and speaker:
            key = f"{speaker}\x00{key}"
        if key in self.seen_text:
            self.rejected["duplicate transcript"] += 1
            return False

        try:
            samples, rate = decode_audio(data, path)
        except Exception:
            self.rejected["unreadable"] += 1
            return False

        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        duration = len(samples) / rate if rate else 0.0
        if duration < MIN_DURATION_S:
            self.rejected["too short"] += 1
            return False
        if duration > MAX_DURATION_S:
            self.rejected["too long"] += 1
            return False
        if rate != TARGET_RATE:
            count = int(len(samples) / rate * TARGET_RATE)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")

        if speaker:
            group = f"{source}_{speaker}"
        else:
            group = f"{source}_unk{self.unlabelled[source] // BUCKET_SIZE:04d}"
            self.unlabelled[source] += 1

        directory = self.root / group
        directory.mkdir(parents=True, exist_ok=True)
        sf.write(str(directory / f"{utt_id}.wav"), samples, TARGET_RATE, subtype="PCM_16")
        (directory / f"{utt_id}.lab").write_text(normalize(text), encoding="utf-8")

        self.seen_text.add(key)
        if per_speaker and speaker:
            # Also record the bare transcript, so a later strict source does not
            # readmit this utterance under its own weaker rule.
            self.seen_text.add(key.split("\x00", 1)[1])
        self.kept[source] += 1
        self.hours[source] += duration / 3600
        self.speakers.setdefault(source, set()).add(group)
        return True


def existing_transcripts(config: EngineConfig) -> set[str]:
    """Normalized transcripts already in the local corpus, for deduplication."""
    seen = set()
    index = config.line_index
    if not index.exists():
        return seen
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 3:
                seen.add(" ".join(tokenize(normalize(row[2]), keep_noise=False)))
    return seen


def staged_transcripts(root: Path) -> set[str]:
    """Deduplication keys for everything already staged in the output corpus.

    Emits both forms of key, because the two deduplication rules have to share
    one set: the bare transcript, which strict sources match on, and the
    speaker-qualified transcript that ``per_speaker`` sources match on. The
    speaker is recovered from the group directory name, which is written as
    ``{source}_{speaker}`` -- so a recording reaching us again through a
    different corpus is still recognised, as the speaker ID survives the
    re-export even when the audio has been re-encoded.
    """
    seen = set()
    for lab in root.rglob("*.lab"):
        try:
            text = lab.read_text(encoding="utf-8")
        except OSError:
            continue
        key = " ".join(tokenize(normalize(text), keep_noise=False))
        seen.add(key)
        group = lab.parent.name
        if "_" in group:
            seen.add(f"{group.split('_', 1)[1]}\x00{key}")
    return seen


def utterance_phones(text: str) -> set[str]:
    """The distinct vowel phones a transcript would produce, or an empty set."""
    from yoruba_engine.exceptions import G2PError  # noqa: PLC0415
    from yoruba_engine.g2p.dictionary import grapheme_to_phonemes  # noqa: PLC0415

    found = set()
    for word in text.split():
        try:
            symbols = [p.symbol for p in grapheme_to_phonemes(normalize(word))]
        except (G2PError, Exception):  # noqa: B014
            continue
        found.update(s.split("_")[0] for s in symbols)
    return found


def ingest_parquet(stager: Stager, paths: list[Path], source: str,
                   limit: int = 0, prefer: set[str] | None = None) -> None:
    """Read HuggingFace parquet shards into the corpus.

    ``limit`` caps how many utterances this source may contribute, spread as an
    equal quota per shard rather than taken from the front. michsethowusu is
    1,647,022 fragments totalling 216 hours at a median of 0.26 seconds; taken
    whole it would be 96% of the corpus by utterance count and teach the
    acoustic model almost nothing but context-free fragments. The quota keeps
    the sample spread across all 50 shards instead of exhausting the first few.

    ``prefer`` restricts the pass to utterances containing one of the named
    vowel phones, and ignores ``limit``. The corpus is not short of vowels in
    general but of one in particular: ``ẹn`` occurs 45 times against ``ọn``'s
    3,246, which is why it scored 0.009 F1 and looked unlearnable. It is rare in
    the language rather than in our sampling -- it appears in 0.04% of
    michsethowusu too -- so the only way to raise it is to read every shard and
    keep the few hundred utterances that contain it. Selecting on the text
    column first means audio is decoded only for rows that qualify.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    quota = 0 if prefer else ((limit // max(len(paths), 1)) if limit else 0)
    for path in paths:
        if not path.exists():
            continue
        if limit and not prefer and stager.kept[source] >= limit:
            break
        target = stager.kept[source] + quota if quota else 0
        parquet = pq.ParquetFile(path)
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=256)):
            data = batch.to_pydict()
            texts = data.get("sentence") or data.get("text") or []
            audios = data.get("audio") or []
            speakers = data.get("speaker_id") or data.get("client_id")
            if target and stager.kept[source] >= target:
                break
            for row, (text, audio) in enumerate(zip(texts, audios)):
                if not audio or not audio.get("bytes"):
                    continue
                if prefer and not (utterance_phones(str(text)) & prefer):
                    continue
                speaker = str(speakers[row]) if speakers else None
                stager.add(source, speaker,
                           f"{source}_{path.stem[-6:]}_{batch_index}_{row}",
                           text, data=audio["bytes"])
        print(f"  {path.name}: running total {stager.kept[source]:,}", flush=True)


def ingest_common_voice(stager: Stager, root: Path, source: str,
                        per_speaker: bool = True) -> None:
    """Read a Common Voice export: validated.tsv plus clips/.

    ``per_speaker`` defaults on here: Common Voice records a stable
    ``client_id`` and deliberately collects the same prompt from several
    speakers, which is acoustic variety rather than duplication.
    """
    table = root / "validated.tsv"
    if not table.exists():
        print(f"  {table} not found"); return
    with table.open(encoding="utf-8", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            clip = root / "clips" / (row.get("path") or "")
            if not clip.exists():
                continue
            # client_id is a long hash; the tail is unique enough to name a dir.
            speaker = (row.get("client_id") or "")[-16:] or None
            stager.add(source, speaker, f"{source}_{clip.stem}",
                       row.get("sentence") or "", path=clip,
                       per_speaker=per_speaker)
            if number and number % 500 == 0:
                print(f"  {number} rows, kept {stager.kept[source]:,}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "studio_corpus")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--max-michsethowusu", type=int, default=120_000,
                        metavar="N",
                        help="cap on michsethowusu utterances (default 120,000, "
                             "about 28 h; the full set is 216 h of fragments)")
    parser.add_argument("--prefer-vowels", nargs="*", metavar="PHONE",
                        help="keep only utterances containing these vowel "
                             "phones, e.g. --prefer-vowels En. Overrides the cap.")
    parser.add_argument("--only", nargs="*", metavar="SOURCE",
                        help="ingest only these sources (default: all)")
    args = parser.parse_args()
    wanted = set(args.only) if args.only else None

    config = EngineConfig()
    seen = existing_transcripts(config)
    print(f"{len(seen):,} transcripts already in the local corpus")

    # Anything already staged counts as seen too, so a re-run adds only what is
    # new rather than rebuilding the corpus.
    staged = staged_transcripts(args.out)
    print(f"{len(staged):,} already staged in {args.out.name}")
    seen |= staged
    print(f"{len(seen):,} deduped against\n")

    args.out.mkdir(parents=True, exist_ok=True)
    stager = Stager(args.out, seen)

    def run(source: str, label: str, action) -> None:
        if wanted is not None and source not in wanted:
            return
        print(label)
        action()
        print()

    run("iroyin", "IroyinSpeech (Common Voice export)",
        lambda: ingest_common_voice(stager, args.data / "IroyinSpeech", "iroyin"))

    # Common Voice 26.0 is the canonical release IroyinSpeech was cut from, so
    # most of it deduplicates away. It is ingested for the residue: a few
    # hundred sentences and a dozen or so speakers the earlier export predates.
    run("commonvoice", "Common Voice Scripted Speech 26.0 (CC0)",
        lambda: ingest_common_voice(
            stager, args.data / "cv-corpus-26.0-2026-06-12" / "yo", "commonvoice"))

    run("ljspeech", "At-Tawheed yoruba-tts-ljspeech",
        lambda: ingest_parquet(
            stager, [args.data / "train-00000-of-00001.parquet"], "ljspeech"))

    # One- and two-word fragments averaging under a second. Low context for
    # triphone modelling, but 24 hours of it are in the shipped acoustic model
    # and 85.8% of its lines are tone-marked, so it earns its place on phone
    # coverage rather than on prosody.
    run("michsethowusu", "michsethowusu yoruba-speech-text-parallel",
        lambda: ingest_parquet(
            stager,
            sorted(args.data.glob("michsethowusu/data/train-*-of-00050-*.parquet"))
            or sorted(args.data.glob("train-*-of-00050-*.parquet")),
            "michsethowusu", limit=args.max_michsethowusu,
            prefer=set(args.prefer_vowels) if args.prefer_vowels else None))

    # Long read sentences at 48 kHz, downsampled on ingest. Only 58.4% of its
    # lines carry a tone mark, and at ~10 seconds an utterance with no mark
    # anywhere is genuinely undiacritized rather than legitimately all-mid, so
    # the tone-mark filter drops roughly 42% of it. The shipped model was
    # trained with that filter disabled and so learned from the unmarked lines.
    run("waxal", "google WaxalNLP",
        lambda: ingest_parquet(
            stager,
            sorted(args.data.glob("WAXAL/yor-*.parquet"))
            or sorted(args.data.glob("yor-train-*.parquet")),
            "waxal"))

    run("thisniyi", "thisniyi cv_and_openslr",
        lambda: ingest_parquet(
            stager, sorted(args.data.glob("thisniyi_*.parquet")), "thisniyi"))

    print(f"\n{'source':<12}{'utterances':>12}{'hours':>9}{'speakers':>10}")
    for source in sorted(stager.kept):
        print(f"{source:<12}{stager.kept[source]:>12,}{stager.hours[source]:>9.2f}"
              f"{len(stager.speakers[source]):>10}")
    print(f"{'TOTAL':<12}{sum(stager.kept.values()):>12,}"
          f"{sum(stager.hours.values()):>9.2f}"
          f"{sum(len(s) for s in stager.speakers.values()):>10}")

    print("\nrejected:")
    for reason, count in stager.rejected.most_common():
        print(f"  {reason:<22}{count:>8,}")
    print(f"\nCorpus written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
