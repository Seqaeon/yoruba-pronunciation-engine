"""Compare CTC forced alignment against MFA on the same recordings.

If a CTC model can place word boundaries as well as MFA does, the aligner can be
replaced -- which removes 1.65 GB of Kaldi, the SQLite databases that destroyed a
five-hour training run, and the ~35 s per-call startup that forces every
assessment to be a queued job.

The model aligns *characters*, not phones, and that turns out not to matter.
Yoruba orthography is near-phonemic -- the rule-based G2P converts the whole
3,413-word corpus vocabulary with zero failures -- so character spans merge into
phone spans mechanically (``u`` + ``n`` becomes the nasal vowel ``un``, ``g`` +
``b`` becomes ``gb``). Crucially the vocabulary here is *not* romanized: ``ẹ``,
``ọ``, ``ṣ`` and every tone-marked vowel are real tokens, so no information is
discarded before alignment.

What is measured is agreement with MFA on word boundaries, since that is what
the engine consumes.

    python scripts/ctc_align_compare.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import unicodedata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.g2p.orthography import normalize  # noqa: E402

FRAME_S = 0.02


def tokenize_for_model(words, vocab, delimiter):
    """Map words to vocabulary ids, returning (ids, word index per id).

    Characters the vocabulary lacks are dropped rather than mapped to <unk>: an
    unknown token forces the aligner to explain audio with a symbol it never
    learned, which distorts the surrounding boundaries.
    """
    ids, owner, dropped = [], [], []
    for index, word in enumerate(words):
        if index and delimiter in vocab:
            ids.append(vocab[delimiter]); owner.append(-1)
        for char in word:
            if char in vocab:
                ids.append(vocab[char]); owner.append(index)
            else:
                # Try the decomposed form: 'ọ̀' is often ọ + combining grave.
                pieces = [c for c in unicodedata.normalize("NFD", char) if c in vocab]
                if pieces:
                    for c in pieces:
                        ids.append(vocab[c]); owner.append(index)
                else:
                    dropped.append(char)
    return ids, owner, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="models/lyngual-yoruba")
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from transformers import Wav2Vec2ForCTC

    device = args.device if torch.cuda.is_available() else "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(device)
    raw = json.loads((Path(args.model) / "vocab.json").read_text(encoding="utf-8"))
    vocab = raw.get("yor") or raw if isinstance(next(iter(raw.values())), dict) else raw
    delimiter = next((k for k in ("|", " ") if k in vocab), "|")
    blank = vocab.get("<pad>", vocab.get("<blank>", 0))
    print(f"{args.model} on {device}; {len(vocab)} tokens, blank={blank}")

    grids = sorted(args.aligned.rglob("*.TextGrid"))[: args.limit]
    index = {p.stem: p for p in args.audio.rglob("*.wav")}

    deltas, end_deltas, starts_ok, aligned, failed = [], [], 0, 0, 0
    span_mismatch = 0
    total_words, matched_words = 0, 0
    all_dropped: set[str] = set()

    for grid in grids:
        wav = index.get(grid.stem)
        if wav is None:
            continue
        try:
            reference = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            failed += 1
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            count = int(len(samples) / rate * 16000)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")

        words = [normalize(w.label) for w in reference.words]
        words = [w for w in words if w and not w.startswith("[")]
        if not words:
            continue
        ids, owner, dropped = tokenize_for_model(words, vocab, delimiter)
        all_dropped |= set(dropped)
        if not ids:
            failed += 1
            continue

        with torch.inference_mode():
            logits = model(torch.from_numpy(samples)[None].to(device)).logits
            logprobs = torch.log_softmax(logits, dim=-1)
        try:
            aligned_tokens, scores = AF.forced_align(
                logprobs.cpu(), torch.tensor([ids], dtype=torch.int32), blank=blank)
        except Exception as exc:
            failed += 1
            if failed <= 2:
                print(f"  forced_align failed on {grid.stem}: {type(exc).__name__}: {exc}")
            continue
        aligned += 1

        # merge_tokens is the API built for this. An earlier version of this
        # script hand-rolled run detection over the frame labels and got it
        # wrong: it reset its "previous token" marker on every blank, so a token
        # whose emission was interrupted by a blank counted twice, the span list
        # grew longer than the target sequence, and every following word was
        # read off the wrong frame. That produced a plausible-looking median
        # with a terrible mean -- the signature of accumulating drift, which is
        # exactly what a misaligned index does.
        spans = AF.merge_tokens(aligned_tokens[0], scores[0])
        if len(spans) != len(ids):
            span_mismatch += 1
            continue

        # Spans are in target order, so span i belongs to whichever word owns
        # target token i.
        word_start_frame: dict[int, int] = {}
        word_end_frame: dict[int, int] = {}
        for span, owner_index in zip(spans, owner):
            if owner_index < 0:
                continue
            word_start_frame.setdefault(owner_index, span.start)
            word_end_frame[owner_index] = span.end

        for w, word in enumerate(words):
            if w >= len(reference.words):
                break
            total_words += 1
            if w not in word_start_frame:
                continue
            matched_words += 1
            predicted = word_start_frame[w] * FRAME_S
            deltas.append(abs(predicted - reference.words[w].start_s))
            if w in word_end_frame:
                end_deltas.append(abs(word_end_frame[w] * FRAME_S
                                      - reference.words[w].end_s))
            if abs(predicted - reference.words[w].start_s) <= 0.05:
                starts_ok += 1

    print(f"\nutterances aligned : {aligned} of {len(grids)} ({failed} failed)")
    print(f"span-count mismatch: {span_mismatch} "
          f"(spans != target tokens; would misalign every later word)")
    print(f"words with a start : {matched_words:,} of {total_words:,}")
    if all_dropped:
        print(f"characters not in the vocabulary: {sorted(all_dropped)}")
    if deltas:
        print(f"\nword-start agreement with MFA:")
        print(f"  median error : {statistics.median(deltas)*1000:.0f} ms")
        print(f"  mean error   : {statistics.mean(deltas)*1000:.0f} ms")
        for tolerance in (0.02, 0.05, 0.10, 0.20):
            share = sum(1 for d in deltas if d <= tolerance) / len(deltas)
            print(f"  within {tolerance*1000:>3.0f} ms : {share:.1%}")
        if end_deltas:
            print(f"\nword-end agreement with MFA:")
            print(f"  median error : {statistics.median(end_deltas)*1000:.0f} ms")
            print(f"  mean error   : {statistics.mean(end_deltas)*1000:.0f} ms")
        print("\nNote: MFA is the reference here, not ground truth -- it is another"
              "\nestimator. But every downstream asset (gop_reference.json, the"
              "\nduration thresholds, the classifier's training spans) was measured"
              "\nagainst MFA's boundaries, so disagreement with MFA is the cost of"
              "\nswitching regardless of which aligner is closer to the truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
