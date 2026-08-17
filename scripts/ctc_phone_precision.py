"""Phone-level precision of CTC alignment, which is what the engine consumes.

Word boundaries agreed with MFA to 40 ms median once silence was trimmed, but
nothing downstream reads words. GOP scores phones, the vowel classifier pools
frames over a vowel, and the duration model thresholds phone counts. A 40 ms
error on a 97 ms vowel is 40% of the segment, so word-level agreement does not
settle the question.

CTC emits **character** spans. Getting phone spans means knowing which
characters produced which phone, which is the same walk
:func:`grapheme_to_phonemes` performs -- ``u`` + ``n`` collapses to the nasal
vowel ``un``, ``g`` + ``b`` to ``gb`` -- so it is replicated here with the
character indices retained.

    python scripts/ctc_phone_precision.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.g2p.orthography import (  # noqa: E402
    CONSONANTS, MISSING_DOT_NASAL, ORAL_VOWELS, SYLLABIC_NASAL,
    Tone, decompose_char, grapheme_clusters, normalize,
)
from yoruba_engine.pitch.extractor import is_vowel_phone  # noqa: E402

FRAME_S = 0.02


def phone_spans(word: str):
    """Return ``[(phone, first_char, last_char)]`` for one word.

    Mirrors the consumption rules in :func:`grapheme_to_phonemes`: the point is
    not to re-derive pronunciations but to know how many *characters* each phone
    was built from, so a character-level alignment can be folded into phones.
    """
    chars = grapheme_clusters(normalize(word))
    out, index = [], 0
    while index < len(chars):
        decomposed = decompose_char(chars[index])
        letter = decomposed.letter

        if index + 1 < len(chars):
            pair = letter + decompose_char(chars[index + 1]).letter
            if pair in ("gb", "sh"):
                out.append(("gb" if pair == "gb" else "S", index, index + 1))
                index += 2
                continue

        if letter in ORAL_VOWELS:
            base, consumed = letter, 1
            if index + 1 < len(chars):
                nxt = decompose_char(chars[index + 1])
                nasalizes = (
                    nxt.letter == SYLLABIC_NASAL
                    and nxt.tone is Tone.MID
                    and (index + 2 >= len(chars)
                         or decompose_char(chars[index + 2]).letter not in ORAL_VOWELS)
                )
                if nasalizes:
                    base = MISSING_DOT_NASAL.get(letter, letter) + "n"
                    consumed = 2
            label = base.replace("ẹ", "E").replace("ọ", "O")
            out.append((label, index, index + consumed - 1))
            index += consumed
            continue

        if letter == SYLLABIC_NASAL:
            out.append(("n", index, index))
            index += 1
            continue

        if letter in CONSONANTS:
            out.append((letter.replace("ṣ", "S"), index, index))
            index += 1
            continue

        index += 1          # punctuation, elision apostrophe: no sound
    return out


def speech_bounds(samples, rate=16000, window_s=0.02, floor_db=-40.0, pad_s=0.05):
    """Energy-based speech extent, as in ``ctc_align_trimmed.py``."""
    window = max(int(window_s * rate), 1)
    frames = len(samples) // window
    if frames < 3:
        return 0, len(samples)
    energy = np.abs(samples[: frames * window].reshape(frames, window)).mean(axis=1)
    peak = energy.max()
    if peak <= 0:
        return 0, len(samples)
    loud = np.flatnonzero(20 * np.log10(np.maximum(energy, 1e-10) / peak) > floor_db)
    if loud.size == 0:
        return 0, len(samples)
    pad = int(pad_s * rate)
    return (max(int(loud[0]) * window - pad, 0),
            min((int(loud[-1]) + 1) * window + pad, len(samples)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default="models/lyngual-yoruba")
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "mfa_corpus")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(args.model).eval().to(args.device)
    raw = json.loads((Path(args.model) / "vocab.json").read_text(encoding="utf-8"))
    vocab = raw.get("yor") or raw if isinstance(next(iter(raw.values())), dict) else raw
    delimiter = "|" if "|" in vocab else " "
    blank = vocab.get("<pad>", 0)

    grids = sorted(args.aligned.rglob("*.TextGrid"))[: args.limit]
    index = {p.stem: p for p in args.audio.rglob("*.wav")}

    starts, ends, durations = [], [], []
    by_phone: dict[str, list[float]] = defaultdict(list)
    vowel_starts, vowel_durations = [], []
    label_disagreements = 0
    compared = 0

    for grid in grids:
        wav = index.get(grid.stem)
        if wav is None:
            continue
        try:
            reference = alignment_from_textgrid(grid, utterance_id=grid.stem)
            samples, rate = sf.read(str(wav), dtype="float32", always_2d=False)
        except Exception:
            continue
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if rate != 16000:
            count = int(len(samples) / rate * 16000)
            samples = np.interp(np.linspace(0, len(samples) - 1, count),
                                np.arange(len(samples)), samples).astype("float32")

        words = [w for w in reference.words
                 if normalize(w.label) and not w.label.startswith("[")]
        if not words:
            continue
        begin, finish = speech_bounds(samples)
        offset = begin / 16000
        trimmed = samples[begin:finish]
        if len(trimmed) < 1600:
            continue

        # Character ids, remembering which (word, char index) each id came from.
        ids, source = [], []
        for w_index, word in enumerate(words):
            if w_index:
                ids.append(vocab[delimiter]); source.append(None)
            for c_index, char in enumerate(grapheme_clusters(normalize(word.label))):
                pieces = [char] if char in vocab else [
                    p for p in unicodedata.normalize("NFD", char) if p in vocab]
                for piece in pieces:
                    ids.append(vocab[piece]); source.append((w_index, c_index))
        if not ids:
            continue

        with torch.inference_mode():
            logits = model(torch.from_numpy(trimmed)[None].to(args.device)).logits
            logprobs = torch.log_softmax(logits, dim=-1).cpu()
        try:
            tokens, scores = AF.forced_align(
                logprobs, torch.tensor([ids], dtype=torch.int32), blank=blank)
            spans = AF.merge_tokens(tokens[0], scores[0])
        except Exception:
            continue
        if len(spans) != len(ids):
            continue

        # Character index -> (first frame, last frame).
        char_frames: dict[tuple, tuple] = {}
        for span, origin in zip(spans, source):
            if origin is None:
                continue
            if origin in char_frames:
                previous = char_frames[origin]
                char_frames[origin] = (previous[0], span.end)
            else:
                char_frames[origin] = (span.start, span.end)

        # Fold characters into phones and compare against MFA's phone intervals
        # inside the same word.
        for w_index, word in enumerate(words):
            predicted = []
            for phone, first, last in phone_spans(word.label):
                pieces = [char_frames.get((w_index, c)) for c in range(first, last + 1)]
                pieces = [p for p in pieces if p]
                if not pieces:
                    continue
                predicted.append((phone,
                                  min(p[0] for p in pieces) * FRAME_S + offset,
                                  max(p[1] for p in pieces) * FRAME_S + offset))
            actual = [p for p in reference.phones
                      if word.start_s - 0.02 <= (p.start_s + p.end_s) / 2 <= word.end_s + 0.02]
            if len(predicted) != len(actual):
                label_disagreements += 1
                continue
            for (phone, p_start, p_end), reference_phone in zip(predicted, actual):
                compared += 1
                starts.append(abs(p_start - reference_phone.start_s))
                ends.append(abs(p_end - reference_phone.end_s))
                durations.append(abs((p_end - p_start)
                                     - (reference_phone.end_s - reference_phone.start_s)))
                by_phone[phone].append(abs(p_start - reference_phone.start_s))
                if is_vowel_phone(reference_phone.label):
                    vowel_starts.append(abs(p_start - reference_phone.start_s))
                    vowel_durations.append(
                        abs((p_end - p_start)
                            - (reference_phone.end_s - reference_phone.start_s)))

    print(f"phones compared    : {compared:,}")
    print(f"words skipped      : {label_disagreements:,} "
          f"(phone count from G2P did not match MFA's -- not measurable)\n")
    if not starts:
        print("nothing to report")
        return 1

    reference_vowel_ms = 97
    print(f"{'quantity':<24}{'median':>9}{'mean':>9}{'<=20ms':>9}{'<=50ms':>9}")
    for name, values in (("phone start", starts), ("phone end", ends),
                         ("duration error", durations),
                         ("VOWEL start", vowel_starts),
                         ("VOWEL duration error", vowel_durations)):
        if not values:
            continue
        print(f"{name:<24}{statistics.median(values)*1000:>8.0f}ms"
              f"{statistics.mean(values)*1000:>8.0f}ms"
              f"{sum(1 for v in values if v <= .02)/len(values):>9.0%}"
              f"{sum(1 for v in values if v <= .05)/len(values):>9.0%}")
    if vowel_durations:
        share = statistics.median(vowel_durations) * 1000 / reference_vowel_ms
        print(f"\nmedian vowel duration error is {share:.0%} of a typical "
              f"{reference_vowel_ms} ms vowel")
    print(f"\n{'phone':<8}{'n':>7}{'median start error':>20}")
    for phone in sorted(by_phone, key=lambda p: -statistics.median(by_phone[p]))[:8]:
        values = by_phone[phone]
        if len(values) >= 20:
            print(f"{phone:<8}{len(values):>7}{statistics.median(values)*1000:>17.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
