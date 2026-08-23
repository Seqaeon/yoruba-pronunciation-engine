"""Read a whole utterance's tone at once, instead of one unit at a time.

The shipped tone classifier is a gradient-boosted tree over 14 hand-designed
features per tone-bearing unit. It reaches 75.9% on nine SLR86 speakers and
69.6% on 675 mixed ones, and -- the finding that motivates this -- feeding it
eight times as much data moved it 69.6% to 70.2%. It is not short of examples.
It is short of the thing it is being asked to model.

**Yoruba tone is relative, and the current features only glimpse that.**
Permutation importance on the held-out speakers ranks the top five as:

    excursion            +0.092     contour shape within the unit
    delta_prev           +0.074     the previous unit
    vowels_in_word       +0.066     word structure
    delta_next           +0.058     the next unit
    level_z              +0.053     speaker-normalised height

Three of five are context. F0 declines across an utterance, so a High late in a
phrase can sit below a Low early in it, and a model that sees one unit plus
three fixed lags is reconstructing that through a keyhole. A sequence model sees
the whole utterance and can learn declination, neighbour interaction and phrase
position together.

**Tone-bearing units, not vowels.** Tone in Yoruba sits on vowels *and* syllabic
nasals. A tone-stripped ``n_H`` reaches a TextGrid as a bare ``n``, so the engine
treated it as a consonant -- and because ``tone_pattern`` counts it while the
alignment did not, every word containing ``ń`` was dropped from tone training for
a syllable-count mismatch rather than merely measured short.
``mark_syllabic_nasals`` recovers the distinction from the transcript, and this
trains on both.

The architecture is deliberately small. 205,690 units is not much for a deep
model, and the input is 14 numbers per unit, so a two-layer BiGRU over the
sequence has enough capacity to model context without the data to overfit
wildly. It is compared against the tree on the same speakers, and the tree wins
by default: a sequence model that ties is worse, because it is slower to run and
harder to reason about.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def build_sequences(config, aligned_dir: Path, corpus_dir: Path, limit: int | None):
    """One variable-length sequence of tone-bearing units per utterance.

    The tree is trained on rows; this needs them kept in utterance order, with
    the utterance boundary preserved, because the boundary is where declination
    resets.
    """
    import random

    from yoruba_engine.alignment.textgrid import (
        alignment_from_textgrid, mark_syllabic_nasals,
    )
    from yoruba_engine.datatypes import Tone
    from yoruba_engine.g2p.orthography import is_noise_token, tone_pattern
    from yoruba_engine.pitch.extractor import PitchExtractor
    from yoruba_engine.tone.analyzer import ToneAnalyzer
    from yoruba_engine.tone.features import extract_features, measurable

    by_id = {}
    for lab in corpus_dir.rglob("*.lab"):
        wav = lab.with_suffix(".wav")
        if wav.exists():
            text = lab.read_text(encoding="utf-8").strip()
            if text:
                by_id[lab.stem] = (wav, text, lab.parent.name)

    grids = sorted(aligned_dir.rglob("*.TextGrid"))
    random.Random(0).shuffle(grids)
    if limit:
        grids = grids[:limit]
    print(f"{len(grids)} alignments, {len(by_id)} transcripts", flush=True)

    extractor = PitchExtractor(config.pitch)
    analyzer = ToneAnalyzer(config.tone)
    tone_index = {Tone.HIGH: 0, Tone.MID: 1, Tone.LOW: 2}

    sequences, targets, speakers = [], [], []
    mismatched = skipped = 0
    for position, grid in enumerate(grids, 1):
        entry = by_id.get(grid.stem)
        if entry is None:
            skipped += 1
            continue
        wav, transcript, speaker = entry
        try:
            alignment = alignment_from_textgrid(grid, utterance_id=grid.stem)
            # Without this the alignment has no syllabic nasals, so tone_pattern
            # and the vowel count disagree on any word containing one -- and such
            # words are dropped whole, not merely measured short.
            mark_syllabic_nasals(alignment, transcript)
            words, _ = extractor.extract(wav, alignment)
        except Exception:
            skipped += 1
            continue
        analyzer.analyse(words)

        # extract_features works on the whole utterance and emits rows only for
        # vowels carrying a usable pitch, so labels are mapped back through each
        # vowel's index in the full list rather than zipped positionally.
        gold: dict[int, object] = {}
        at = 0
        for word in words:
            if is_noise_token(word.word):
                continue
            written = tone_pattern(word.word)
            if len(written) != len(word.vowels):
                mismatched += 1
                at += len(word.vowels)
                continue
            for offset, tone in enumerate(written):
                gold[at + offset] = tone
            at += len(word.vowels)

        vowels = [vowel for word in words for vowel in word.vowels]
        if not measurable(vowels):
            skipped += 1
            continue
        features = extract_features(vowels)
        measured_positions = [i for i, v in enumerate(vowels)
                              if v.mean_semitones is not None]

        rows, labels = [], []
        for row, original in zip(features, measured_positions):
            tone = gold.get(original)
            if tone is None or tone is Tone.UNKNOWN:
                continue
            rows.append(row)
            labels.append(tone_index[tone])
        if len(rows) >= 2:
            sequences.append(np.asarray(rows, dtype=np.float32))
            targets.append(np.asarray(labels, dtype=np.int64))
            speakers.append(speaker)
        if position % 2500 == 0:
            print(f"  {position}/{len(grids)}  {len(sequences)} usable, "
                  f"{sum(len(t) for t in targets)} units", flush=True)

    print(f"\n{len(sequences)} utterances, "
          f"{sum(len(t) for t in targets)} tone-bearing units "
          f"({skipped} skipped, {mismatched} syllable-count mismatches)")
    return sequences, targets, speakers


class ToneSequenceModel:
    """A two-layer BiGRU over an utterance's tone-bearing units."""

    def __init__(self, features: int, hidden: int = 96, classes: int = 3):
        import torch.nn as nn

        self.torch_nn = nn
        self.net = nn.ModuleDict({
            "project": nn.Sequential(nn.Linear(features, hidden), nn.GELU()),
            "rnn": nn.GRU(hidden, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2),
            "head": nn.Linear(hidden * 2, classes),
        })

    def __call__(self, padded, lengths):
        import torch
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        hidden = self.net["project"](padded)
        packed = pack_padded_sequence(hidden, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        output, _ = self.net["rnn"](packed)
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=padded.size(1))
        return self.net["head"](output)


def main() -> int:
    import torch
    import torch.nn as nn

    from yoruba_engine.config import EngineConfig
    from yoruba_engine.tone.features import FEATURE_COUNT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=ROOT / "output" / "train_aligned")
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "tone_sequence.pt")
    parser.add_argument("--limit", type=int,
                        help="Only this many alignments, for a trial. The default "
                             "is all of them: pitch extraction runs at about 140 "
                             "utterances a second and the whole corpus of "
                             "sequences is under 100 MB in memory, so there is no "
                             "reason to cap it. An earlier default of 40000 meant "
                             "a run launched without the flag silently trained on "
                             "less data than the run it was meant to replace.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--cache", type=Path,
                        default=ROOT / "data" / "tone_sequences.npz",
                        help="Where to keep extracted sequences so a "
                             "second architecture does not repeat 23 "
                             "minutes of pitch extraction.")
    args = parser.parse_args()

    config = EngineConfig()

    # Pitch extraction over the corpus is 23 minutes and produces the same
    # sequences every time, so trying a second architecture should not pay for
    # it again. The cache is keyed on what changes the sequences, not on the
    # model: a different --limit or a different corpus writes a different file.
    cache = args.cache
    if cache is not None:
        cache = cache.with_name(
            f"{cache.stem}_{args.limit or 'all'}{cache.suffix}")
    if cache is not None and cache.exists():
        blob = np.load(cache, allow_pickle=True)
        sequences = list(blob["sequences"])
        targets = list(blob["targets"])
        speakers = list(blob["speakers"])
        print(f"reusing {len(sequences)} cached sequences from {cache}")
    else:
        sequences, targets, speakers = build_sequences(
            config, args.aligned, args.corpus_dir, args.limit)
        if cache is not None and sequences:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache,
                     sequences=np.array(sequences, dtype=object),
                     targets=np.array(targets, dtype=object),
                     speakers=np.array(speakers, dtype=object))
            print(f"cached sequences to {cache}")
    if not sequences:
        print("No usable sequences", file=sys.stderr)
        return 1

    # Split by speaker. A tone model asked to read a voice it trained on
    # reports its own memory, and this project has been bitten by that before.
    import hashlib

    def held_out(speaker: str) -> bool:
        draw = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return draw < args.holdout

    train_index = [i for i, s in enumerate(speakers) if not held_out(s)]
    test_index = [i for i, s in enumerate(speakers) if held_out(s)]
    print(f"split by speaker: {len(train_index)} train / {len(test_index)} held-out "
          f"utterances across {len({speakers[i] for i in test_index})} held-out speakers")

    # Standardise on the training split only; the held-out speakers must not
    # contribute to the statistics that normalise them.
    stacked = np.concatenate([sequences[i] for i in train_index])
    mean = stacked.mean(axis=0)
    sd = stacked.std(axis=0)
    sd[sd < 1e-6] = 1.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ToneSequenceModel(FEATURE_COUNT, args.hidden)
    net = model.net.to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)

    counts = Counter(int(t) for i in train_index for t in targets[i])
    weights = np.array([len(train_index) / max(counts[c], 1) for c in range(3)])
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device),
        ignore_index=-100)
    print(f"class weights: {dict(zip(('H','M','L'), weights.round(3)))}")

    def batches(indices, size, shuffle):
        order = list(indices)
        if shuffle:
            np.random.default_rng(0).shuffle(order)
        order.sort(key=lambda i: len(targets[i]))       # group similar lengths
        chunks = [order[at:at + size] for at in range(0, len(order), size)]
        if shuffle:
            np.random.default_rng(1).shuffle(chunks)
        for chunk in chunks:
            longest = max(len(targets[i]) for i in chunk)
            padded = np.zeros((len(chunk), longest, FEATURE_COUNT), dtype=np.float32)
            labels = np.full((len(chunk), longest), -100, dtype=np.int64)
            lengths = np.zeros(len(chunk), dtype=np.int64)
            for row, i in enumerate(chunk):
                span = len(targets[i])
                padded[row, :span] = (sequences[i] - mean) / sd
                labels[row, :span] = targets[i]
                lengths[row] = span
            yield (torch.tensor(padded, device=device),
                   torch.tensor(labels, device=device),
                   torch.tensor(lengths))

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        net.train()
        running = seen = 0.0
        for padded, labels, lengths in batches(train_index, args.batch, True):
            logits = model(padded, lengths)
            loss = criterion(logits.reshape(-1, 3), labels.reshape(-1))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()
            running += float(loss.detach()) * len(lengths)
            seen += len(lengths)
        schedule.step()

        net.eval()
        correct = total = 0
        confusion = np.zeros((3, 3), dtype=np.int64)
        with torch.inference_mode():
            for padded, labels, lengths in batches(test_index, args.batch, False):
                predicted = model(padded, lengths).argmax(-1)
                mask = labels != -100
                correct += int((predicted[mask] == labels[mask]).sum())
                total += int(mask.sum())
                for want, got in zip(labels[mask].cpu().numpy(),
                                     predicted[mask].cpu().numpy()):
                    confusion[want, got] += 1
        accuracy = correct / max(total, 1)
        print(f"  epoch {epoch:>2}  loss {running/max(seen,1):.4f}  "
              f"held-out accuracy {accuracy:.4f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"state_dict": net.state_dict(), "mean": mean, "sd": sd,
                        "hidden": args.hidden, "features": FEATURE_COUNT,
                        "holdout_accuracy": accuracy,
                        "units": int(sum(len(targets[i]) for i in train_index)),
                        "speakers": len(set(speakers)),
                        "confusion": confusion.tolist()}, args.out)

    print(f"\nbest held-out accuracy {best:.4f}, saved to {args.out}")
    print("\nconfusion (rows = written tone, columns = predicted)")
    print(f"  {'':6}{'H':>8}{'M':>8}{'L':>8}")
    for name, row in zip("HML", confusion):
        print(f"  {name:<6}{row[0]:>8}{row[1]:>8}{row[2]:>8}")
    print("\nThe gradient-boosted tree reaches 0.696 on 675 held-out speakers.")
    print("A sequence model that only ties is worse: slower, and harder to reason about.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
