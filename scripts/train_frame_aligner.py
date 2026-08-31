"""Train a frame-level phone classifier to replace MFA's GMM.

MFA costs 13.79 s of an assessment's 14.31 s. Everything else in the engine --
Whisper, the vowel classifier, pitch, tone, GOP -- totals 0.52 s. None of MFA's
cost is alignment: it is corpus setup, a SQLite database and feature extraction,
charged per invocation, and no combination of its flags moves it.

What replaces it is the same architecture MFA uses -- an acoustic model scoring
frames, then Viterbi against the known phone sequence -- with the GMM swapped
for the encoder this project already trains. The encoder pass costs about 16 ms.

**The measure that matters is agreement with MFA, not accuracy.** The vowel
classifier's 575,125 training vowels were cut using MFA spans, and word-duration
and GOP statistics are calibrated against them. A model with excellent frame
accuracy whose boundaries sit 40 ms from MFA's has not made the engine faster,
it has invalidated every threshold in it. So this reports boundary error in
milliseconds, and frame accuracy only as a diagnostic.

For scale: MMS_FA, an off-the-shelf CTC aligner, scatters 36 ms on vowel edges.
Frame quantisation alone costs about 5 ms, since wav2vec2 steps 20 ms and MFA
works at 10 ms. Somewhere under 20 ms and the calibration transfers.

Three stages, each skipped if its output exists:

    python scripts/train_frame_aligner.py --stage features
    python scripts/train_frame_aligner.py --stage train
    python scripts/train_frame_aligner.py --stage evaluate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

FRAME_S = 0.02
SAMPLE_RATE = 16000

#: Held out by speaker. Whisper and wav2vec2 both memorise a voice readily, and
#: a random utterance split would report a number that says nothing about a
#: learner the model has never heard.
DEV_SHARE, TEST_SHARE = 0.04, 0.04


def _portable(path: Path) -> str:
    """The encoder path as a checkpoint should record it.

    Recorded verbatim this was an absolute path, which named the encoder
    correctly on the machine that trained the head and named nothing at all
    anywhere else -- on Windows it reached transformers as
    '\\home\\...\\models\\finetuned_utterance' and stopped the backend.
    A path inside the project is recorded relative to it so that any
    installation can find it.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def split_of(speaker: str) -> str:
    draw = int(hashlib.sha256(speaker.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if draw < TEST_SHARE:
        return "test"
    if draw < TEST_SHARE + DEV_SHARE:
        return "dev"
    return "train"


def load_manifest(labels_dir: Path, limit_per_split: dict[str, int]) -> dict[str, list]:
    rows: dict[str, list] = {"train": [], "dev": [], "test": []}
    with (labels_dir / "frames.tsv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            split = split_of(row["speaker"])
            if len(rows[split]) < limit_per_split.get(split, 0):
                rows[split].append(row)
    return rows


# ---------------------------------------------------------------------------
# Stage 1: encoder features
# ---------------------------------------------------------------------------

def extract_features(args) -> None:
    """Run the frozen encoder once and cache what it produces.

    The encoder never changes, so its output is a constant. Caching it turns
    every later epoch from a 16 ms-per-utterance GPU pass into a memmap read,
    which is what makes it possible to train a head for fifty epochs in the time
    one epoch would otherwise take.

    fp16 on disk, because the head reads it in fp16 anyway and fp32 would double
    a file that is already the largest thing this produces.
    """
    import torch
    from transformers import Wav2Vec2Model

    from yoruba_engine.audio import read_mono

    labels = np.load(args.labels / "labels.npz")
    rows = load_manifest(args.labels, {"train": args.train_utterances,
                                       "dev": args.eval_utterances,
                                       "test": args.eval_utterances})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Wav2Vec2Model.from_pretrained(args.encoder, torch_dtype=dtype).to(device).eval()
    width = model.config.hidden_size
    print(f"encoder {args.encoder} on {device} ({dtype}), hidden {width}")

    args.features.mkdir(parents=True, exist_ok=True)
    for split, entries in rows.items():
        if not entries:
            continue
        target = args.features / f"{split}.npy"
        # Every output, not just the features. An interrupted run leaves a
        # part-written .npy behind, and testing only for that made a partial
        # cache look complete -- the next run skipped extraction and then failed
        # on the labels that were never written.
        produced = [target, args.features / f"{split}_labels.npy",
                    args.features / f"{split}_index.npy",
                    args.features / f"{split}_frames.txt"]
        if all(path.exists() for path in produced) and not args.force:
            print(f"  {split}: already extracted")
            continue
        for path in produced:
            path.unlink(missing_ok=True)

        # Two passes: size the file, then fill it. Pre-allocating avoids holding
        # the whole split in RAM, which is how an earlier version of this
        # project's feature extraction reached 21 GiB and was killed.
        kept, total_frames = [], 0
        for row in entries:
            frames = min(int(row["frames"]), args.max_frames)
            if frames >= 5:
                kept.append((row, frames))
                total_frames += frames
        print(f"  {split}: {len(kept)} utterances, {total_frames} frames "
              f"({total_frames * FRAME_S / 3600:.1f} h)")

        store = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float16, shape=(total_frames, width))
        label_store = np.zeros(total_frames, dtype=np.int16)
        index = np.zeros((len(kept), 2), dtype=np.int64)

        at = 0
        started = time.perf_counter()
        for position, (row, frames) in enumerate(kept):
            # read_mono resamples with a filter. Reading raw samples here and
            # slicing at a hardcoded 16 kHz silently fed 48 kHz audio to the
            # encoder as though it were 16 kHz -- everything stretched threefold,
            # which put one whole speaker's boundaries 360 ms out while every
            # other speaker sat at 10 ms. 2.2% of this corpus is 48 kHz.
            samples = read_mono(ROOT / row["audio"], SAMPLE_RATE)
            samples = samples[:frames * int(FRAME_S * SAMPLE_RATE)]
            with torch.inference_mode():
                batch = torch.tensor(samples, dtype=dtype).unsqueeze(0).to(device)
                hidden = model(batch).last_hidden_state[0].float().cpu().numpy()
            usable = min(len(hidden), frames)
            store[at:at + usable] = hidden[:usable].astype(np.float16)
            label_store[at:at + usable] = labels[row["utterance"]][:usable]
            index[position] = (at, usable)
            at += usable
            if position % 2000 == 0 and position:
                rate_s = position / (time.perf_counter() - started)
                print(f"    {position}/{len(kept)}  {rate_s:.0f} utt/s", flush=True)

        store.flush()
        np.save(args.features / f"{split}_labels.npy", label_store[:at])
        np.save(args.features / f"{split}_index.npy", index)
        # The memmap was sized before per-utterance truncation; record the truth.
        (args.features / f"{split}_frames.txt").write_text(str(at), encoding="utf-8")
        print(f"  {split}: wrote {at} frames to {target}")


# ---------------------------------------------------------------------------
# Stage 2: the head
# ---------------------------------------------------------------------------

def build_head(width: int, classes: int, hidden: int):
    import torch.nn as nn

    # One hidden layer. A linear probe underfits a 31-way phone decision, and
    # anything deeper is fitting MFA's quirks rather than its boundaries -- the
    # teacher is a GMM, not ground truth.
    return nn.Sequential(
        nn.Linear(width, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, classes),
    )


def train(args) -> None:
    import torch
    import torch.nn as nn

    phones = (args.labels / "phones.txt").read_text(encoding="utf-8").split()
    # The memmap was sized before per-utterance truncation, so it is longer than
    # the labels. Trusting its length walks off the end of the label array.
    features = np.load(args.features / "train.npy", mmap_mode="r")
    targets = np.load(args.features / "train_labels.npy")
    features = features[:len(targets)]
    dev_features = np.load(args.features / "dev.npy", mmap_mode="r")
    dev_targets = np.load(args.features / "dev_labels.npy")
    dev_features = dev_features[:len(dev_targets)]
    print(f"{len(features)} training frames, {len(dev_features)} dev, "
          f"{len(phones)} classes")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(features.shape[1], len(phones), args.hidden).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    # Silence is 38.9% of frames. Left alone the head learns to predict it and
    # the boundaries -- which are exactly the silence/phone transitions -- get
    # worse. Inverse-frequency weighting, square-rooted so rare phones do not
    # dominate the loss outright.
    counts = np.bincount(targets, minlength=len(phones)).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    print("class weights: " + "  ".join(
        f"{p}:{w:.2f}" for p, w in list(zip(phones, weights))[:6]) + " ...")

    order = np.arange(len(features))
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        head.train()
        np.random.default_rng(epoch).shuffle(order)
        running, seen = 0.0, 0
        for start in range(0, len(order) - args.batch, args.batch):
            picks = np.sort(order[start:start + args.batch])
            x = torch.tensor(np.asarray(features[picks], dtype=np.float32), device=device)
            y = torch.tensor(targets[picks].astype(np.int64), device=device)
            loss = criterion(head(x), y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(picks)
            seen += len(picks)
            if seen >= args.frames_per_epoch:
                break

        head.eval()
        correct = total = 0
        with torch.inference_mode():
            for start in range(0, len(dev_features), 8192):
                x = torch.tensor(np.asarray(dev_features[start:start + 8192],
                                            dtype=np.float32), device=device)
                predicted = head(x).argmax(-1).cpu().numpy()
                truth = dev_targets[start:start + 8192]
                correct += int((predicted == truth).sum())
                total += len(truth)
        accuracy = correct / max(total, 1)
        print(f"  epoch {epoch:>2}  loss {running/max(seen,1):.4f}  "
              f"dev frame accuracy {accuracy:.4f}", flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"state_dict": head.state_dict(), "phones": phones,
                        "hidden": args.hidden, "width": features.shape[1],
                        "dev_frame_accuracy": accuracy,
                        "encoder": _portable(args.encoder)}, args.model)
    print(f"\nbest dev frame accuracy {best:.4f}, saved to {args.model}")


# ---------------------------------------------------------------------------
# Stage 3: does it put boundaries where MFA does?
# ---------------------------------------------------------------------------

def viterbi(log_posteriors: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Monotonic alignment of a known phone sequence to frames.

    Exactly what MFA's Viterbi does, with the GMM's likelihoods replaced. Each
    phone occupies a contiguous run, runs appear in order, and every frame
    belongs to one -- so this cannot invent, drop or reorder a phone, which is
    the property that makes forced alignment forced.

    Returns the frame index at which each phone starts.
    """
    frames, count = len(log_posteriors), len(sequence)
    scores = np.full((frames, count), -np.inf, dtype=np.float32)
    came_from = np.zeros((frames, count), dtype=np.int8)
    emission = log_posteriors[:, sequence]

    scores[0, 0] = emission[0, 0]
    for t in range(1, frames):
        stay = scores[t - 1]
        advance = np.concatenate(([-np.inf], scores[t - 1, :-1]))
        take_advance = advance > stay
        scores[t] = np.where(take_advance, advance, stay) + emission[t]
        came_from[t] = take_advance

    starts = np.zeros(count, dtype=np.int64)
    position = count - 1
    for t in range(frames - 1, 0, -1):
        if came_from[t, position]:
            starts[position] = t
            position -= 1
            if position < 0:
                break
    return starts


def evaluate(args) -> None:
    import torch

    phones = (args.labels / "phones.txt").read_text(encoding="utf-8").split()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_head(checkpoint["width"], len(phones), checkpoint["hidden"])
    head.load_state_dict(checkpoint["state_dict"])
    head = head.to(device).eval()

    features = np.load(args.features / "test.npy", mmap_mode="r")
    targets = np.load(args.features / "test_labels.npy")
    features = features[:len(targets)]
    index = np.load(args.features / "test_index.npy")

    vowel_ids = {i for i, p in enumerate(phones)
                 if p[0] in "aeiouEO" and p != "sil"}
    all_errors, vowel_errors = [], []
    frame_correct = frame_total = 0

    for start, length in index:
        if length < 5:
            continue
        block = np.asarray(features[start:start + length], dtype=np.float32)
        truth = targets[start:start + length]
        with torch.inference_mode():
            logits = head(torch.tensor(block, device=device))
            log_posteriors = torch.log_softmax(logits, -1).cpu().numpy()
        frame_correct += int((log_posteriors.argmax(-1) == truth).sum())
        frame_total += length

        # MFA's own segmentation, run-length encoded: the sequence to align, and
        # the boundaries to be judged against.
        edges = np.flatnonzero(np.diff(truth)) + 1
        sequence = truth[np.concatenate(([0], edges))]
        reference = np.concatenate(([0], edges))
        if len(sequence) < 2:
            continue

        predicted = viterbi(log_posteriors, sequence)
        for phone, want, got in zip(sequence[1:], reference[1:], predicted[1:]):
            error = abs(int(got) - int(want)) * FRAME_S * 1000
            all_errors.append(error)
            if phone in vowel_ids:
                vowel_errors.append(error)

    def report(name, values):
        if not values:
            print(f"  {name}: nothing to report")
            return
        a = np.array(values)
        print(f"  {name:<18}median {np.median(a):5.1f} ms   mean {a.mean():5.1f} ms   "
              f"p90 {np.percentile(a, 90):6.1f} ms   within 20 ms {np.mean(a <= 20):.0%}")

    print(f"\nheld-out speakers, {len(index)} utterances, "
          f"{len(all_errors)} boundaries\n")
    print(f"  frame accuracy    {frame_correct / max(frame_total, 1):.4f}")
    report("all boundaries", all_errors)
    report("vowel onsets", vowel_errors)
    print("\n  for comparison, MMS_FA off the shelf: 36 ms median on vowel edges")
    print("  frame quantisation floor (20 ms step against MFA's 10 ms): ~5 ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("features", "train", "evaluate", "all"),
                        default="all")
    parser.add_argument("--labels", type=Path, default=ROOT / "data" / "aligner_labels")
    parser.add_argument("--features", type=Path, default=ROOT / "data" / "aligner_features")
    parser.add_argument("--encoder", type=Path, default=ROOT / "models" / "finetuned_utterance")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "frame_aligner.pt")
    parser.add_argument("--train-utterances", type=int, default=20000)
    parser.add_argument("--eval-utterances", type=int, default=1500)
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Cap per utterance; 1000 frames is 20 s")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--frames-per-epoch", type=int, default=2_000_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.stage in ("features", "all"):
        extract_features(args)
    if args.stage in ("train", "all"):
        train(args)
    if args.stage in ("evaluate", "all"):
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
