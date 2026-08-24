#!/usr/bin/env python3
"""Extract igba word pronunciations using iterative Whisper-guided trimming.

Based on visual frame analysis, each section starts with the speaker
pronouncing the word. We extract a candidate 1.5s clip, verify with
Whisper, and adjust if needed.
"""

import json
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / "igba.mp4"
CLIPS_DIR = ROOT / "data" / "igba_clips"
WHISPER_MODEL = ROOT / "models" / "whisper-yoruba"

# Based on careful visual frame analysis, the speaker pronounces
# the word right as the on-screen label appears. Timestamps below
# are the start of each word pronunciation (when label fades in).
# We extract 1.5s windows and let Whisper verify.
CANDIDATES = [
    # (start, dur, stem, expected_word, meaning)
    # Word 1: IGBA (re re) — label appears ~22.0s, speaker says "igba, igba"
    (21.8, 1.5, "igba_mid_mid_200",        "igba",  "Two hundred (200)"),
    # Word 2: ÌGBÁ (do mi) — label appears ~29.5s, speaker says "ìgbá"
    (29.5, 1.5, "igba_low_high_garden_egg", "ìgbá",  "Garden egg"),
    # Word 3: IGBÁ (re mi) — label appears ~36.5s, speaker says "igbá"
    (36.2, 1.5, "igba_mid_high_calabash",   "igbá",  "Calabash"),
    # Word 4: IGBÀ (re do) — label appears ~41.5s, speaker says "igbà"
    (41.3, 1.5, "igba_mid_low_climbing_rope","igbà", "Climbing rope"),
    # Word 5: ÌGBÀ (do mi) — label appears ~49.0s, speaker says "ìgbà"
    (48.8, 1.5, "igba_low_low_time",        "ìgbà",  "Time / season"),
]


def strip_tones(s):
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn").lower()


def extract_clip(start, duration, stem):
    out = CLIPS_DIR / f"{stem}.wav"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", str(VIDEO),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    return out


def whisper_check(clip_path, processor, model, device):
    samples, rate = sf.read(str(clip_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    import torch
    feats = processor(samples, sampling_rate=16000, return_tensors="pt").input_features
    with torch.inference_mode():
        tokens = model.generate(feats.to(device), max_new_tokens=128)
    return processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


def main():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Whisper-Yoruba model …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(WHISPER_MODEL))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(WHISPER_MODEL)).eval().to(device)
    print(f"Ready on {device}\n")

    print("=" * 60)
    print("Extracting and verifying word-only clips")
    print("=" * 60 + "\n")

    final_results = []

    for start, dur, stem, expected, meaning in CANDIDATES:
        print(f"  {stem}: {expected} ({meaning})")

        # Try the initial candidate
        best_start = start
        best_dur = dur
        best_transcript = ""
        found = False

        # Try a few offsets to find the best position
        offsets = [0.0, -0.3, +0.3, -0.5, +0.5, -0.8]
        durations_to_try = [1.5, 1.2, 1.0, 0.8]

        for offset in offsets:
            if found:
                break
            for d in durations_to_try:
                if found:
                    break
                t = start + offset
                if t < 0:
                    continue
                clip = extract_clip(t, d, stem)
                transcript = whisper_check(clip, processor, model, device)
                has_igba = "igba" in strip_tones(transcript) or "igb" in strip_tones(transcript)

                if has_igba:
                    # Check that the transcript is predominantly just "igba"
                    words = strip_tones(transcript).split()
                    # If the transcript is short (1-3 words) and contains igba, it's good
                    if len(words) <= 3:
                        best_start = t
                        best_dur = d
                        best_transcript = transcript
                        found = True
                        print(f"    ✓ Found at {t:.1f}s ({d:.1f}s): '{transcript}'")
                    elif not found:
                        # Keep as fallback
                        best_start = t
                        best_dur = d
                        best_transcript = transcript

        if not found and best_transcript:
            # Use the best we found even if it has extra words
            clip = extract_clip(best_start, best_dur, stem)
            print(f"    ~ Best match at {best_start:.1f}s ({best_dur:.1f}s): '{best_transcript}'")

        if not best_transcript:
            print(f"    ✗ Could not find igba in any offset")

        size_kb = (CLIPS_DIR / f"{stem}.wav").stat().st_size / 1024
        final_results.append({
            "stem": stem,
            "expected": expected,
            "meaning": meaning,
            "abs_start": round(best_start, 3),
            "duration": round(best_dur, 3),
            "abs_end": round(best_start + best_dur, 3),
            "transcript": best_transcript,
            "size_kb": round(size_kb, 1),
        })
        print()

    # Final summary
    print("=" * 60)
    print("FINAL CLIPS")
    print("=" * 60)
    all_ok = True
    for r in final_results:
        has = "igba" in strip_tones(r["transcript"]) or "igb" in strip_tones(r["transcript"])
        s = "✓" if has else "✗"
        if not has:
            all_ok = False
        print(f"  {s} {r['stem']}.wav  ({r['duration']*1000:.0f}ms, {r['size_kb']:.0f}KB)")
        print(f"      {r['expected']} = {r['meaning']}")
        print(f"      Video: {r['abs_start']:.1f}s – {r['abs_end']:.1f}s")
        print(f"      Whisper: '{r['transcript']}'")

    print(f"\n  {'✓ All clips verified!' if all_ok else '⚠ Some clips need manual review'}")

    meta = CLIPS_DIR / "clip_metadata.json"
    with open(meta, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta}")


if __name__ == "__main__":
    main()
