"""Cut native recordings of the app's phrases out of the corpus.

The app plays a teacher recording before a learner attempts a phrase. Recording
twenty clean takes is work; the corpus already contains 193,007 native
utterances, and many of the phrases appear inside one.

A phrase is usually not a whole utterance -- `kí ni orúkọ rẹ` opens a 2.5 s
sentence that continues `ó sì dáhùn pé` -- so each candidate is force-aligned and
cut at the aligner's own word boundaries.

**Candidates are scored before being kept.** A clip that the engine marks down is
the wrong thing to play to someone learning the phrase, so each cut is run
through the full assessment against its own prompt and only the best-scoring one
is written. That also catches a match that is textually right and acoustically
poor.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

#: The practice phrases, keyed as the app sends them.
PHRASES = {
    1: "Ẹ káàbọ̀", 2: "Jọ̀wọ́", 3: "Báwo ni", 4: "Ó dàbọ̀", 5: "Ẹ ṣé",
    6: "Mo ń bọ̀", 7: "Kí ni orúkọ rẹ", 8: "Níbo ni o ń gbé",
    9: "O ṣeun púpọ̀", 10: "Inú mi dùn", 11: "Mo fẹ́ jẹun", 12: "Ebi ń pa mí",
    13: "Ṣé ó yé ẹ", 14: "Pàtẹ́wọ́ fún ara rẹ", 15: "Mo ń lọ sí ilé ẹ̀kọ́",
    16: "Ṣé o lè ràn mí lọ́wọ́", 17: "Ṣé o ti kọ ọ́ tán", 18: "Ọkùnrin ni mí",
    19: "Obìnrin ni mí", 20: "Ọmọ ọdún mélòó ni ẹ",
}


def key(text: str) -> list[str]:
    """Words with tone marks and sub-dots removed, for matching only.

    Matching folds diacritics away because a corpus spelling and an app spelling
    can differ in marks while naming the same word. What is *kept* is the
    original audio, so nothing about the recording is affected by this.
    """
    decomposed = unicodedata.normalize("NFD", text)
    flat = "".join(c for c in decomposed
                   if not unicodedata.combining(c) and (c.isalnum() or c.isspace()))
    return flat.lower().split()


def main() -> int:
    import soundfile as sf

    from yoruba_engine.audio import read_mono
    from yoruba_engine.config import EngineConfig
    from yoruba_engine.pipeline import AssessmentPipeline
    from yoruba_engine.asr.providers import WhisperTranscriptProvider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "train_corpus")
    parser.add_argument("--out", type=Path,
                        default=Path("/home/seqaeon/Downloads/Learn-Yor/app/public/reference"))
    parser.add_argument("--candidates", type=int, default=4,
                        help="How many matches to score per phrase before choosing")
    parser.add_argument("--skip", type=int, nargs="*", default=[4],
                        help="Phrases to leave alone. 4 by default: the corpus "
                             "has no `dàbọ̀`, and matching folds sub-dots away, "
                             "so `dàbò` -- a different word in a code-switched "
                             "sentence -- would otherwise be cut for it.")
    parser.add_argument("--min-score", type=float, default=0.80,
                        help="Reject a clip the engine scores below this; a teacher "
                             "recording that is marked down teaches the wrong thing")
    args = parser.parse_args()

    targets = {number: key(text) for number, text in PHRASES.items()}
    found: dict[int, list] = {number: [] for number in PHRASES}

    for lab in args.corpus.rglob("*.lab"):
        text = lab.read_text(encoding="utf-8").strip()
        if not text or "[" in text:
            continue
        words = key(text)
        if not (1 <= len(words) <= 12):
            continue
        joined = " ".join(words)
        for number, target in targets.items():
            if len(found[number]) >= args.candidates * 3 or not target:
                continue
            if " ".join(target) in joined:
                # Prefer utterances close to the phrase itself: less to cut, and
                # less chance the aligner drifts before reaching it.
                found[number].append((len(words) - len(target), lab))

    pipeline = AssessmentPipeline(EngineConfig(),
                                  transcript_provider=WhisperTranscriptProvider())
    args.out.mkdir(parents=True, exist_ok=True)
    kept, rejected = {}, {}

    for number, prompt in PHRASES.items():
        if number in (args.skip or []):
            rejected[number] = "left alone by request"
            continue
        candidates = sorted(found[number])[:args.candidates]
        if not candidates:
            rejected[number] = "no corpus utterance contains it"
            continue
        target = targets[number]
        best = None
        for _, lab in candidates:
            wav = lab.with_suffix(".wav")
            if not wav.exists():
                continue
            full = lab.read_text(encoding="utf-8").strip()
            try:
                alignment = pipeline.aligner.align_single(wav, full, utterance_id=lab.stem)
            except Exception:
                continue
            spoken = [w for w in alignment.words if w.label.strip()]
            labels = key(" ".join(w.label for w in spoken))
            # Where in the aligned word sequence the phrase begins.
            start_index = next((i for i in range(len(labels) - len(target) + 1)
                                if labels[i:i + len(target)] == target), None)
            if start_index is None:
                continue
            span = spoken[start_index:start_index + len(target)]
            samples = read_mono(wav, 16000)
            pad = int(0.06 * 16000)
            clip = samples[max(int(span[0].start_s * 16000) - pad, 0):
                           min(int(span[-1].end_s * 16000) + pad, len(samples))]
            if len(clip) < 3200:
                continue
            # Every candidate for a phrase scored through the same temporary
            # path, so promoting a new best deleted the file it had just been
            # given. The winning clip is kept in memory instead and written once.
            temporary = args.out / f".candidate_{number}.wav"
            sf.write(str(temporary), clip, 16000, subtype="PCM_16")
            try:
                result = pipeline.assess(reference_audio=None, learner_audio=temporary,
                                         target_text=prompt, utterance_id=f"p{number}")
                score = result.overall_score
            except Exception:
                continue
            finally:
                temporary.unlink(missing_ok=True)
            if best is None or score > best[0]:
                best = (score, clip, lab.stem, len(clip) / 16000)

        if best is None:
            rejected[number] = "no candidate could be aligned and cut"
            continue
        score, clip, source, seconds = best
        if score < args.min_score:
            rejected[number] = f"best candidate scored only {score:.0%}"
            continue
        sf.write(str(args.out / f"phrase_{number}.wav"), clip, 16000, subtype="PCM_16")
        kept[number] = (score, source, seconds)
        print(f"  {number:>2} {prompt:<24} {score:>5.0%}  {seconds:.2f}s  {source}",
              flush=True)

    print(f"\n{len(kept)} of {len(PHRASES)} phrases have a native reference")
    if rejected:
        print("\nwithout one:")
        for number, why in sorted(rejected.items()):
            print(f"  {number:>2} {PHRASES[number]:<24} {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
