"""Choose the decoder's margin threshold from measured false-substitution rate.

Every word in the aligned corpus was read correctly by a native speaker, so any
substitution the decoder claims there is a false alarm. Sweeping the margin
against that corpus turns "how confident should it be" into a number chosen
from data rather than taste.

    python scripts/calibrate_decoder.py --limit 400
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from yoruba_engine.alignment.textgrid import alignment_from_textgrid  # noqa: E402
from yoruba_engine.config import EngineConfig  # noqa: E402
from yoruba_engine.g2p.orthography import normalize  # noqa: E402
from yoruba_engine.segmental.classifier import VowelClassifier  # noqa: E402
from yoruba_engine.segmental.verifier import (  # noqa: E402
    VowelVerifier, words_from_alignment,
)
from yoruba_engine.segmental.decoder import WordDecoder  # noqa: E402


def collect(config: EngineConfig, aligned: Path, limit: int | None):
    """Return (word, [vowel distributions]) for every aligned word.

    Distributions come from the verifier rather than being rebuilt here, so the
    feature backend the classifier declares is used automatically. Rebuilding
    them locally is what made this script -- and calibrate_suspect.py -- break
    with "X has 39 features, but HistGradientBoostingClassifier is expecting
    1024" the moment the model changed. A calibration script that measures a
    different feature path than production uses is worse than no calibration.
    """
    classifier = VowelClassifier.load()
    verifier = VowelVerifier(classifier, config.segmental)
    print(f"backend: {'wav2vec2' if verifier.uses_wav2vec else 'Praat measurements'}")
    audio_index = {p.stem: p for p in config.audio_dir.rglob("*.wav")}

    collected = []
    textgrids = sorted(aligned.rglob("*.TextGrid"))[: limit or None]
    for number, textgrid in enumerate(textgrids, start=1):
        audio = audio_index.get(textgrid.stem)
        if audio is None:
            continue
        try:
            alignment = alignment_from_textgrid(textgrid, utterance_id=textgrid.stem)
            words = words_from_alignment(alignment)
            result = verifier.verify(audio, words)
        except Exception:
            continue
        for index, distributions in result.distributions.items():
            if index < len(words) and distributions:
                collected.append((normalize(words[index].word), distributions))
        if number % 100 == 0:
            print(f"  {number}/{len(textgrids)} utterances", flush=True)
    return collected, classifier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aligned", type=Path, default=PROJECT_ROOT / "output" / "aligned")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--target-rate", type=float, default=0.05,
                        help="Acceptable share of correct words contradicted")
    args = parser.parse_args()
    logging.disable(logging.INFO)

    config = EngineConfig()
    print(f"Reading alignments from {args.aligned}")
    collected, classifier = collect(config, args.aligned, args.limit)
    print(f"\n{len(collected):,} correctly-spoken words to test against\n")

    decoder = WordDecoder.load(config.mfa.dictionary, classifier=classifier)
    verdicts = [decoder.decode(word, dists) for word, dists in collected]
    comparable = [v for v in verdicts if v.scores]
    print(f"{len(comparable):,} have at least one comparable candidate "
          f"({len(comparable)/max(len(verdicts),1):.0%})")
    with_rivals = [v for v in comparable if len(v.scores) > 1]
    print(f"{len(with_rivals):,} have a minimal-pair rival at all\n")

    print(f"{'margin':>8}{'false substitutions':>22}{'rate':>9}")
    chosen = None
    for margin in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        wrong = sum(1 for v in with_rivals
                    if v.decoded != v.word and v.margin >= margin)
        rate = wrong / max(len(with_rivals), 1)
        flag = ""
        if chosen is None and rate <= args.target_rate:
            chosen, flag = margin, "   <- lowest margin meeting the target"
        print(f"{margin:>8.1f}{wrong:>22,}{rate:>9.1%}{flag}")

    print(f"\nRecommended DEFAULT_MARGIN = {chosen if chosen is not None else 'none met'}")
    if chosen is not None:
        confused = Counter(
            (v.word, v.decoded) for v in with_rivals
            if v.decoded != v.word and v.margin >= chosen
        )
        if confused:
            print("\nMost common false substitutions at that margin:")
            for (word, decoded), count in confused.most_common(8):
                print(f"  {word} -> {decoded}   x{count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
