import logging, sys
from pathlib import Path
sys.path.insert(0, "src")
from yoruba_engine.config import EngineConfig
from yoruba_engine.pipeline import AssessmentPipeline
logging.disable(logging.WARNING)
TARGET = "Àbùbùtán bíi omi òkun."
REF = Path("data/corpus/wavs/yof_00295_00099759266.wav")
SAID = {"my_attempt.wav":  "àbùbùtán bíi omi òkun  (CORRECT)",
        "my_attempt1.wav": "àbùbùtu bíi omi ọkàn",
        "my_attempt2.wav": "àbùbùtan omi ọkàn",
        "my_attempt3.wav": "àbùbùtan omi ọkun"}
SC = Path(__file__).resolve().parent.parent / "tests" / "attempts"
p = AssessmentPipeline(EngineConfig())
print(f"vowel backend: {'wav2vec2' if p.vowel_verifier and p.vowel_verifier.uses_wav2vec else 'Praat'}\n")
for name, said in SAID.items():
    print("=" * 72); print(f"{name}   truth: {said}")
    try:
        r = p.assess(reference_audio=REF, learner_audio=SC / name, target_text=TARGET)
    except Exception as e:
        print(f"  FAILED {type(e).__name__}: {e}"); continue
    print(f"  transcript    : {r.learner_transcript}")
    print(f"  substitutions : {r.substitutions or '(none)'}")
    print(f"  GOP flagged   : {r.poorly_pronounced or '(none)'}")
    print(f"  not detected  : {r.undetected_words or '(none)'}")
    print(f"  check vowels  : {r.mispronounced_words or '(none)'}")
    print(f"  text {r.text_similarity:.0%}  tone {r.tone_similarity:.0%}  overall {r.overall_score:.0%}")
