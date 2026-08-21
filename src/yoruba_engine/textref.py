"""Build a reference from the prompt's spelling, with no recording.

The engine was written around a native-speaker recording of every prompt, which
suits a fixed corpus and not an application. An app offers whatever phrases its
curriculum contains, and recording a native speaker for each is a cost that
grows with the content.

It turns out the recording is not what the comparison needs. Yoruba orthography
encodes tone directly, and the engine already takes its tone target from the
prompt's spelling rather than from the reference recording -- that change was
made because a detector error on the reference became a false target, and a
vowel it could not read silently excused the learner. What remains of the
reference in the comparison is the word sequence, the vowel count per word, and
the transcript. All three come from the text.

**What is lost.** Two things the recording provided:

Speaker-relative pitch normalisation. Tone detection works on the learner's own
median pitch, so this costs nothing for tone; it costs the ability to say a
learner's pitch range differs from a native's, which the engine did not report.

A second opinion on the vowels. The comparison never used one -- vowel identity
is judged against the classifier, not against the reference's vowels -- so
nothing measurable is lost.

**What is gained.** A phrase needs only its diacritized spelling to be
assessable, which is what an application can actually provide.
"""

from __future__ import annotations

from pathlib import Path

from .datatypes import Tone, UtteranceAnalysis, VowelSegment, WordAnalysis
from .g2p.orthography import is_noise_token, normalize, tokenize, tone_pattern


def reference_from_text(
    target_text: str, utterance_id: str = "prompt",
) -> UtteranceAnalysis:
    """An idealised rendition of ``target_text``, built from its spelling.

    Each word carries one vowel per tone-bearing unit, with the tone the
    orthography requires. Timings are nominal: nothing downstream reads them,
    because the comparison aligns words and takes tone targets from spelling.

    Args:
        target_text: The diacritized prompt.
        utterance_id: Identifier carried into the report.

    Returns:
        An analysis shaped like one produced from audio, and accepted anywhere
        a reference is.
    """
    words: list[WordAnalysis] = []
    at = 0.0
    for index, token in enumerate(tokenize(normalize(target_text), keep_noise=False)):
        if is_noise_token(token):
            continue
        tones = tone_pattern(token)
        # A word whose spelling yields no tone-bearing unit still needs one
        # vowel, or the tone comparison finds nothing to align against.
        if not tones:
            tones = [Tone.UNKNOWN]

        vowels = [
            VowelSegment(
                label="", start_s=at + position * 0.1,
                end_s=at + (position + 1) * 0.1,
                word_index=index, detected_tone=tone,
            )
            for position, tone in enumerate(tones)
        ]
        words.append(WordAnalysis(
            word=token, start_s=at, end_s=at + len(tones) * 0.1,
            vowels=vowels, phone_count=len(tones),
        ))
        at = words[-1].end_s

    return UtteranceAnalysis(
        utterance_id=utterance_id,
        audio_path=Path(f"<text:{utterance_id}>"),
        transcript=normalize(target_text),
        words=words,
        audio_duration_s=at or None,
    )
