"""Put a recognised transcript back on the prompt's word boundaries.

Yoruba is written with spaces where the speech has none. ``Ẹ káàbọ̀`` is two
written words and one continuous gesture, so a recogniser transcribing it can
land on ``ekaabo``, ``e ka bo`` or ``ẹ kaabọ`` -- all reasonable readings of the
same sounds, none of them the prompt's segmentation.

Measured on 40 corpus utterances of 2-6 words, Whisper agreed with the
reference word count 50-80% of the time, splitting and merging freely:

    Ọmọkọ́mọ ni Dèjì        ->  ọmọ kan mọ ní déjì     (one word became three)
    Aṣọ àdìrẹ pọ̀ ...        ->  aṣọ adìrẹ́pọ̀ ...         (three became one)
    ... olójúméje ...        ->  ... olójú méje ...      (one became two)

That wrecks a word-by-word report, which pairs prompt words with heard ones: a
learner who said the phrase correctly sees words marked NOT SAID because the
tokens on either side do not line up.

**The prompt already knows how many words there are.** What a recogniser
contributes is which sounds were produced, not where the writing system puts
spaces, so the sensible thing is to keep its characters and impose the prompt's
boundaries on them. Both sides are compared with tone marks removed, because
tone marks drift in recognition and a tonal error belongs in the tone section
rather than shifting a word boundary.

This cannot invent sounds that were not produced. A prompt word with nothing
aligned to it yields an empty segment, which is dropped -- leaving a transcript
shorter than the prompt, which is exactly what "you did not say that word"
should look like downstream.
"""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher

from ..g2p.orthography import _TONE_BY_MARK, normalize


def _fold(text: str) -> tuple[str, list[int], str]:
    """Return a comparison key, its map back into the text, and the text.

    The key is lower-case, space-free and tone-free; every character in it
    records which position of the decomposed text it came from, so a cut made in
    key space can be made in the real string.
    """
    decomposed = unicodedata.normalize("NFD", normalize(text))
    keys: list[str] = []
    index: list[int] = []
    for position, character in enumerate(decomposed):
        if character in _TONE_BY_MARK or character.isspace():
            continue
        if unicodedata.category(character).startswith("P"):
            continue
        keys.append(character.lower())
        index.append(position)
    return "".join(keys), index, decomposed


def _cut_point(decomposed: str, position: int) -> int:
    """Move a cut past any combining marks, which belong to the character before.

    Cutting between a base letter and its dot-below would start the next word
    with an orphaned mark.
    """
    while position < len(decomposed) and unicodedata.combining(decomposed[position]):
        position += 1
    return position


def resegment_to_prompt(prompt: str, heard: str) -> str:
    """Re-space ``heard`` so it carries one word per word of ``prompt``.

    Args:
        prompt: The diacritized text the learner was asked to say.
        heard: What the recogniser produced, with its own spacing.

    Returns:
        The recogniser's characters, cut at the prompt's word boundaries. Words
        the learner did not produce yield nothing, so the result can be shorter
        than the prompt -- never longer.
    """
    prompt_words = normalize(prompt or "").split()
    if len(prompt_words) < 2 or not (heard or "").strip():
        # One word cannot be mis-segmented, and there is nothing to cut.
        return heard

    prompt_key, _, _ = _fold(prompt)
    heard_key, heard_index, heard_text = _fold(heard)
    if not prompt_key or not heard_key:
        return heard

    # Where each prompt word ends, in key space.
    boundaries: list[int] = []
    running = 0
    for word in prompt_words:
        running += len(_fold(word)[0])
        boundaries.append(running)

    # For every prompt position, the heard position aligned to it. Inside a
    # matched block the mapping is one to one; inside a replaced or dropped span
    # it is interpolated, which is the best available guess at where a boundary
    # falls in text the recogniser rendered differently.
    mapped = [0] * (len(prompt_key) + 1)
    for _tag, i1, i2, j1, j2 in SequenceMatcher(None, prompt_key, heard_key).get_opcodes():
        span = i2 - i1
        for i in range(i1, i2):
            offset = (i - i1) / span if span else 0.0
            mapped[i] = j1 + round(offset * (j2 - j1))
        mapped[i2] = j2
    mapped[len(prompt_key)] = len(heard_key)

    segments: list[str] = []
    previous = 0
    for boundary in boundaries:
        position = mapped[min(boundary, len(prompt_key))]
        start = (_cut_point(heard_text, heard_index[previous])
                 if previous < len(heard_index) else len(heard_text))
        end = (_cut_point(heard_text, heard_index[position])
               if position < len(heard_index) else len(heard_text))
        # Any space the recogniser put *inside* this span is spurious by
        # construction: the prompt says these characters are one word.
        piece = "".join(
            unicodedata.normalize("NFC", heard_text[start:end]).split())
        if piece:
            segments.append(piece)
        previous = position

    return " ".join(segments) if segments else heard
