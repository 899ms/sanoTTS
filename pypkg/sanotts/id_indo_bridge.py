"""indo-g2p's Indonesian, written in the codepoints the `id` voice was keyed on.

``id_indo_g2p`` is a faithful port and emits ordinary IPA. This module is the
part that is *not* faithful to anything, and it is separate so that every
decision in it can be argued with one at a time. ``id-newstts-1p46m`` was
distilled from a Piper/VITS teacher whose front end is espeak-ng, so its
``phoneme_id_map`` is keyed on the codepoints espeak-ng printed and on no
standard, and its acoustic model has only ever seen the token distribution
espeak-ng produces. Four decisions follow from that, and each was measured on
espeak-ng's own output over the 438-sentence development corpus before it was
made:

1. **``g`` becomes ``ɡ`` (U+0261).** Not cosmetic. The voice's table has an id
   for both, and ASCII ``g`` is id 154, which is past the deployed embedding's
   ``vocab_size`` of 122 -- it would be remapped to id 59, the schwa. espeak-ng
   prints U+0261 (194 times in the corpus) and never the ASCII one.

2. **The non-schwa ``e`` becomes ``ɛ``.** espeak-ng printed plain ``e`` 9 times
   against 401 ``ɛ`` and 887 ``ə``, so id 18 is a token this voice has
   effectively never seen. ``ɛ`` is the reading it learned for a non-schwa
   ``e`` and it stays.

3. **Stress stays on the penultimate syllable.** espeak-ng put the primary mark
   on the penultimate nucleus in 2,827 of 2,866 multisyllabic tokens (98.6%),
   on the final in 34 (1.2%). Standard Indonesian shifts the stress off a
   penultimate schwa onto the final syllable, and ``stress="schwa_shift"``
   does that -- but it would move the mark on a large share of ordinary words
   into a position the voice has almost never heard, so it is not the default.
   The cost of not doing it is the sequence ``ˈə``, which espeak-ng did print,
   5 times. Both tokens are individually common; only the pairing is rare.
   That is the smaller of the two shifts, and it keeps this arm a clean
   isolation of the vowel question.
   *Measured afterwards, the choice barely matters.* 11.9% of this path's
   stress marks land on a schwa, and rendering and scoring both variants put
   ``schwa_shift`` at 0.2299 word error rate against ``penultimate``'s 0.2320
   -- 0.002, well inside the noise. The stressed schwa was the obvious suspect
   for the regression named below and it is not the cause.

4. **Which stress rule, and which words go unstressed, are copied from
   ``id_g2p``** -- the same penultimate primary, the same single secondary
   mark, the same five monosyllabic clitics. Nothing about stress is being
   tested here, so it is held constant across the two espeak-free arms and only
   the segments differ.

``glottal=False`` turns the glottal stops off while keeping the schwa
corrections, because the two changes have very different risk: espeak-ng
printed ``ʔ`` **0 times** in the corpus, so id 109 is a token this voice barely
saw, while ``ə`` is its fourth most common symbol. If the arm regresses, that
flag says which half did it.

**The outcome, so it is not buried.** On 100 held-out sentences this path costs
Whisper word error rate rather than saving it: 0.2320 against 0.2037 for
``id_g2p``, a paired +0.0283 whose 95% confidence interval excludes zero. The
front end is more correct and the voice is less intelligible, because the voice
was distilled on the incorrect one. Nothing here is the default. The value of
this module is as the front end of the next Indonesian training run.

Measured end to end in ``experiments/evidence/id-indo-g2p-ab-20260904.json``,
and read out in ``docs/id-indo-g2p-frontend.md``.
"""

from __future__ import annotations

import logging
import unicodedata

import numpy as np

from . import frontend, id_g2p, id_indo_g2p

logger = logging.getLogger("sanotts.id_indo_bridge")

STRESS_CHOICES = ("penultimate", "schwa_shift")

# indo-g2p's alphabet -> the codepoints espeak-ng printed for the same sound.
# Only the entries that actually differ; everything else passes through.
SYMBOL_REWRITES: dict[str, str] = {
    "g": "ɡ",    # U+0067 -> U+0261, see decision 1
    "e": "ɛ",    # U+0065 -> U+025B, see decision 2
    "q": "k",    # indo-g2p leaves `q`; espeak-ng reads it as /k/
    "é": "ɛ",    # only reachable with expand_abbr=True (letter names)
    "è": "ɛ",
}

# Every vowel that can be a syllable nucleus on the indo-g2p side, after the
# rewrites above. The stress mark goes immediately before one of these.
NUCLEI = frozenset("aiueoəɛɪʊɔ")


class IdIndoBridgeError(RuntimeError):
    """`kind` separates an expected rejection from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _rewrite(text: str) -> str:
    return "".join(SYMBOL_REWRITES.get(char, char) for char in text)


def _nucleus_index(syllable: str) -> int | None:
    """Where in a syllable the stress mark goes: before its first nucleus."""
    for index, char in enumerate(syllable):
        if char in NUCLEI:
            return index
    return None


def _stress_marks(syllables: list[str], word: str, stress: str) -> list[str]:
    """Which mark each syllable carries.

    Primary on the penultimate syllable -- or, under ``schwa_shift``, on the
    final one when the penultimate nucleus is a schwa. One secondary mark, on
    the first syllable unless its nucleus is a schwa, in which case on the
    second, and only when it lands at least two syllables clear of the primary.
    ``id_g2p`` reaches the same placement from the spelling; here the schwa is
    known outright, so the prefix list it needs is not needed.
    """
    count = len(syllables)
    if count == 0:
        return []
    marks = [""] * count
    if word in id_g2p.UNSTRESSED_WORDS:
        return marks

    def nucleus(index: int) -> str | None:
        at = _nucleus_index(syllables[index])
        return None if at is None else syllables[index][at]

    primary = count - 2 if count >= 2 else 0
    if stress == "schwa_shift" and count >= 2 and nucleus(primary) == "ə":
        primary = count - 1
    marks[primary] = id_g2p.STRESS_PRIMARY

    secondary = 1 if (count > 1 and nucleus(0) == "ə") else 0
    if secondary <= primary - 2:
        marks[secondary] = id_g2p.STRESS_SECONDARY
    return marks


def _stressed_word(syllables: list[str], word: str, stress: str) -> str:
    marks = _stress_marks(syllables, word, stress)
    out: list[str] = []
    for syllable, mark in zip(syllables, marks):
        if not mark:
            out.append(syllable)
            continue
        at = _nucleus_index(syllable)
        if at is None:                      # a syllable with no vowel at all
            out.append(syllable)
            continue
        out.append(syllable[:at] + mark + syllable[at:])
    return "".join(out)


def phonemize_to_espeak_ipa(
    text: str,
    *,
    glottal: bool = True,
    stress: str = "penultimate",
    english: bool = False,
    expand_abbr: bool = False,
) -> str:
    """Indonesian text -> espeak-shaped IPA, by way of the indo-g2p port.

    This walks the words itself rather than calling ``id_indo_g2p.convert``,
    because the stress mark has to go inside a syllable and ``convert`` returns
    the syllables already joined.
    """
    if not isinstance(text, str):
        raise IdIndoBridgeError("type", f"text must be str, got {type(text)!r}")
    if not text.strip():
        raise IdIndoBridgeError("empty", "text must be a non-empty string")
    if stress not in STRESS_CHOICES:
        raise IdIndoBridgeError("option",
                                f"stress must be one of {STRESS_CHOICES}, got {stress!r}")

    lowered = id_indo_g2p.normalize_text(text).lower()
    matches = list(id_indo_g2p.WORD_PATTERN.finditer(lowered))
    words = [match.group(0) for match in matches]
    resolutions = id_indo_g2p.resolve_collocations(words)

    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        word = match.group(0)
        phonemes, abbr = id_indo_g2p.word_to_phonemes(
            word, expand_abbr, resolutions[index], english)
        if not glottal:
            # The schwa corrections stay; only the glottal-stop rule is undone.
            phonemes = phonemes.replace("ʔ", "k")
        syllables = id_indo_g2p.to_syllables(phonemes)
        if not abbr and any(pair in phonemes for pair, _ in id_indo_g2p.DIPHTHONGS):
            syllables = [id_indo_g2p._apply_replacements(s, id_indo_g2p.DIPHTHONGS)
                         for s in syllables]
        syllables = [_rewrite(syllable) for syllable in syllables]
        parts.append(lowered[cursor:match.start()])
        parts.append(_stressed_word(syllables, word, stress))
        cursor = match.end()
    parts.append(lowered[cursor:])

    ipa = "".join(parts).strip()
    if not ipa:
        raise IdIndoBridgeError("empty",
                                f"phonemization produced no symbols for text: {text!r}")
    return ipa


def text_to_phoneme_ids(
    text: str,
    table: frontend.PhonemeTable,
    *,
    glottal: bool = True,
    stress: str = "penultimate",
    english: bool = False,
) -> tuple[np.ndarray, str]:
    """Drop-in for ``frontend.text_to_phoneme_ids``, plus the unmapped symbols.

    ``unmapped`` is every produced codepoint this voice's ``phoneme_id_map``
    has no id for. The espeak path logs and discards those; here the caller
    gets to see them.
    """
    ipa = phonemize_to_espeak_ipa(text, glottal=glottal, stress=stress, english=english)
    symbols = list(unicodedata.normalize("NFD", ipa))
    unmapped = "".join(symbol for symbol in symbols if symbol not in table.id_map)
    ids = frontend.phonemes_to_ids(symbols, table)
    if len(ids) <= 3:
        raise IdIndoBridgeError("empty",
                                f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64), unmapped


def producible_symbols() -> set[str]:
    """Every codepoint this path can emit, NFD-decomposed.

    Enumerated from the port's own tables rather than reasoned about, so a
    change upstream cannot quietly introduce a symbol no voice has an id for.
    """
    produced: set[str] = {" ", id_g2p.STRESS_PRIMARY, id_g2p.STRESS_SECONDARY}
    produced.update(id_indo_g2p._KEPT_PUNCTUATION)
    # An over-approximation on purpose: every letter, whether or not the
    # replacement table consumes it, plus everything that table produces. Too
    # many symbols makes the coverage test stricter, never weaker.
    produced.update(_rewrite("abcdefghijklmnopqrstuvwxyz"))
    for _, target in id_indo_g2p.PHONEME_REPLACEMENTS:
        produced.update(unicodedata.normalize("NFD", _rewrite(target)))
    for _, target in id_indo_g2p.DIPHTHONGS:
        produced.update(unicodedata.normalize("NFD", target))
    produced.update({"ə", "ʔ", "ɲ"})
    produced.update(_rewrite("".join(id_indo_g2p.LETTER_NAMES.values())))
    return produced


def coverage_report(table: frontend.PhonemeTable) -> dict[str, object]:
    """What this path can emit for one voice, and what its table refuses."""
    produced = producible_symbols()
    unmappable = sorted(symbol for symbol in produced if symbol not in table.id_map)
    return {
        "espeak_voice": table.espeak_voice,
        "table_size": len(table.id_map),
        "producible_symbols": len(produced),
        "symbol_rewrites": dict(SYMBOL_REWRITES),
        "upstream_revision": id_indo_g2p.UPSTREAM_REVISION,
        "english_table_vendored": id_indo_g2p.english_available(),
        "unmappable_symbols": unmappable,
        "unmappable_codepoints": [f"U+{ord(s):04X}" for s in unmappable],
    }
