"""Indonesian text -> the espeak-shaped IPA the `id` piperlite voice is keyed on.

`id-newstts-1p46m` was distilled from a Piper/VITS teacher whose front end is
espeak-ng, so the target here is not "correct Indonesian IPA" -- it is the
symbol stream that voice was trained on. This module reaches that stream with
hand-written rules and no espeak-ng, which is the whole point: espeak-ng is
GPL-3.0 and has no output exception.

**Provenance of the rules.** The grapheme-to-phoneme rules below are written
from Indonesian orthography, which is close to one-to-one: the 1972 EYD
spelling reform left five digraphs (`ng ny sy kh`, plus `gh` in Arabic loans),
three falling diphthongs (`ai au oi`) and one genuinely ambiguous letter, `e`.
Nothing was copied out of espeak-ng: `id_rules`, `id_list` and the espeak
source tree were never read. espeak-ng was run only as a black-box oracle, to
(a) settle which *codepoints* it prints for each phoneme -- an encoding
question, since the voice's `phoneme_id_map` is keyed on espeak's spelling and
not on any standard -- and (b) measure the disagreement afterwards. Both are
recorded in `experiments/evidence/idvi-espeak-free-ab-20260904.json`.

**Where this deliberately differs from espeak-ng.** espeak-ng carries a
function-word list that demotes the primary stress of about twenty grammatical
words (`di`, `ke`, `dan`, `yang`, `untuk`, `pada`, `dari`, `adalah`, `bahwa`,
...) to a secondary mark or drops it entirely. That list is data inside
espeak-ng and is not reproduced here; those words get the regular penultimate
primary stress instead. It is the single largest source of divergence and it
is measured separately in the evidence file, with the stress marks stripped, so
the segmental agreement can be read on its own.

**The `e` rule.** Indonesian writes /e/, /ɛ/ and /ə/ all as `e`. espeak-ng
resolves it positionally rather than lexically -- `kecil` comes out `kˈɛtʃil`
where the language has /kətʃil/ -- and since the voice learned espeak's
version, this module reproduces the positional rule: `e` is `ɛ` in a stressed
syllable and `ə` everywhere else. That is stated as an observation about the
oracle's behaviour, not as a claim about Indonesian.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np

from . import frontend

logger = logging.getLogger("sanotts.id_g2p")


class IdG2PError(RuntimeError):
    """`kind` separates an expected rejection (empty input) from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# espeak-ng's own punctuation set, the one frontend.PUNCTUATION_MARKS passes to
# phonemizer, so both front ends keep and drop exactly the same marks.
PUNCTUATION = frontend.PUNCTUATION_MARKS

# Curly quotes and dashes are not in the table; espeak-ng drops the dashes and
# rewrites the quotes, and this matches that so the two arms stay comparable.
_PUNCT_REWRITE: dict[str, str] = {"“": '"', "”": '"',
                                  "‘": "'", "’": "'"}
_PUNCT_DROPPED: frozenset[str] = frozenset("—–…")

VOWEL_LETTERS = frozenset("aeiou")

# Consonant graphemes, longest first. `ɡ` is U+0261 LATIN SMALL LETTER SCRIPT G,
# which is what the voice's phoneme_id_map holds -- not ASCII `g`.
CONSONANTS: tuple[tuple[str, str], ...] = (
    ("ngg", "ŋɡ"),   # ŋɡ -- "tinggal", a cluster, not a single coda
    ("ng", "ŋ"),
    ("ny", "ɲ"),          # ɲ
    ("sy", "ʃ"),          # ʃ
    ("kh", "x"),
    ("gh", "ɣ"),          # ɣ, Arabic loans
    ("b", "b"), ("c", "tʃ"), ("d", "d"), ("f", "f"), ("g", "ɡ"),
    ("h", "h"), ("j", "dʒ"), ("k", "k"), ("l", "l"), ("m", "m"),
    ("n", "n"), ("p", "p"), ("q", "k"), ("r", "r"), ("s", "s"), ("t", "t"),
    ("v", "v"), ("w", "w"), ("x", "ks"), ("y", "j"), ("z", "z"),
)
_CONSONANT_MAP: dict[str, str] = dict(CONSONANTS)
_MAX_CONSONANT = max(len(key) for key, _ in CONSONANTS)

# The three falling diphthongs. espeak-ng prints the off-glide as a lax vowel
# (`aɪ`, `aʊ`), not as `j`/`w`; the voice's table has ids for both, so this is
# an encoding choice that has to match rather than a free one.
DIPHTHONGS: dict[str, str] = {
    "ai": "aɪ",   # aɪ
    "au": "aʊ",   # aʊ
    "oi": "ɔɪ",   # ɔɪ
}

# Monophthongs. `e` is resolved later, by stress, and is absent here.
MONOPHTHONGS: dict[str, str] = {"a": "a", "i": "i", "o": "o", "u": "u"}

STRESS_PRIMARY = "ˈ"
STRESS_SECONDARY = "ˌ"

# Two-consonant onsets Indonesian permits, so "istri" splits is-tri and not
# ist-ri. Obstruent+liquid, s+stop, and the digraphs, which are single
# consonants and never split.
ONSET_CLUSTERS: frozenset[str] = frozenset({
    "pr", "br", "tr", "dr", "kr", "gr", "fr", "sr", "vr",
    "pl", "bl", "kl", "gl", "fl", "sl",
})

_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

# Indonesian numerals, needed because espeak-ng reads digits as words and a
# front end that emitted the digit ids instead would not be comparable.
_UNITS = ("nol", "satu", "dua", "tiga", "empat",
          "lima", "enam", "tujuh", "delapan", "sembilan")
_SCALES = ((1_000_000_000, "miliar"), (1_000_000, "juta"), (1_000, "ribu"))


def spell_number(value: int) -> list[str]:
    """Indonesian for a non-negative integer, as a list of words.

    Regular apart from the `se-` prefix that replaces `satu` before `puluh`,
    `belas`, `ratus` and `ribu`, and the `belas` teens.
    """
    if value < 0:
        raise IdG2PError("value", f"spell_number wants a non-negative int, got {value}")
    if value < 10:
        return [_UNITS[value]]
    if value < 20:
        if value == 10:
            return ["sepuluh"]
        if value == 11:
            return ["sebelas"]
        return [_UNITS[value - 10], "belas"]
    if value < 100:
        tens, unit = divmod(value, 10)
        words = [_UNITS[tens], "puluh"]
        if unit:
            words.append(_UNITS[unit])
        return words
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        words = ["seratus"] if hundreds == 1 else [_UNITS[hundreds], "ratus"]
        if rest:
            words.extend(spell_number(rest))
        return words
    for scale, name in _SCALES:
        if value >= scale:
            count, rest = divmod(value, scale)
            if count == 1 and name == "ribu":
                words = ["seribu"]
            else:
                words = spell_number(count) + [name]
            if rest:
                words.extend(spell_number(rest))
            return words
    # Above the largest named scale, read the digits. espeak-ng does something
    # else here; the divergence is real and is reported rather than hidden.
    return [_UNITS[int(digit)] for digit in str(value)]


def _units(word: str) -> list[tuple[str, str]]:
    """One word -> [(kind, grapheme)], kind in {"V", "C"}.

    Digraphs are one unit. A vowel pair that is one of the three diphthongs is
    one unit, so the syllabifier never splits `ai`/`au`/`oi`.
    """
    out: list[tuple[str, str]] = []
    index = 0
    length = len(word)
    while index < length:
        char = word[index]
        if char in VOWEL_LETTERS:
            pair = word[index:index + 2]
            if pair in DIPHTHONGS:
                out.append(("V", pair))
                index += 2
                continue
            out.append(("V", char))
            index += 1
            continue
        matched = False
        for width in range(min(_MAX_CONSONANT, length - index), 0, -1):
            candidate = word[index:index + width]
            if candidate in _CONSONANT_MAP:
                out.append(("C", candidate))
                index += width
                matched = True
                break
        if matched:
            continue
        out.append(("?", char))
        index += 1
    return out


def syllabify(word: str) -> list[list[tuple[str, str]]]:
    """Split one word into syllables, as lists of units.

    Standard Indonesian syllabification: every syllable has exactly one vowel
    unit, a consonant between two vowels joins the following syllable, two
    consonants split unless they are an obstruent-plus-liquid onset cluster
    (`is|tri`, not `ist|ri`), and three split after the first. An `s` plus a
    stop does *not* count -- `diskusi` is `dis|ku|si`.
    """
    units = _units(word)
    vowel_positions = [i for i, (kind, _) in enumerate(units) if kind == "V"]
    if not vowel_positions:
        return [units] if units else []

    boundaries: list[int] = []
    for first, second in zip(vowel_positions, vowel_positions[1:]):
        gap = units[first + 1:second]
        if not gap:
            boundaries.append(second)          # V-V hiatus
        elif len(gap) == 1:
            boundaries.append(second - 1)      # V-CV
        else:
            tail = "".join(grapheme for _, grapheme in gap[-2:])
            if len(gap) == 2 and tail in ONSET_CLUSTERS:
                boundaries.append(second - 2)  # V-CCV, "is|tri" -> "i|stri"
            else:
                boundaries.append(second - 1)  # VC-CV / VCC-CV
    syllables: list[list[tuple[str, str]]] = []
    start = 0
    for boundary in boundaries:
        syllables.append(units[start:boundary])
        start = boundary
    syllables.append(units[start:])
    return [syllable for syllable in syllables if syllable]


# The derivational prefixes whose vowel is a schwa. A schwa syllable does not
# host stress, which is why `menggunakan` gets no secondary mark while
# `teknologi` does, even though the two have the same shape. Written from
# Indonesian morphology, not extracted from anywhere: me-/pe- with their nasal
# assimilations, ber-/ter-/per- with their `bel-`/`pel-` allomorphs, plus ke-
# and se-. `di-` and `ku-` are deliberately absent -- their vowel is a full /i/
# or /u/, so they do host stress (`digunakan` is `dˌiɡunˈakan`).
PREFIX_SYLLABLES: frozenset[str] = frozenset({
    "me", "mem", "men", "meng", "meny",
    "pe", "pem", "pen", "peng", "peny", "per", "pel",
    "be", "ber", "bel", "te", "ter", "ke", "se",
})

# Function words that carry no stress of their own. Restricted to the
# monosyllabic grammatical clitics -- the relativiser, the coordinator, the two
# core prepositions and the definite particle -- which are unstressed in any
# description of Indonesian prosody. espeak-ng demotes a much longer list, and
# that list is data inside espeak-ng and is not reproduced here.
UNSTRESSED_WORDS: frozenset[str] = frozenset({"yang", "di", "ke", "dan", "si"})


def _stress_pattern(syllables: list[list[tuple[str, str]]], word: str) -> list[str]:
    """Which stress mark, if any, each syllable carries.

    Primary on the penultimate syllable, or on the only syllable of a
    monosyllable. One secondary mark, on the first syllable unless that is a
    prefix, in which case on the second; and only when it would land at least
    two syllables clear of the primary, since the two never sit adjacent.
    """
    count = len(syllables)
    if count == 0:
        return []
    marks = [""] * count
    if word in UNSTRESSED_WORDS:
        return marks
    primary = count - 2 if count >= 2 else 0
    marks[primary] = STRESS_PRIMARY

    def spelling(index: int) -> str:
        return "".join(grapheme for _kind, grapheme in syllables[index])

    secondary = 1 if spelling(0) in PREFIX_SYLLABLES else 0
    if secondary <= primary - 2:
        marks[secondary] = STRESS_SECONDARY
    return marks


def word_to_ipa(word: str) -> str:
    """One lowercase Indonesian word -> espeak-shaped IPA."""
    syllables = syllabify(word)
    if not syllables:
        return ""
    marks = _stress_pattern(syllables, word)
    flat = [unit for syllable in syllables for unit in syllable]
    position = 0
    pieces: list[str] = []
    for syllable, mark in zip(syllables, marks):
        stressed = mark != ""
        emitted_mark = False
        for kind, grapheme in syllable:
            next_letter = flat[position + 1][1] if position + 1 < len(flat) else ""
            position += 1
            if kind == "V":
                if not emitted_mark:
                    pieces.append(mark)
                    emitted_mark = True
                if grapheme in DIPHTHONGS:
                    pieces.append(DIPHTHONGS[grapheme])
                elif grapheme == "e":
                    pieces.append("ɛ" if stressed else "ə")   # ɛ / ə
                elif grapheme == "o" and next_letter == "r":
                    pieces.append("ɔ")   # lowered before a rhotic: kantor, orang
                else:
                    pieces.append(MONOPHTHONGS[grapheme])
            elif kind == "C":
                pieces.append(_CONSONANT_MAP[grapheme])
            else:
                pieces.append(grapheme)
    return "".join(pieces)


def _tokenize(text: str) -> list[tuple[str, str]]:
    """Text -> [(kind, value)] with kind in {"word", "number", "punct", "space"}.

    Everything that is neither a word nor one of espeak's punctuation marks is
    returned as `"other"` so the caller can see it rather than lose it.
    """
    tokens: list[tuple[str, str]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            tokens.append(("space", " "))
            index += 1
            continue
        match = _WORD_RE.match(text, index)
        if match is not None:
            value = match.group(0)
            tokens.append(("number" if value[0].isdigit() else "word", value))
            index = match.end()
            continue
        if char in _PUNCT_REWRITE:
            tokens.append(("punct", _PUNCT_REWRITE[char]))
        elif char in _PUNCT_DROPPED:
            pass
        elif char in PUNCTUATION:
            tokens.append(("punct", char))
        else:
            tokens.append(("other", char))
        index += 1
    return tokens


def phonemize_to_espeak_ipa(text: str) -> tuple[str, str]:
    """Indonesian text -> (espeak-shaped IPA, characters this module ignored).

    The second value is every input character that produced no phoneme and is
    not one of the marks espeak-ng itself drops. Nothing is discarded quietly.
    """
    if not isinstance(text, str):
        raise IdG2PError("type", f"text must be str, got {type(text)!r}")
    if not text.strip():
        raise IdG2PError("empty", "text must be a non-empty string")

    pieces: list[str] = []
    ignored: list[str] = []
    for kind, value in _tokenize(unicodedata.normalize("NFC", text)):
        if kind == "space":
            pieces.append(" ")
        elif kind == "punct":
            pieces.append(value)
        elif kind == "other":
            ignored.append(value)
        elif kind == "number":
            words = spell_number(int(value))
            pieces.append(" ".join(word_to_ipa(word) for word in words))
        else:
            lowered = value.lower()
            for _kind, grapheme in _units(lowered):
                if _kind == "?":
                    ignored.append(grapheme)
            pieces.append(word_to_ipa(lowered))
    ipa = re.sub(r" {2,}", " ", "".join(pieces)).strip()
    if not ipa:
        raise IdG2PError("empty", f"phonemization produced no symbols for text: {text!r}")
    return ipa, "".join(ignored)


def text_to_phoneme_ids(text: str, table: frontend.PhonemeTable) -> tuple[np.ndarray, str]:
    """Drop-in for `frontend.text_to_phoneme_ids`, with the unmapped symbols.

    `unmapped` collects both the characters the rules ignored and any produced
    symbol this voice's `phoneme_id_map` has no id for; the espeak path logs
    and discards the latter, here the caller gets to see them.
    """
    ipa, ignored = phonemize_to_espeak_ipa(text)
    symbols = list(unicodedata.normalize("NFD", ipa))
    unmapped = ignored + "".join(s for s in symbols if s not in table.id_map)
    ids = frontend.phonemes_to_ids(symbols, table)
    if len(ids) <= 3:
        raise IdG2PError("empty", f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64), unmapped


def producible_symbols() -> set[str]:
    """Every codepoint these rules can emit, NFD-decomposed.

    Enumerated from the tables rather than reasoned about, so a new rule cannot
    quietly introduce a symbol no voice has an id for.
    """
    produced: set[str] = {" ", STRESS_PRIMARY, STRESS_SECONDARY}
    produced.update(PUNCTUATION)
    for value in _CONSONANT_MAP.values():
        produced.update(unicodedata.normalize("NFD", value))
    for value in list(DIPHTHONGS.values()) + list(MONOPHTHONGS.values()):
        produced.update(unicodedata.normalize("NFD", value))
    produced.update({"ɛ", "ə", "ɔ"})
    return produced


def coverage_report(table: frontend.PhonemeTable) -> dict[str, object]:
    """What these rules can emit, and what this voice's table refuses."""
    produced = producible_symbols()
    unmappable = sorted(symbol for symbol in produced if symbol not in table.id_map)
    return {
        "espeak_voice": table.espeak_voice,
        "table_size": len(table.id_map),
        "producible_symbols": len(produced),
        "grapheme_inventory": {
            "consonants": len(CONSONANTS),
            "diphthongs": len(DIPHTHONGS),
            "monophthongs": len(MONOPHTHONGS) + 1,   # + the positional `e`
        },
        "unmappable_symbols": unmappable,
        "unmappable_codepoints": [f"U+{ord(s):04X}" for s in unmappable],
    }
