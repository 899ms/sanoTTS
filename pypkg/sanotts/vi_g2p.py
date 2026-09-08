"""Vietnamese text -> the espeak-shaped IPA the `vi` piperlite voice is keyed on.

`vi-vais1000-1p46m` was distilled from a Piper/VITS teacher whose front end is
espeak-ng, so the target is the symbol stream that voice was trained on, not
"correct Vietnamese IPA". This module reaches that stream with hand-written
rules and no espeak-ng, which is the point: espeak-ng is GPL-3.0 and has no
output exception.

**Why not misaki's `vi.py`.** misaki ships a Vietnamese G2P and the repository
declares Apache-2.0, but `misaki/vi.py` is a port of vPhon
(<https://github.com/kirbyj/vPhon>), which is **GPL-3.0**. Diffed at
misaki `9e02a0b6` against vPhon `89d8ffe`, the two `trans()` functions are
90.5% line-identical after stripping comments, and 33 comments survive
verbatim, including dated ones ("Modified 20 Sep 2008 to fix aberrant 33
error", "There is also this reverse fronting, see Thompson 1965:94 ff."). Using
it would trade espeak-ng's GPL-3.0 for vPhon's. It is also unusable on its own
terms: its `[vi]` extra pulls in `underthesea`, `spacy` and
`spacy-curated-transformers`, and its output alphabet (`ʐ` for `r`, `ɓ`/`ɗ`,
`ŋ͡m`, tone digits 1-6 in a different slot) is not the alphabet this voice's
`phoneme_id_map` is keyed on. Nothing from either project is used here.

**Provenance of the rules.** Vietnamese orthography is a closed, enumerable
syllable system: an onset from a fixed set, an optional labiovelar on-glide, a
nucleus, an optional off-glide or coda, and a tone carried entirely by the
diacritic. The tables below are written from that description of the language,
in the Hanoi variety this voice speaks (`d`, `gi` and `r` all merge to /z/;
`s` and `x` both to /s/; `tr` and `ch` both to /tɕ/). espeak-ng's `vi_rules`,
`vi_list` and the espeak source tree were never read; espeak-ng was run only as
a black-box oracle, to settle which *codepoints* it prints for each phoneme --
an encoding question, since the voice's table is keyed on espeak's spelling and
not on any standard -- and to measure the disagreement afterwards.

That encoding is idiosyncratic and is reproduced deliberately, not endorsed:
`t` prints as dental `t̪` while `th` prints as plain `t`; `ư` prints as `y`;
a final `c`/`ch` prints as `c` except after a rounded back vowel, where it is
`k`; the `anh`/`ach` nucleus prints as `e-`, with an ASCII hyphen inside the
phoneme string; and the tone slot -- which sits after the nucleus and before
any coda -- prints `ɜ` for the rising `sắc` tone rather than a digit. All of
that is what the student saw during distillation, so all of it has to be
matched.

**Where this deliberately differs from espeak-ng.** espeak-ng carries a
function-word list that marks perhaps a fifth of Vietnamese syllables with
secondary rather than primary stress. That list is data inside espeak-ng and is
not reproduced; every syllable here takes a primary mark. It is the largest
single source of divergence and the evidence file reports agreement both with
the stress marks and with them stripped.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np

from . import frontend

logger = logging.getLogger("sanotts.vi_g2p")


class ViG2PError(RuntimeError):
    """`kind` separates an expected rejection (empty input) from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


PUNCTUATION = frontend.PUNCTUATION_MARKS
_PUNCT_REWRITE: dict[str, str] = {"“": '"', "”": '"', "‘": "'", "’": "'"}
_PUNCT_DROPPED: frozenset[str] = frozenset("—–…")

STRESS_PRIMARY = "ˈ"

# --- tone -------------------------------------------------------------------
# The six tones, keyed by the combining diacritic Vietnamese writes them with.
# NFD puts the tone mark last, after any vowel-quality mark (the breve of `ă`,
# the circumflex of `â ê ô`, the horn of `ơ ư`), so the two are separable.
TONE_MARKS: dict[str, str] = {
    "":       "",    # ngang, unmarked, and espeak prints nothing for it
    "̀": "2",   # huyền, grave
    "́": "ɜ",   # sắc, acute -- espeak prints ɜ here, not a digit
    "̉": "4",   # hỏi, hook above
    "̃": "5",   # ngã, tilde
    "̣": "6",   # nặng, dot below
}
_TONE_CHARS = frozenset(TONE_MARKS) - {""}

# Quality diacritics that are part of the letter rather than the tone.
_QUALITY_MARKS = frozenset({"̆", "̂", "̛"})   # breve, circumflex, horn

# --- onsets -----------------------------------------------------------------
# Longest first. Hanoi Vietnamese: d/gi/r -> z, s/x -> s, tr/ch -> tʃ,
# th -> t and t -> t̪ (espeak's own spelling of the dental/aspirated pair).
ONSETS: tuple[tuple[str, str], ...] = (
    ("ngh", "ŋ"),
    ("ng", "ŋ"), ("nh", "ɲ"), ("ch", "tʃ"), ("tr", "tʃ"),
    ("th", "t"), ("ph", "f"), ("kh", "x"), ("gh", "ɣ"), ("gi", "z"),
    ("qu", "kw"),
    ("b", "b"), ("c", "k"), ("d", "z"), ("đ", "ɗ"), ("g", "ɣ"),
    ("h", "h"), ("k", "k"), ("l", "l"), ("m", "m"), ("n", "n"),
    ("p", "p"), ("q", "k"), ("r", "z"), ("s", "s"), ("t", "t̪"),
    ("v", "v"), ("x", "s"),
    # Letters outside the Vietnamese alphabet, in loans and names.
    ("f", "f"), ("j", "z"), ("w", "v"), ("z", "z"),
)
_ONSET_MAP: dict[str, str] = dict(ONSETS)
_MAX_ONSET = max(len(key) for key, _ in ONSETS)

# --- codas ------------------------------------------------------------------
CODAS: tuple[tuple[str, str], ...] = (
    ("ng", "ŋ"), ("nh", "ɲ"), ("ch", "c"),
    ("c", "c"), ("m", "m"), ("n", "n"), ("p", "p"), ("t", "t̪"),
)
_CODA_MAP: dict[str, str] = dict(CODAS)
_MAX_CODA = max(len(key) for key, _ in CODAS)

# `c`/`ch` print as `k` rather than `c` after a rounded back vowel: `học` is
# `hɔ6k`, `độc` is `ɗo6k`, while `cục` is `ku6c` and `thích` is `tiɜc`.
_ROUNDED_BACK = frozenset({"ɔ", "o"})

# --- nuclei -----------------------------------------------------------------
# Written with the base vowel letters after the tone has been stripped, so
# `ấ` and `ầ` both arrive here as `â`. Longest first.
NUCLEI: tuple[tuple[str, str], ...] = (
    # rising diphthongs. The `ia`/`ưa`/`ua` spellings are the open-syllable
    # variants of `iê`/`ươ`/`uô`; espeak keeps `uə` and `yə` in both, but
    # lowers `iə` to `iɛ` as soon as anything follows the nucleus.
    ("iê", "iɛ"), ("yê", "iɛ"), ("ia", "iə"), ("ya", "iə"),
    ("ươ", "yə"), ("ưa", "yə"),
    ("uô", "uə"), ("ua", "uə"),
    ("a", "aː"), ("ă", "a"), ("â", "ə"),
    ("e", "ɛ"), ("ê", "e"),
    ("i", "i"), ("y", "i"),
    ("o", "ɔ"), ("ô", "o"), ("ơ", "əː"),
    ("u", "u"), ("ư", "y"),
)
_NUCLEUS_MAP: dict[str, str] = dict(NUCLEI)
_MAX_NUCLEUS = max(len(key) for key, _ in NUCLEI)

# `a` fronts and shortens before the palatal codas: `anh` is `e-ɲ`, `ach` is
# `e-c`. The trailing hyphen is espeak's own spelling of that vowel and is a
# real id in the voice's table.
_PALATAL_CODAS = frozenset({"nh", "ch"})
_PALATAL_NUCLEUS: dict[str, str] = {"a": "e-", "ă": "e-", "e": "ɛ"}

# Off-glides. The written vowel is the glide; espeak prints `j` and `w` --
# except after `â`, where `ây` is one unit `əɪ` with the tone *after* it, and
# `âu` prints a `1` in the tone slot when the tone is the unmarked ngang.
_OFFGLIDES: dict[str, str] = {"i": "j", "y": "j", "o": "w", "u": "w"}

# The two mid vowels that take espeak's `1` filler in the tone slot when the
# tone is the unmarked ngang and a `w` off-glide follows.
_NGANG_FILLER_VOWELS = frozenset({"ə", "e"})

# Vowel letters, base and decorated, used to find the nucleus.
_VOWEL_LETTERS = frozenset("aăâeêioôơuưy")

_WORD_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

# --- numerals ---------------------------------------------------------------
_UNITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
_SCALES = ((1_000_000_000, "tỷ"), (1_000_000, "triệu"), (1_000, "nghìn"))


def spell_number(value: int) -> list[str]:
    """Vietnamese for a non-negative integer, as a list of syllables.

    Regular apart from three sandhi rules every grammar states: `mười` becomes
    `mươi` after a multiplier, `một` becomes `mốt` in the units slot after a
    non-zero tens digit, and `năm` becomes `lăm` there.
    """
    if value < 0:
        raise ViG2PError("value", f"spell_number wants a non-negative int, got {value}")
    if value < 10:
        return [_UNITS[value]]
    if value < 20:
        unit = value - 10
        if unit == 0:
            return ["mười"]
        if unit == 5:
            return ["mười", "lăm"]
        return ["mười", _UNITS[unit]]
    if value < 100:
        tens, unit = divmod(value, 10)
        words = [_UNITS[tens], "mươi"]
        if unit == 1:
            words.append("mốt")
        elif unit == 5:
            words.append("lăm")
        elif unit:
            words.append(_UNITS[unit])
        return words
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        words = [_UNITS[hundreds], "trăm"]
        if rest == 0:
            return words
        if rest < 10:
            words.append("lẻ")
        words.extend(spell_number(rest))
        return words
    for scale, name in _SCALES:
        if value >= scale:
            count, rest = divmod(value, scale)
            words = spell_number(count) + [name]
            if rest:
                words.extend(spell_number(rest))
            return words
    return [_UNITS[int(digit)] for digit in str(value)]


def split_tone(word: str) -> tuple[str, str]:
    """One syllable -> (letters with the tone diacritic removed, tone symbol).

    Works on the NFD form and keeps the quality marks, so `ấy` comes back as
    `ây` plus the `sắc` symbol. A syllable carrying two tone marks is a typo or
    a non-Vietnamese string; the last one wins, which is what the orthography
    implies for the composed form.
    """
    decomposed = unicodedata.normalize("NFD", word)
    tone = ""
    kept: list[str] = []
    for char in decomposed:
        if char in _TONE_CHARS:
            tone = TONE_MARKS[char]
            continue
        kept.append(char)
    return unicodedata.normalize("NFC", "".join(kept)), tone


def _split_onset(letters: str) -> tuple[str, str, str]:
    """letters -> (onset IPA, on-glide IPA, the rime's letters).

    `qu` is an onset plus the glide, except before `uô`, where espeak reads the
    `u` as part of the nucleus (`quốc` is `kuəɜc`, not `kwoɜc`).
    """
    for width in range(min(_MAX_ONSET, len(letters)), 0, -1):
        head = letters[:width]
        if head not in _ONSET_MAP:
            continue
        rest = letters[width:]
        if head == "qu":
            if letters[2:4] == "uô" or letters[1:3] == "uô":
                # `quô...`: the `u` belongs to the nucleus, so `quốc` comes out
                # `kuəɜc` and not `kwoɜc`.
                return "k", "", letters[1:]
            return "k", "w", letters[2:]
        if head == "q":
            # A `q` with no `u` after it is not Vietnamese; read it as /k/ and
            # let the rime scan decide whether the rest is a syllable at all.
            return "k", "", rest
        if head == "gi":
            # `gi` spells /z/ before a vowel (`gia` -> `zaː`) but /zi/ when
            # nothing or only a coda follows (`gì` -> `zi2`, `gìn` -> `zin`):
            # the `i` doubles as the nucleus. Give the rime its `i` back.
            if not rest or rest[0] not in _VOWEL_LETTERS:
                return "z", "", "i" + rest
            return "z", "", rest
        onset = _ONSET_MAP[head]
        if not rest:
            # An onset with no rime is not a syllable; treat the whole string
            # as a rime instead and let the nucleus scan fail loudly.
            return "", "", letters
        return onset, "", rest
    return "", "", letters


def _split_rime(rime: str) -> tuple[str, str, str, str] | None:
    """rime letters -> (on-glide, nucleus letters, off-glide letter, coda letters).

    Returns None when the string is not a Vietnamese rime, so the caller can
    report it rather than emit something invented.
    """
    coda = ""
    for width in range(min(_MAX_CODA, len(rime) - 1), 0, -1):
        candidate = rime[len(rime) - width:]
        if candidate in _CODA_MAP and any(c in _VOWEL_LETTERS for c in rime[:len(rime) - width]):
            coda = candidate
            break
    body = rime[:len(rime) - len(coda)] if coda else rime
    if not body:
        return None

    onglide = ""
    # `oa oă oe` and `uâ uê uơ uy uy...`: the first letter is the labiovelar
    # glide, not the nucleus. `ua`/`uô`/`uơ` are nuclei in their own right and
    # are matched by the nucleus table first.
    if len(body) >= 2 and body[:2] not in _NUCLEUS_MAP:
        if body[0] == "o" and body[1] in "aăe":
            onglide, body = "w", body[1:]
        elif body[0] == "u" and body[1] in "âêya":
            onglide, body = "w", body[1:]

    for width in range(min(_MAX_NUCLEUS, len(body)), 0, -1):
        head = body[:width]
        if head in _NUCLEUS_MAP:
            tail = body[width:]
            if not tail:
                return onglide, head, "", coda
            if len(tail) == 1 and tail in _OFFGLIDES and not coda:
                return onglide, head, tail, coda
            return None
    return None


def syllable_to_ipa(syllable: str) -> str | None:
    """One Vietnamese syllable (any case) -> espeak-shaped IPA, or None.

    None means the string is not a Vietnamese syllable; the caller reports it
    instead of guessing a pronunciation for it.
    """
    letters, tone = split_tone(syllable.lower())
    if not letters:
        return None
    onset, glide, rime = _split_onset(letters)
    parts = _split_rime(rime)
    if parts is None:
        return None
    onglide, nucleus_letters, offglide_letter, coda_letters = parts
    glide = glide or onglide

    if coda_letters in _PALATAL_CODAS and nucleus_letters in _PALATAL_NUCLEUS:
        nucleus = _PALATAL_NUCLEUS[nucleus_letters]
    elif nucleus_letters == "a" and offglide_letter in ("y", "u"):
        # `ay` and `au` spell the short /ă/, `ai` and `ao` the long /aː/ --
        # the off-glide letter is the only thing that distinguishes them.
        nucleus = "a"
    elif nucleus_letters in ("ia", "ya") and (offglide_letter or coda_letters):
        nucleus = "iɛ"
    elif nucleus_letters == "ư" and offglide_letter == "u":
        # `ưu` prints `iw`, not the `yw` the letter `ư` takes everywhere else.
        nucleus = "i"
    else:
        nucleus = _NUCLEUS_MAP[nucleus_letters]

    offglide = _OFFGLIDES[offglide_letter] if offglide_letter else ""
    coda = ""
    if coda_letters:
        coda = _CODA_MAP[coda_letters]
        if coda == "c" and nucleus in _ROUNDED_BACK:
            coda = "k"

    if nucleus_letters == "â" and offglide_letter in ("y", "i"):
        # `ây`: one nucleus `əɪ`, and the tone comes after it, not before.
        return f"{onset}{glide}{STRESS_PRIMARY}əɪ{tone}"
    if not tone and offglide == "w" and nucleus in _NGANG_FILLER_VOWELS:
        # `âu` and `êu` on the unmarked ngang tone print a `1` in the tone
        # slot -- `câu` is `kə1w`, `kêu` is `ke1w` -- where every other
        # nucleus leaves the slot empty (`nhau` is `ɲaw`, `lưu` is `liw`).
        tone = "1"
    return f"{onset}{glide}{STRESS_PRIMARY}{nucleus}{tone}{offglide}{coda}"


def split_syllables(word: str) -> list[str] | None:
    """A solid token -> the Vietnamese syllables it is made of, or None.

    Vietnamese writes one syllable per token, so this only fires for the
    loanwords that break the rule (`ôtô`, `capô`). Prefixes are tried shortest
    first, which is onset maximisation: `capô` comes out `ca|pô` rather than
    the equally well-formed `cap|ô`.
    """
    if not word:
        return []
    for cut in range(1, len(word) + 1):
        if syllable_to_ipa(word[:cut]) is None:
            continue
        rest = split_syllables(word[cut:])
        if rest is not None:
            return [word[:cut]] + rest
    return None


def _tokenize(text: str) -> list[tuple[str, str]]:
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


def _letter_name_syllables(word: str) -> list[str]:
    """A non-Vietnamese string read out as Vietnamese letter names.

    espeak-ng hands unknown Latin strings to its English rules, which this
    package cannot do without espeak; spelling them out is the honest
    alternative to inventing a pronunciation, and it keeps the phoneme stream
    inside the alphabet the voice was trained on.
    """
    names = {
        "a": "a", "b": "bê", "c": "xê", "d": "dê", "e": "e", "f": "ép",
        "g": "giê", "h": "hát", "i": "i", "j": "gi", "k": "ca", "l": "e-lờ",
        "m": "em", "n": "en", "o": "o", "p": "pê", "q": "quy", "r": "e-rờ",
        "s": "ét", "t": "tê", "u": "u", "v": "vê", "w": "vê kép", "x": "ích",
        "y": "i", "z": "dét",
    }
    out: list[str] = []
    for char in word:
        name = names.get(char)
        if name is None:
            continue
        out.extend(name.split(" "))
    return out


def phonemize_to_espeak_ipa(text: str) -> tuple[str, str]:
    """Vietnamese text -> (espeak-shaped IPA, the input it could not read).

    The second value lists every whitespace-delimited token that is not a
    Vietnamese syllable sequence, plus any character outside the punctuation
    set. Those tokens are still spoken -- as letter names -- but the caller is
    told, because a silently anglicised name is exactly the failure a MOS score
    cannot see.
    """
    if not isinstance(text, str):
        raise ViG2PError("type", f"text must be str, got {type(text)!r}")
    if not text.strip():
        raise ViG2PError("empty", "text must be a non-empty string")

    pieces: list[str] = []
    unreadable: list[str] = []
    for kind, value in _tokenize(unicodedata.normalize("NFC", text)):
        if kind == "space":
            pieces.append(" ")
        elif kind == "punct":
            pieces.append(value)
        elif kind == "other":
            unreadable.append(value)
        elif kind == "number":
            pieces.append(" ".join(
                syllable_to_ipa(syllable) or "" for syllable in spell_number(int(value))))
        else:
            ipa = syllable_to_ipa(value)
            if ipa is not None:
                pieces.append(ipa)
                continue
            parts = split_syllables(value.lower())
            if parts is not None and len(parts) > 1:
                pieces.append(" ".join(syllable_to_ipa(part) or "" for part in parts))
                continue
            unreadable.append(value)
            spelled = [syllable_to_ipa(name) for name in _letter_name_syllables(value.lower())]
            pieces.append(" ".join(part for part in spelled if part))
    ipa = re.sub(r" {2,}", " ", "".join(pieces)).strip()
    if not ipa:
        raise ViG2PError("empty", f"phonemization produced no symbols for text: {text!r}")
    return ipa, " ".join(unreadable)


def text_to_phoneme_ids(text: str, table: frontend.PhonemeTable) -> tuple[np.ndarray, str]:
    """Drop-in for `frontend.text_to_phoneme_ids`, with the unmapped symbols."""
    ipa, unreadable = phonemize_to_espeak_ipa(text)
    symbols = list(unicodedata.normalize("NFD", ipa))
    missing = "".join(symbol for symbol in symbols if symbol not in table.id_map)
    unmapped = " ".join(part for part in (unreadable, missing) if part)
    ids = frontend.phonemes_to_ids(symbols, table)
    if len(ids) <= 3:
        raise ViG2PError("empty", f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64), unmapped


def producible_symbols() -> set[str]:
    """Every codepoint these rules can emit, NFD-decomposed."""
    # The literals are the ones no table holds: the space and stress mark that
    # frame a syllable, the `ə`+`ɪ` of the `ây` nucleus, the `1` filler, and the
    # `k` a final `c` becomes after a rounded back vowel.
    produced: set[str] = {" ", STRESS_PRIMARY, "1", "ə", "ɪ", "k"}
    produced.update(PUNCTUATION)
    for value in TONE_MARKS.values():
        produced.update(value)
    for mapping in (_ONSET_MAP, _CODA_MAP, _NUCLEUS_MAP, _PALATAL_NUCLEUS, _OFFGLIDES):
        for value in mapping.values():
            produced.update(unicodedata.normalize("NFD", value))
    return {symbol for symbol in produced if symbol}


def coverage_report(table: frontend.PhonemeTable) -> dict[str, object]:
    """What these rules can emit, and what this voice's table refuses."""
    produced = producible_symbols()
    unmappable = sorted(symbol for symbol in produced if symbol not in table.id_map)
    return {
        "espeak_voice": table.espeak_voice,
        "table_size": len(table.id_map),
        "producible_symbols": len(produced),
        "grapheme_inventory": {
            "onsets": len(ONSETS),
            "nuclei": len(NUCLEI),
            "codas": len(CODAS),
            "tones": len(TONE_MARKS),
        },
        "unmappable_symbols": unmappable,
        "unmappable_codepoints": [f"U+{ord(s):04X}" for s in unmappable],
    }
