"""Indonesian grapheme-to-phoneme, ported from snowfluke/indo-g2p (MIT).

Where ``id_g2p`` reproduces what espeak-ng does, this module reproduces what
Indonesian actually is. The two differ on exactly the two points
`Ampixa/sanoTTS#2 <https://github.com/Ampixa/sanoTTS/issues/2>`_ raises, and
both were re-measured against an independent oracle before any of this was
written (``experiments/evidence/id-indo-g2p-ab-20260904.json``):

* **The letter ``e`` spells two vowels**, /e/ and the pepet /ə/, and which one
  a word takes is lexical. espeak-ng resolves it positionally -- stressed
  syllable gets /ɛ/, everything else /ə/ -- and lands on the reading a curated
  dictionary gives for 68.4% of the Indonesian word types in a 28,194-sentence
  corpus. This module looks the word up and reaches 97.3%.
* **Word-final and pre-consonantal ``k`` is a glottal stop.** espeak-ng emits
  one only for words in its 151-entry ``id_list``: 1 of the 78 word types that
  need one in that corpus. This module emits all 78.

**This is a port, not a wrapper.** indo-g2p is TypeScript; the algorithm here
is a line-by-line transcription of ``src/g2p.ts``, ``src/schwa.ts``,
``src/affix.ts``, ``src/syllabifier.ts``, ``src/crf-model.ts``,
``src/normalize.ts``, ``src/number.ts``, ``src/collocations.ts`` and
``src/constants.ts`` at revision ``dd5f102cba7345ba46adef8c2ad9fa261587ea4e``,
and it is checked against that revision's own output on every word of the
evaluation corpus by ``pypkg/tests/test_id_indo_g2p.py``.

**Licences.** indo-g2p is MIT. Its data files carry two upstream projects with
them -- Wikidepia/g2p-id (MIT) for the schwa dictionary and the CRF
syllabifier, bookbot-kids/g2p_id (Apache-2.0) for the POS tagger and homograph
table -- and open-dict-data/ipa-dict (MIT) for the English table. All of that
is recorded, with what is and is not vendored here, in
``pypkg/sanotts/g2p_data/NOTICE.md``.

**What this module does not do** is speak to the voice. Its output is ordinary
IPA with no stress marks, an ASCII ``g``, and a plain ``e`` for the non-schwa
reading; the ``id`` voice's ``phoneme_id_map`` is keyed on the codepoints
espeak-ng printed. ``id_indo_bridge`` is the layer that reconciles the two, and
it is separate on purpose: this file is a faithful port and that one is a set
of encoding decisions, each of which was measured.
"""

from __future__ import annotations

import logging
import lzma
import re
from pathlib import Path

logger = logging.getLogger("sanotts.id_indo_g2p")

DATA_DIR = Path(__file__).resolve().parent / "g2p_data" / "indo_g2p"

# The upstream revision this was ported from. Recorded so a future diff has a
# fixed point to diff against.
UPSTREAM_REVISION = "dd5f102cba7345ba46adef8c2ad9fa261587ea4e"
UPSTREAM_VERSION = "0.1.2"


class IndoG2PError(RuntimeError):
    """`kind` separates an expected rejection from a bug or a missing file."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# --------------------------------------------------------------------------
# constants.ts
# --------------------------------------------------------------------------

VOWELS = frozenset("aiueo")

# Spoken name of each letter, used only when `expand_abbr` is on.
LETTER_NAMES: dict[str, str] = {
    "a": "a", "b": "bé", "c": "cé", "d": "dé", "e": "é", "f": "èf", "g": "gé",
    "h": "ha", "i": "i", "j": "jé", "k": "ka", "l": "èl", "m": "èm", "n": "èn",
    "o": "o", "p": "pé", "q": "ki", "r": "èr", "s": "ès", "t": "té", "u": "u",
    "v": "vé", "w": "wé", "x": "èks", "y": "yé", "z": "zèt",
}

# Consonant/vowel shapes a real Indonesian word can be built from. A word whose
# shape contains none of these is an abbreviation to be read out letter by
# letter.
SYLLABLE_PATTERNS: tuple[str, ...] = (
    "VK", "KV", "KVK", "VKK", "KKV", "KKVK", "KVKK", "KKKV", "KKKVK",
    "KKVKK", "KVKKK",
)

# A `k` between a vowel and a consonant is a glottal stop, as in `ba/ʔ/so`.
# `h`, `r` and `l` are deliberately absent from the following class: `kh` is a
# digraph, and `kr`/`kl` are Latin onset clusters that only appear in
# borrowings (`demokrat`, `iklan`).
GLOTTAL_STOP_PATTERN = re.compile(r"[aiueəo]k[bcdfgjkmnpqstvwxyz]")

# The one place a `k` before `l` still is a glottal stop: `-lah` is a clitic,
# so its `k` closes a root rather than opening a cluster.
GLOTTAL_BEFORE_CLITIC_PATTERN = re.compile(r"[aiueəo]k(?=lah$)")

WORD_PATTERN = re.compile(r"[a-z]+")

# `ny` is the digraph /ɲ/ only before a vowel. Indonesian words do not end in
# `ny`, so anything else that reaches here is a borrowed name (`denny`).
NASAL_DIGRAPH_PATTERN = re.compile(r"ny(?=[aiueoəéè])")

# Grapheme-to-phoneme substitutions, applied in this order. `ch` has to precede
# `c`, or every borrowed name with it would end up as `tʃh`.
PHONEME_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ch", "tʃ"), ("x", "ks"), ("c", "tʃ"), ("j", "dʒ"), ("ng", "ŋ"),
    ("sy", "ʃ"), ("kh", "x"), ("v", "f"), ("y", "j"),
)

DIPHTHONGS: tuple[tuple[str, str], ...] = (("ai", "aɪ"), ("au", "aʊ"), ("oi", "ɔɪ"))


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def _read_packed(name: str) -> str:
    """One vendored payload, xz-decompressed.

    The files are the packed strings indo-g2p keeps in `src/data/*.ts`, byte
    for byte, compressed because a wheel that ships them raw is 711 KB heavier
    for nothing. `tools/vendor_indo_g2p.py` regenerates them from a checkout
    and is the only thing that should write here.
    """
    path = DATA_DIR / f"{name}.xz"
    if not path.is_file():
        raise IndoG2PError(
            "missing_data",
            f"vendored indo-g2p data not found: {path}. Reinstall the package, or "
            f"regenerate it with tools/vendor_indo_g2p.py",
        )
    try:
        return lzma.decompress(path.read_bytes()).decode("utf-8")
    except (lzma.LZMAError, UnicodeDecodeError) as exc:
        raise IndoG2PError("corrupt_data", f"could not read {path}: {exc}") from exc


def _parse_masks(packed: str) -> dict[str, int]:
    """`word hexmask` per line, the format every schwa table uses."""
    out: dict[str, int] = {}
    for line in packed.split("\n"):
        if not line:
            continue
        space = line.rfind(" ")
        if space < 1:
            raise IndoG2PError("corrupt_data", f"malformed schwa line: {line!r}")
        try:
            out[line[:space]] = int(line[space + 1:], 16)
        except ValueError as exc:
            raise IndoG2PError("corrupt_data", f"malformed schwa mask: {line!r}") from exc
    return out


_masks: dict[str, int] | None = None
_mask_sources: dict[str, str] | None = None


def schwa_masks() -> dict[str, int]:
    """The merged schwa table, parsed on first use.

    Read weakest first so a later source wins, exactly as upstream does:
    Bookbot's lexicon fills gaps, the curated dictionary overrules it on native
    vocabulary, and the hand-written corrections overrule both.
    """
    global _masks, _mask_sources
    if _masks is not None:
        return _masks
    merged: dict[str, int] = {}
    provenance: dict[str, str] = {}
    for source, name in (("lexicon", "lexicon"),
                         ("dictionary", "schwa_dict"),
                         ("override", "schwa_overrides")):
        for word, mask in _parse_masks(_read_packed(name)).items():
            merged[word] = mask
            provenance[word] = source
    _masks, _mask_sources = merged, provenance
    return merged


def apply_mask(word: str, mask: int) -> str:
    """Rewrite the `e`s selected by `mask` as `ə`."""
    out: list[str] = []
    seen = 0
    for char in word:
        if char != "e":
            out.append(char)
            continue
        out.append("ə" if (mask >> seen) & 1 else "e")
        seen += 1
    return "".join(out)


# --------------------------------------------------------------------------
# affix.ts
# --------------------------------------------------------------------------

# Longest first, so `ter-` is tried before `te-`. Every one of these whose
# vowel is written `e` has a schwa there, which is a fact about the language
# rather than about any word list. `di-` has no `e` and contributes no bits; it
# is listed because stripping it exposes a root that can be looked up.
SCHWA_PREFIXES: tuple[str, ...] = (
    "memper", "member", "seper", "meng", "meny", "mem", "men", "peng", "peny",
    "pem", "pen", "ber", "bel", "ter", "tel", "per", "pel", "me", "pe", "be",
    "te", "se", "ke", "di",
)

SUFFIXES: tuple[str, ...] = (
    "kannya", "annya", "nya", "kan", "an", "lah", "kah", "pun", "i",
)

# What an Indonesian root may start with. This is what stops `teknologi` being
# read as `te` + `knologi`: `kn` is not a possible onset.
ROOT_ONSET = re.compile(
    r"^([aiueo]|ng[aiueo]|ny[aiueo]|sy[aiueo]|kh[aiueo]|[pbtdkgfsr]r[aiueo]"
    r"|[pbkgf]l[aiueo]|s[ptkw][aiueo]|[bcdfghjklmnpqrstvwxyz][aiueoy])"
)

MAX_AFFIX_DEPTH = 3


def affix_schwa_mask(word: str, lookup, depth: int = 0) -> int | None:
    """The schwa mask of a derived word, from its affixes.

    The dictionary lists roots, but Indonesian builds most of its vocabulary by
    affixing them, so 28% of running text misses it. An unknown root keeps its
    own `e`s and only the prefix is claimed.
    """
    if depth >= MAX_AFFIX_DEPTH:
        return None
    for prefix in SCHWA_PREFIXES:
        if not word.startswith(prefix) or len(word) <= len(prefix) + 1:
            continue
        rest = word[len(prefix):]
        if ROOT_ONSET.match(rest) is None:
            continue
        prefix_vowels = prefix.count("e")
        prefix_mask = (1 << prefix_vowels) - 1
        root_mask = _resolve_root(rest, lookup, depth)
        if root_mask is None:
            return prefix_mask
        return prefix_mask | (root_mask << prefix_vowels)
    return None


def _resolve_root(rest: str, lookup, depth: int) -> int | None:
    """The part after a prefix, peeling one suffix if that is what it takes."""
    direct = lookup(rest)
    if direct is None:
        direct = affix_schwa_mask(rest, lookup, depth + 1)
    if direct is not None:
        return direct
    for suffix in SUFFIXES:
        if not rest.endswith(suffix) or len(rest) <= len(suffix) + 1:
            continue
        stem = rest[:len(rest) - len(suffix)]
        mask = lookup(stem)
        if mask is None:
            mask = affix_schwa_mask(stem, lookup, depth + 1)
        if mask is not None:
            return mask
    return None


def apply_schwa(word: str) -> str:
    """Rewrite the `e`s of one word that are pronounced /ə/."""
    masks = schwa_masks()
    mask = masks.get(word)
    if mask is None:
        mask = affix_schwa_mask(word, masks.get)
    return word if mask is None else apply_mask(word, mask)


def schwa_source(word: str) -> str:
    """Which layer places a word: dictionary, lexicon, override, affix, rules.

    Reporting only -- nothing in the conversion path calls it. It exists so a
    disagreement can be traced to the table that caused it.
    """
    masks = schwa_masks()
    if _mask_sources is None:                    # schwa_masks() sets both
        raise IndoG2PError("corrupt_data", "the schwa provenance table was not built")
    listed = _mask_sources.get(word)
    if listed is not None:
        return listed
    return "rules" if affix_schwa_mask(word, masks.get) is None else "affix"


# --------------------------------------------------------------------------
# crf-model.ts + syllabifier.ts
# --------------------------------------------------------------------------

CONTINUE, BOUNDARY = 0, 1
CRF_CONTEXT = 5

# Transition weights, row-major over [from, to] with labels [O, S].
TRANSITIONS: tuple[float, ...] = (-3.366998, -6.859418, 32.800146, -22.573729)

_state_features: dict[str, tuple[float, float]] | None = None


def state_features() -> dict[str, tuple[float, float]]:
    """CRF state weights: `attr\\tw` (antisymmetric) or `attr\\twO\\twS`."""
    global _state_features
    if _state_features is not None:
        return _state_features
    parsed: dict[str, tuple[float, float]] = {}
    for line in _read_packed("syllabifier_state").split("\n"):
        if not line:
            continue
        fields = line.split("\t")
        try:
            if len(fields) == 2:
                weight = float(fields[1])
                parsed[fields[0]] = (weight, -weight)
            elif len(fields) == 3:
                parsed[fields[0]] = (float(fields[1]), float(fields[2]))
            else:
                raise ValueError(f"{len(fields)} fields")
        except ValueError as exc:
            raise IndoG2PError("corrupt_data",
                               f"malformed CRF feature line {line!r}: {exc}") from exc
    _state_features = parsed
    return parsed


def _attributes_at(word: str, index: int) -> list[str]:
    """The CRF attribute names for one character. These are load-bearing:
    renaming one silently invalidates every weight."""
    last = len(word) - 1
    attrs = ["bias", f"c={word[index]}"]
    if index > 0:
        attrs.append(f"c[-1:0]={word[index - 1:index + 1]}")
    if index > 1:
        attrs.append(f"c[-2:0]={word[index - 2:index + 1]}")
    if index < last:
        attrs.append(f"c[0:+1]={word[index:index + 2]}")
    if index < last - 1:
        attrs.append(f"c[0:+2]={word[index:index + 3]}")
    for n in range(1, min(CRF_CONTEXT, index) + 1):
        attrs.append(f"c[-{n}]={word[index - n]}")
    for n in range(1, min(CRF_CONTEXT, last - index) + 1):
        attrs.append(f"c[+{n}]={word[index + n]}")
    if index == 0:
        attrs.append("BOS")
    if index == last:
        attrs.append("EOS")
    return attrs


def _emissions_at(model: dict[str, tuple[float, float]], word: str,
                  index: int) -> tuple[float, float]:
    cont = bound = 0.0
    for attr in _attributes_at(word, index):
        weights = model.get(attr)
        if weights is not None:
            cont += weights[0]
            bound += weights[1]
    return cont, bound


def _decode(word: str) -> list[int]:
    """Viterbi. crfsuite compares with a strict `>`, so `O` wins a tie."""
    model = state_features()
    scores = _emissions_at(model, word, 0)
    backpointers: list[tuple[int, int]] = []
    for index in range(1, len(word)):
        cont, bound = _emissions_at(model, word, index)
        row: list[tuple[int, float]] = []
        for to in (CONTINUE, BOUNDARY):
            via_continue = scores[CONTINUE] + TRANSITIONS[CONTINUE * 2 + to]
            via_boundary = scores[BOUNDARY] + TRANSITIONS[BOUNDARY * 2 + to]
            if via_boundary > via_continue:
                row.append((BOUNDARY, via_boundary))
            else:
                row.append((CONTINUE, via_continue))
        backpointers.append((row[CONTINUE][0], row[BOUNDARY][0]))
        scores = (row[CONTINUE][1] + cont, row[BOUNDARY][1] + bound)

    tags = [CONTINUE] * len(word)
    tags[-1] = BOUNDARY if scores[BOUNDARY] > scores[CONTINUE] else CONTINUE
    for index in range(len(backpointers) - 1, -1, -1):
        tags[index] = backpointers[index][tags[index + 1]]
    return tags


# Every vowel this module emits, including the borrowed ones.
_SYLLABLE_VOWELS = re.compile(r"[aiueoəɪʊɔ]")


def _keep_affricates_whole(syllables: list[str]) -> list[str]:
    """Move a boundary that fell inside `tʃ` or `dʒ`.

    The model scores characters, so it happily cuts between the two halves of
    an affricate. A syllable ending in `t` followed by one starting with `ʃ` is
    not Indonesian; the cut belongs one character earlier.
    """
    fixed: list[str] = []
    for syllable in syllables:
        previous = fixed[-1] if fixed else None
        splits = previous is not None and (
            (previous.endswith("t") and syllable.startswith("ʃ"))
            or (previous.endswith("d") and syllable.startswith("ʒ"))
        )
        if not splits or previous is None:
            fixed.append(syllable)
            continue
        fixed[-1] = previous[:-1]
        fixed.append(previous[-1] + syllable)
        if fixed[-2] == "":
            del fixed[-2]
    return fixed


def _require_nucleus(syllables: list[str]) -> list[str]:
    """Fold away any piece with no vowel in it -- a syllable needs a nucleus."""
    fixed: list[str] = []
    for syllable in syllables:
        if fixed and _SYLLABLE_VOWELS.search(fixed[-1]) is None:
            fixed[-1] = fixed[-1] + syllable
            continue
        fixed.append(syllable)
    if len(fixed) > 1 and _SYLLABLE_VOWELS.search(fixed[-1]) is None:
        tail = fixed.pop()
        fixed[-1] = fixed[-1] + tail
    return fixed


def to_syllables(word: str) -> list[str]:
    """Split one word into syllables with the CRF model ported from g2p-id.

    The model was trained on uppercase text with the digraphs already mapped
    and no schwa marking, so `ə` and `é` are folded to `e` for tagging only.
    Both foldings are one character for one, so the boundaries still line up
    and the returned syllables keep the real characters.
    """
    if not word:
        return [""]
    tags = _decode(word.replace("é", "e").replace("ə", "e").upper())
    syllables: list[str] = []
    current = ""
    for index, char in enumerate(word):
        current += char
        if tags[index] == BOUNDARY:
            syllables.append(current)
            current = ""
    syllables.append(current)
    return _require_nucleus(_keep_affricates_whole(syllables))


# --------------------------------------------------------------------------
# number.ts + normalize.ts
# --------------------------------------------------------------------------

_ONES = ("nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh",
         "delapan", "sembilan")
_SCALES = ((1_000_000_000_000, "triliun"), (1_000_000_000, "miliar"),
           (1_000_000, "juta"), (1_000, "ribu"))
_NUMBER_LIMIT = 1_000_000_000_000_000

NUMBER_WORDS = frozenset(_ONES) | {
    "sepuluh", "sebelas", "belas", "puluh", "seratus", "ratus", "seribu",
    "ribu", "juta", "miliar", "triliun", "koma",
}


def _under_thousand(value: int) -> str:
    if value < 10:
        return _ONES[value]
    if value < 20:
        if value == 10:
            return "sepuluh"
        if value == 11:
            return "sebelas"
        return f"{_ONES[value - 10]} belas"
    if value < 100:
        tens = f"{_ONES[value // 10]} puluh"
        rest = value % 10
        return tens if rest == 0 else f"{tens} {_ONES[rest]}"
    hundreds = "seratus" if value < 200 else f"{_ONES[value // 100]} ratus"
    rest = value % 100
    return hundreds if rest == 0 else f"{hundreds} {_under_thousand(rest)}"


def spell_number(value: int) -> str:
    """Indonesian for a whole number, or the digits back if out of range."""
    if value < 0 or value >= _NUMBER_LIMIT:
        return str(value)
    if value < 1000:
        return _under_thousand(value)
    for scale, name in _SCALES:
        if value < scale:
            continue
        count = value // scale
        head = "seribu" if (count == 1 and scale == 1000) else f"{spell_number(count)} {name}"
        rest = value % scale
        return head if rest == 0 else f"{head} {spell_number(rest)}"
    return str(value)


def spell_decimal(whole: int, fraction: str) -> str:
    """A decimal, reading the fractional digits one at a time."""
    digits = " ".join(_ONES[int(d)] if d.isdigit() else d for d in fraction)
    return f"{spell_number(whole)} koma {digits}"


_KEPT_PUNCTUATION = frozenset(".,!?;:'\"()-")
_TYPOGRAPHY: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("[‘’‛]"), "'"),
    (re.compile("[“”„«»]"), '"'),
    (re.compile("[–—―]"), "-"),
    (re.compile("…"), "..."),
    (re.compile(" "), " "),
)
_SPOKEN_SYMBOLS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("%"), " persen"),
    (re.compile("&"), " dan "),
    (re.compile("@"), " at "),
    (re.compile(r"°\s*c\b", re.IGNORECASE), " derajat celsius"),
    (re.compile("°"), " derajat"),
    (re.compile(r"\+"), " plus "),
    (re.compile("="), " sama dengan "),
    (re.compile("/"), " garis miring "),
    (re.compile("[$€£]"), " dolar "),
)
_RUPIAH = re.compile(r"\brp\s*([\d.,]*\d)", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d.]*(?:,\d+)?")
_GROUPED = re.compile(r"^\d{1,3}(\.\d{3})*$")
_WHITESPACE = re.compile(r"\s+")
# Python has no \p{Letter}; str.isalpha() is the same test.


def _read_number(text: str) -> str:
    """One written number, honouring Indonesian digit grouping.

    A dotted group that is not a thousands separator -- a version, an IP
    address -- is read digit group by digit group rather than as one huge
    number. Everything here mirrors what JavaScript's `Number()` does with the
    same strings, including reading an empty part as zero.
    """
    whole, separator, fraction_text = text.partition(",")
    fraction = fraction_text if separator else None
    digits = whole.replace(".", "")
    grouped = _GROUPED.match(whole) is not None or "." not in whole
    if not grouped:
        return " titik ".join(spell_number(int(part) if part else 0)
                              for part in whole.split("."))
    value = int(digits) if digits else 0
    # JS `Number.isSafeInteger`; past 2**53 the upstream reads the digits back.
    if value > 2 ** 53 - 1:
        return text
    return spell_number(value) if fraction is None else spell_decimal(value, fraction)


def normalize_text(text: str) -> str:
    """Rewrite text so a speech model sees only words and phrasing punctuation.

    Times and dates are deliberately not interpreted: `07.30` could be a time,
    a version or a price, and guessing wrong is worse than reading the digits.
    """
    result = text
    for pattern, replacement in _TYPOGRAPHY:
        result = pattern.sub(replacement, result)
    result = _RUPIAH.sub(lambda m: f"{_read_number(m.group(1))} rupiah", result)
    for pattern, replacement in _SPOKEN_SYMBOLS:
        result = pattern.sub(replacement, result)
    result = _NUMBER.sub(lambda m: _read_number(m.group(0)), result)
    result = "".join(char for char in result
                     if char.isalpha() or char.isspace() or char in _KEPT_PUNCTUATION)
    return _WHITESPACE.sub(" ", result)


# --------------------------------------------------------------------------
# collocations.ts
# --------------------------------------------------------------------------

COLLOCATION_WINDOW = 4

_collocations: dict[str, tuple[int, frozenset[str]]] | None = None


def collocation_rules() -> dict[str, tuple[int, frozenset[str]]]:
    """`word mask trigger...` per line: which reading a nearby word selects."""
    global _collocations
    if _collocations is not None:
        return _collocations
    parsed: dict[str, tuple[int, frozenset[str]]] = {}
    for line in _read_packed("collocations").split("\n"):
        if not line:
            continue
        fields = line.split(" ")
        if len(fields) < 3:
            raise IndoG2PError("corrupt_data", f"malformed collocation line: {line!r}")
        try:
            parsed[fields[0]] = (int(fields[1], 16), frozenset(fields[2:]))
        except ValueError as exc:
            raise IndoG2PError("corrupt_data",
                               f"malformed collocation mask: {line!r}") from exc
    _collocations = parsed
    return parsed


def resolve_collocations(words: list[str]) -> list[str | None]:
    """Resolve homographs from the words around them, with no model.

    A homograph keeps its dictionary reading unless one of its trigger words is
    within `COLLOCATION_WINDOW` real words. Spelled-out numbers do not count
    against the window, because one number is several words.
    """
    rules = collocation_rules()
    out: list[str | None] = []
    for index, word in enumerate(words):
        rule = rules.get(word)
        if rule is None:
            out.append(None)
            continue
        mask, triggers = rule
        resolved: str | None = None
        for step in (-1, 1):
            spent = 0
            near = index + step
            while 0 <= near < len(words):
                other = words[near]
                if other in triggers:
                    resolved = apply_mask(word, mask)
                    break
                if other not in NUMBER_WORDS:
                    spent += 1
                    if spent >= COLLOCATION_WINDOW:
                        break
                near += step
            if resolved is not None:
                break
        out.append(resolved)
    return out


# --------------------------------------------------------------------------
# english.ts -- optional, and by default not shipped
# --------------------------------------------------------------------------

_english: dict[str, str] | None = None


def english_available() -> bool:
    """Whether the optional English table was vendored into this install."""
    return (DATA_DIR / "english.xz").is_file()


def look_up_english(word: str) -> str | None:
    """An English word Indonesian spelling rules would mangle, or None.

    The table can only ever answer for words no Indonesian source places, so it
    cannot override `jakarta` or `april`. It is 652 KB in the wheel and its
    measured contribution is in the evidence file; when it is absent this
    returns None for everything and the Indonesian rules read the word, which
    is exactly what `indo-g2p/core` does.
    """
    global _english
    if _english is None:
        if not english_available():
            _english = {}
        else:
            table: dict[str, str] = {}
            for line in _read_packed("english").split("\n"):
                if not line:
                    continue
                space = line.rfind(" ")
                if space < 1:
                    raise IndoG2PError("corrupt_data", f"malformed english line: {line!r}")
                table[line[:space]] = line[space + 1:]
            _english = table
    return _english.get(word)


# --------------------------------------------------------------------------
# g2p.ts
# --------------------------------------------------------------------------

def _consonant_vowel_pattern(word: str) -> str:
    return "".join("V" if char in VOWELS else "K" for char in word)


def is_abbreviation(word: str) -> bool:
    """A word that spells no valid syllable is read letter by letter."""
    spelling = _consonant_vowel_pattern(word)
    return not any(pattern in spelling for pattern in SYLLABLE_PATTERNS)


def _spell_out(word: str) -> str:
    return "".join(LETTER_NAMES.get(char, char) for char in word)


def apply_glottal_stops(word: str) -> str:
    """Mark every `k` that sits between a vowel and a consonant as /ʔ/.

    Both patterns are matched against the original spelling, not against the
    partially rewritten one, which is what upstream does and what keeps
    `kk` sequences from cascading.
    """
    chars = list(word)
    for pattern in (GLOTTAL_STOP_PATTERN, GLOTTAL_BEFORE_CLITIC_PATTERN):
        for match in pattern.finditer(word):
            chars[match.start() + 1] = "ʔ"
    return "".join(chars)


def _apply_replacements(text: str, pairs: tuple[tuple[str, str], ...]) -> str:
    for source, target in pairs:
        text = text.replace(source, target)
    return text


def word_to_phonemes(word: str, expand_abbr: bool, resolved: str | None,
                     use_english: bool) -> tuple[str, bool]:
    """One already-lowercased word -> (phonemes, was read letter by letter)."""
    abbr = expand_abbr and is_abbreviation(word)

    # Consulted before the Indonesian rules, but the table holds only words
    # those rules have no answer for, so it can never override them.
    if use_english and not abbr and resolved is None:
        borrowed = look_up_english(word)
        if borrowed is not None:
            return borrowed, abbr

    result = _spell_out(word) if abbr else word
    if not abbr and resolved is not None:
        result = resolved
    elif "e" in result:
        result = apply_schwa(result)
    if result.endswith("k"):
        result = result[:-1] + "ʔ"
    result = apply_glottal_stops(result)
    result = NASAL_DIGRAPH_PATTERN.sub("ɲ", result)
    return _apply_replacements(result, PHONEME_REPLACEMENTS), abbr


def convert(text: str, *, normalize: bool = True, expand_abbr: bool = False,
            english: bool = True, resolve_schwa=resolve_collocations
            ) -> tuple[str, list[str]]:
    """Indonesian text -> (phonemes, syllables).

    `resolve_schwa` takes the sentence's words and returns one resolved
    spelling per word or None; pass None to turn resolution off entirely.
    """
    if not isinstance(text, str):
        raise IndoG2PError("type", f"text must be str, got {type(text)!r}")
    prepared = normalize_text(text) if normalize else text
    lowered = prepared.lower()
    matches = list(WORD_PATTERN.finditer(lowered))
    words = [match.group(0) for match in matches]
    resolutions: list[str | None] = ([None] * len(words) if resolve_schwa is None
                                     else resolve_schwa(words))
    if len(resolutions) != len(words):
        raise IndoG2PError("resolver",
                           f"resolver returned {len(resolutions)} values for {len(words)} words")

    syllables: list[str] = []
    parts: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        phonemes, abbr = word_to_phonemes(match.group(0), expand_abbr,
                                          resolutions[index], english)
        word_syllables = to_syllables(phonemes)
        # Diphthongs are only recognised inside a syllable, never across a break.
        if not abbr and any(pair in phonemes for pair, _ in DIPHTHONGS):
            word_syllables = [_apply_replacements(s, DIPHTHONGS) for s in word_syllables]
        parts.append(lowered[cursor:match.start()])
        parts.append("".join(word_syllables))
        cursor = match.end()
        syllables.extend(word_syllables)
        syllables.append(" ")
    parts.append(lowered[cursor:])
    return "".join(parts), syllables


def to_phoneme(text: str, **options) -> str:
    """The phoneme string alone, which is what a caller usually wants."""
    return convert(text, **options)[0]
