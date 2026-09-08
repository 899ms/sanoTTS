"""Text -> phoneme ids for the nano voices.

Ported from `web/trellis_frontend.js`, which is what the browser demo runs,
so the Python and JavaScript paths agree symbol for symbol. That file is
itself a transcription of misaki's `EspeakFallback`, and its comments record
which upstream behaviour each step reproduces.

The pipeline is:

    text -> espeak-ng IPA -> misaki E2M normalisation -> 62-symbol ids

espeak-ng comes from the same `phonemizer-fork` + `espeakng-loader` pair the
piperlite voices already depend on, so text input needs no extra install. The
espeak *backend* differs though: this one runs with ties enabled, because the
E2M table below matches on tie characters ("a^ɪ", "d^ʒ"), whereas the
piperlite frontend deliberately turns them off. Same library, different
configuration, so the two do not share a backend instance.

Callers that already have phoneme ids can skip all of this and use
`Synthesizer.synthesize_ids()`.
"""

from __future__ import annotations

import re
from typing import Any

# The frozen 62-symbol vocabulary from the Trellis-RIFT package. The browser
# keeps its own copy for the same reason: so nothing has to fetch a checkpoint
# just to tokenise. A package that ships a vocabulary overrides this, and
# kokoro_frontend.validate_frontend() checks the two agree.
DEFAULT_VOCABULARY: dict[str, int] = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2,
    " ": 3, "!": 4, '"': 5, "(": 6, ")": 7, ",": 8, ".": 9, ":": 10, ";": 11,
    "?": 12, "A": 13, "I": 14, "O": 15, "T": 16, "W": 17, "Y": 18, "b": 19,
    "d": 20, "f": 21, "h": 22, "i": 23, "j": 24, "k": 25, "l": 26, "m": 27,
    "n": 28, "p": 29, "s": 30, "t": 31, "u": 32, "v": 33, "w": 34, "z": 35,
    "æ": 36, "ð": 37, "ŋ": 38, "ɐ": 39, "ɑ": 40,
    "ɔ": 41, "ə": 42, "ɛ": 43, "ɜ": 44, "ɡ": 45,
    "ɪ": 46, "ɹ": 47, "ʃ": 48, "ʊ": 49, "ʌ": 50,
    "ʒ": 51, "ʤ": 52, "ʧ": 53, "ˈ": 54, "ˌ": 55,
    "θ": 56, "ᵊ": 57, "ᵻ": 58, "—": 59, "“": 60,
    "”": 61,
}

SPECIAL_IDS = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
DEFAULT_MAX_TOKENS = 207          # configs.front.max_tokens

SYLLABIC = "̩"               # COMBINING VERTICAL LINE BELOW, chr(809)
NASAL = "̃"                  # COMBINING TILDE
TIE_DEFAULT = "͡"            # espeak's tie
TIE_MISAKI = "^"

# misaki/espeak.py EspeakFallback.E2M, in the order python produces it
# (sorted by -len(key), stable over insertion order). Order matters: "e^ɪ"
# must be tried before the bare "e".
E2M: list[tuple[str, str]] = [
    ("ʔˌn" + SYLLABIC, "ʔn"),
    ("ʔn" + SYLLABIC, "ʔn"),
    ("a^ɪ", "I"), ("a^ʊ", "W"), ("d^ʒ", "ʤ"), ("e^ɪ", "A"),
    ("t^ʃ", "ʧ"), ("ɔ^ɪ", "Y"), ("ə^l", "ᵊl"),
    ("ʲo", "jo"), ("ʲə", "jə"), ("e", "A"), ("ʲ", ""),
    ("ɚ", "əɹ"), ("r", "ɹ"), ("x", "k"), ("ç", "k"),
    ("ɐ", "ə"), ("ɬ", "l"), (NASAL, ""),
]


class FrontendError(RuntimeError):
    """`kind` separates an expected rejection (empty / too_long) from a bug."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def postprocess_line(line: str) -> str:
    """phonemizer's line post-processing, with the flags this frontend uses."""
    line = " ".join(line.strip().split("\n")).replace("  ", " ")
    line = re.sub(r"_+", "_", line).replace("_ ", " ")
    if not line:
        return ""
    # _process_stress is the identity at with_stress=True, and strip=False
    # keeps the trailing separator; only the tie is rewritten.
    return "".join(word.strip().replace(TIE_DEFAULT, TIE_MISAKI) + " "
                   for word in line.split(" "))


def apply_e2m(ps: str) -> str:
    """misaki EspeakFallback.__call__ tail, british=False, version=None."""
    ps = ps.strip()
    for needle, replacement in E2M:
        ps = ps.replace(needle, replacement)
    ps = re.sub(r"(\S)" + SYLLABIC, "ᵊ\\1", ps).replace(SYLLABIC, "")
    ps = ps.replace("o^ʊ", "O")
    ps = ps.replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ")
    ps = ps.replace("ɪə", "iə").replace("ː", "")
    ps = ps.replace("o", "ɔ")     # espeak < 1.52
    ps = ps.replace("ɾ", "T")     # version != '2.0'
    ps = ps.replace("ʔ", "t")
    return ps.replace("^", "")


def phonemes_to_token_ids(phonemes: str, vocabulary: dict[str, int] | None = None,
                          max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[list[int], str]:
    """IPA string -> (ids, dropped). Unknown symbols are dropped, as Kokoro does."""
    vocab = vocabulary or DEFAULT_VOCABULARY
    if max_tokens < 2:
        raise FrontendError("config", "max_tokens must allow BOS and EOS")
    kept: list[str] = []
    dropped: list[str] = []
    for ch in phonemes:
        if ch in vocab:
            if ch not in SPECIAL_IDS:
                kept.append(ch)
        else:
            dropped.append(ch)
    if not kept:
        raise FrontendError("empty", "phonemization produced no symbols in the packaged vocabulary")
    ids = [SPECIAL_IDS["<bos>"], *(int(vocab[c]) for c in kept), SPECIAL_IDS["<eos>"]]
    if len(ids) > max_tokens:
        raise FrontendError(
            "too_long",
            f"phoneme sequence has {len(ids)} tokens including BOS/EOS; maximum is {max_tokens}",
        )
    return ids, "".join(dropped)


_BACKENDS: dict[str, Any] = {}


def _espeak_ipa(text: str, voice: str = "en-us") -> str:
    """espeak-ng IPA with ties, post-processed as trellis_frontend.js does."""
    backend = _BACKENDS.get(voice)
    if backend is None:
        try:
            from phonemizer.backend import EspeakBackend  # noqa: PLC0415
        except ImportError as exc:
            raise FrontendError(
                "missing_dependency",
                "text input needs phonemizer-fork and espeakng-loader, which are "
                "declared dependencies of this package; reinstall with "
                "`pip install --force-reinstall sanotts`. Callers that already have "
                "phoneme ids can use Synthesizer.synthesize_ids() instead.",
            ) from exc
        # Reuse the piperlite frontend's espeak setup: it locates the shared
        # library and works around espeakng-loader wheels that report a data
        # path from the machine they were BUILT on. Rediscovering that here
        # would mean maintaining the same workaround twice.
        from .frontend import _ENGINE, PUNCTUATION_MARKS  # noqa: PLC0415

        _ENGINE._configure_once()
        try:
            backend = EspeakBackend(
                voice,
                preserve_punctuation=True,
                punctuation_marks=PUNCTUATION_MARKS,
                with_stress=True,
                tie=True,               # E2M matches on tie characters
                language_switch="remove-flags",
            )
        except Exception as exc:  # noqa: BLE001 - surfaced with context below
            raise FrontendError(
                "espeak", f"could not start espeak-ng for voice {voice!r}: {exc}"
            ) from exc
        _BACKENDS[voice] = backend
    out = backend.phonemize([text], strip=False, separator=None)
    if not out or not out[0]:
        raise FrontendError("empty", f"espeak-ng produced no phonemes for text: {text!r}")
    return postprocess_line(out[0])


def phonemize(text: str, *, vocabulary: dict[str, int] | None = None,
              voice: str = "en-us",
              max_tokens: int = DEFAULT_MAX_TOKENS) -> tuple[list[int], str]:
    """Text -> (phoneme ids, dropped symbols) for a nano voice."""
    if not isinstance(text, str) or not text.strip():
        raise FrontendError("empty", "text must be a non-empty string")
    return phonemes_to_token_ids(apply_e2m(_espeak_ipa(text, voice)), vocabulary, max_tokens)


def vocabulary_for(meta: dict[str, Any]) -> dict[str, int]:
    """A package's own vocabulary when it ships one, else the frozen default."""
    frontend = meta.get("frontend")
    if isinstance(frontend, dict) and isinstance(frontend.get("vocabulary"), dict):
        from .kokoro_frontend import validate_frontend  # noqa: PLC0415
        return validate_frontend(frontend, expected_vocab_size=int(meta.get("vocab_size", 0)) or None)
    expected = int(meta.get("vocab_size", len(DEFAULT_VOCABULARY)))
    if expected != len(DEFAULT_VOCABULARY):
        raise FrontendError(
            "frontend",
            f"package declares vocab_size {expected} but ships no vocabulary, and the "
            f"built-in table has {len(DEFAULT_VOCABULARY)} symbols",
        )
    return dict(DEFAULT_VOCABULARY)
