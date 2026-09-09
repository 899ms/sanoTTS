"""Pre-phonemization text fixes for Mexican Spanish (espeak voice ``es-419``).

    from sanotts.mexican_g2p import normalize_mexican_g2p
    text = normalize_mexican_g2p(text, espeak_voice)

espeak-ng has no rules for Nahuatl-derived toponyms, so it reads their ``x``
as /ks/. Measured with piper's bundled espeak at ``es-419``:

    Oaxaca   -> ˌoaksˈaka     Oajaca   -> ˌoaxˈaka
    Tlaxcala -> tlakskˈala    Tlaskala -> tlaskˈala
    Texcoco  -> tekskˈoko     Tescoco  -> teskˈoko
    Mixteca  -> mikstˈeka     Misteka  -> mistˈeka
    Xalapa   -> salˈapa       Jalapa   -> xalˈapa

Respelling the input is the only lever we have, since the phonemizer is a
fixed dependency. The rewrite happens before G2P and never reaches the user's
eyes.

THE GATE IS NOT OPTIONAL. `espeak_voice` is required, and everything here is a
no-op unless it names Mexican Spanish. These rules are actively wrong
elsewhere: "$" means pesos only here, and stripping the thousands comma from
"12,500" turns twelve-point-five into twelve thousand five hundred in every
comma-decimal locale, which is German, French, Russian, Turkish and Spanish
itself. An earlier revision applied them to every language, which corrupted
the training text of all twenty-odd non-Spanish voices.

Original toponym and number analysis by @zkartamx (Ampixa/sanoTTS#6).
"""

from __future__ import annotations

import re

# espeak's Latin-American Spanish voice, which is what an es_MX Piper teacher
# declares. Peninsular "es" is deliberately excluded: the toponym fixes suit it,
# but the peso and phone-number rules do not.
_MEXICAN_VOICES = ("es-419", "es_mx", "es-mx", "es_419")

# Nahuatl-derived spellings espeak mis-reads, and the respelling that fixes it.
# Applied case-sensitively in both cases so sentence-initial forms survive.
_TOPONYMS = (
    ("Oaxaca", "Oajaca"), ("Oaxaqueñ", "Oajaqueñ"),
    ("Xalapa", "Jalapa"), ("Xalapeñ", "Jalapeñ"),
    ("Tlaxcala", "Tlaskala"),
    ("Mixteca", "Misteka"),
    ("Texcoco", "Tescoco"),
)

_UNITS = (
    (r"\bkm\s?²", "kilómetros cuadrados"), (r"\bkm\s?³", "kilómetros cúbicos"),
    (r"\bcm\s?²", "centímetros cuadrados"), (r"\bcm\s?³", "centímetros cúbicos"),
    (r"\bm\s?²", "metros cuadrados"), (r"\bm\s?³", "metros cúbicos"),
)

# Digits and thousands commas, then optionally exactly two decimal places, and
# not butted against another digit. Anchoring the cents to \d{2} is what stops
# the match swallowing a sentence-final period: "$20. Es barato" keeps its
# full stop, and espeak still gets a clause boundary to break on.
_CURRENCY = re.compile(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?(?!\d)")
_THOUSANDS = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_NEGATIVE = re.compile(r"(?<![\w-])-(\d)")
_DIGIT_GROUPS = re.compile(r"\b\d{1,4}(?: \d{1,4}){2,}\b")


def applies_to(espeak_voice: str | None) -> bool:
    """True only for the Mexican/Latin-American Spanish espeak voice."""
    if not espeak_voice:
        return False
    return str(espeak_voice).strip().lower() in _MEXICAN_VOICES


def _currency(match: re.Match) -> str:
    pesos = match.group(1).replace(",", "")
    centavos = match.group(2)
    if centavos and int(centavos):
        return f"{pesos} pesos con {int(centavos)} centavos"
    return f"{pesos} pesos"


def _thousands(match: re.Match) -> str:
    """Strip thousands commas; collapse to "N millones" only when exact.

    The earlier revision returned ``n // 1_000_000`` for anything past a
    billion, so 1,500,750,000 was read as "1500 millones" and the remaining
    750,000 vanished silently. Only collapse when nothing is lost.
    """
    digits = match.group(0).replace(",", "")
    value = int(digits)
    if value >= 1_000_000_000 and value % 1_000_000 == 0:
        return f"{value // 1_000_000} millones"
    return digits


def normalize_mexican_g2p(text: str, espeak_voice: str) -> str:
    """Rewrite `text` for espeak's es-419; a no-op for every other voice."""
    if not applies_to(espeak_voice):
        return text

    for source, replacement in _TOPONYMS:
        text = re.sub(rf"\b{source}", replacement, text)
        text = re.sub(rf"\b{source.lower()}", replacement.lower(), text)

    text = _NEGATIVE.sub(r"menos \1", text)
    text = _CURRENCY.sub(_currency, text)
    text = _DIGIT_GROUPS.sub(lambda m: " ".join(re.sub(r"\s+", "", m.group(0))), text)
    for pattern, replacement in _UNITS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = _THOUSANDS.sub(_thousands, text)
    return text
