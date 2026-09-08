"""Shared Mexican-Spanish pre-phonemization normalizer for espeak-ng.

Used by BOTH the serve dashboard and the pack builder so the model learns and
pronounces these cases the same way. espeak-ng (the G2P that both the Piper
teacher and the students use) lacks Mexican indigenous toponym/number rules, so
we rewrite the text before phonemization:

Toponyms (Nahuatl 'x'):
    Oaxaca->Oajaca  Xalapa->Jalapa  Tlaxcala->Tlaskala  Mixteca->Misteka
    Huastec->Guastec  Texcoco->Tescoco
Numbers (es-419 reads ',' as decimal, '.' as thousands):
    strip comma-thousands ('12,500'->'12500'), billion scale -> 'X millones',
    leading '-' -> 'menos', '$X.YY' -> 'X pesos con YY centavos',
    phone-like space groups -> digit-by-digit, 'm²'->'metros cuadrados'.
"""

from __future__ import annotations

import re


def _replace_number_group(match: re.Match) -> str:
    """'12,500' -> '12500'; billion scale -> '<value/1e6> millones'."""
    digits = match.group(0).replace(",", "")
    n = int(digits)
    if n >= 1_000_000_000:
        return f"{n // 1_000_000} millones"
    return digits


def _currency(match: re.Match) -> str:
    """'$1,850.75' -> '1850 pesos con 75 centavos'; '$12,500' -> '12500 pesos'."""
    raw = match.group(1)
    digits = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group(0).replace(",", ""), raw)
    if "." in digits:
        whole, frac = digits.split(".", 1)
        centavos = re.sub(r"\D", "", frac)[:2]
        if centavos:
            return f"{whole} pesos con {centavos} centavos"
        return f"{whole} pesos"
    return f"{digits} pesos"


def _phone(match: re.Match) -> str:
    """'55 5489 2726' -> '5 5 5 4 8 9 2 7 2 6' (read digit-by-digit)."""
    return " ".join(re.sub(r"\s+", "", match.group(0)))


def normalize_mexican_g2p(text: str) -> str:
    """Pre-phonemization normalization for Mexican Spanish with espeak-ng."""
    text = re.sub(r"\bOaxaca\b", "Oajaca", text)
    text = re.sub(r"\boaxaca\b", "oajaca", text)
    text = re.sub(r"\bOaxaqueñ", "Oajaqueñ", text)
    text = re.sub(r"\boaxaqueñ", "oajaqueñ", text)
    text = re.sub(r"\bXalapa\b", "Jalapa", text)
    text = re.sub(r"\bxalapa\b", "jalapa", text)
    text = re.sub(r"\bXalapeñ", "Jalapeñ", text)
    text = re.sub(r"\bxalapeñ", "jalapeñ", text)
    text = re.sub(r"\bTlaxcala\b", "Tlaskala", text)
    text = re.sub(r"\btlaxcala\b", "tlaskala", text)
    text = re.sub(r"\bMixteca\b", "Misteka", text)
    text = re.sub(r"\bmixteca\b", "misteka", text)
    text = re.sub(r"\bHuastec", "Guastec", text)
    text = re.sub(r"\bhuastec", "guastec", text)
    text = re.sub(r"\bTexcoco\b", "Tescoco", text)
    text = re.sub(r"\btexcoco\b", "tescoco", text)
    # negative numbers
    text = re.sub(r"(?<![\w-])-(\d)", r"menos \1", text)
    # currency
    text = re.sub(r"\$(\d[\d.,]*)", _currency, text)
    # phone-like space-separated digit groups -> digit-by-digit
    text = re.sub(r"\b\d{1,4}(?: \d{1,4}){2,}\b", _phone, text)
    # units (espeak reads the superscript "²"/"³" as 'dos'/'tres')
    text = re.sub(r"\bkm\s?²\b", "kilómetros cuadrados", text, flags=re.I)
    text = re.sub(r"\bkm\s?³\b", "kilómetros cúbicos", text, flags=re.I)
    text = re.sub(r"\bcm\s?²\b", "centímetros cuadrados", text, flags=re.I)
    text = re.sub(r"\bcm\s?³\b", "centímetros cúbicos", text, flags=re.I)
    text = re.sub(r"\bm\s?²\b", "metros cuadrados", text, flags=re.I)
    text = re.sub(r"\bm\s?³\b", "metros cúbicos", text, flags=re.I)
    # comma-thousands + billions
    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", _replace_number_group, text)
    return text
