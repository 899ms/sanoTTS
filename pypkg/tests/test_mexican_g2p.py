"""The es_MX normalizer must fix Mexican Spanish and touch nothing else.

The bug this guards against shipped in the first revision of Ampixa/sanoTTS#6:
the rewrite was applied with no language gate, so every non-Spanish voice had
its text mangled -- German "12,500 Kilometer" (12.5 km) became "12500", and
English "$20" became "20 pesos".
"""

import pytest

from sanotts.mexican_g2p import applies_to, normalize_mexican_g2p

OTHER_VOICES = ["en-us", "de", "fr-fr", "ru", "tr", "es", "pt-br", "it"]

UNTOUCHED = [
    "Die Strecke ist 12,500 Kilometer lang.",     # comma is a DECIMAL point here
    "Le taux est de 3,750 pour cent.",
    "Температура 12,500 градуса.",
    "The laptop costs $1,299.99 and weighs -5 kg.",
    "Alan 240 m² ve sicaklik -5 derece.",
    "Call 555 4489 2726 tomorrow.",
]


@pytest.mark.parametrize("text", UNTOUCHED)
@pytest.mark.parametrize("voice", OTHER_VOICES)
def test_other_languages_pass_through(text, voice):
    assert normalize_mexican_g2p(text, voice) == text


def test_gate():
    assert applies_to("es-419")
    assert applies_to("es_MX")
    assert not applies_to("es")        # peninsular: peso/phone rules do not apply
    assert not applies_to("en-us")
    assert not applies_to("")
    assert not applies_to(None)


@pytest.mark.parametrize("source,expected", [
    ("Oaxaca", "Oajaca"),
    ("Tlaxcala", "Tlaskala"),
    ("Texcoco", "Tescoco"),
    ("Mixteca", "Misteka"),
    ("Xalapa", "Jalapa"),
    ("oaxaca", "oajaca"),
])
def test_toponyms(source, expected):
    assert normalize_mexican_g2p(source, "es-419") == expected


@pytest.mark.parametrize("source,expected", [
    # The sentence-final period must survive: espeak clause-splits on it.
    ("Cuesta $20. Es barato.", "Cuesta 20 pesos. Es barato."),
    ("Son $5.", "Son 5 pesos."),
    ("Cuesta $1,850.75.", "Cuesta 1850 pesos con 75 centavos."),
    ("Cuesta $1,850.75 en total.", "Cuesta 1850 pesos con 75 centavos en total."),
])
def test_currency_keeps_punctuation(source, expected):
    assert normalize_mexican_g2p(source, "es-419") == expected


def test_billions_do_not_silently_truncate():
    assert normalize_mexican_g2p("1,500,000,000", "es-419") == "1500 millones"
    # 750,000 must not vanish, which is what n // 1_000_000 used to do
    assert normalize_mexican_g2p("1,500,750,000", "es-419") == "1500750000"
