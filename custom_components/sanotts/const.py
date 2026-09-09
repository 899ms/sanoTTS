"""Constants for the sanoTTS integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "sanotts"

# Config-entry keys.
CONF_VOICE: Final = "voice"
CONF_VOICE_DIR: Final = "voice_dir"

# Per-call TTS option key. There is no core constant for speaking rate, so the
# name is ours; the voice option uses homeassistant.components.tts.ATTR_VOICE.
OPT_LENGTH_SCALE: Final = "length_scale"


class VoiceInfo:
    """One selectable voice: its sanotts alias, a label, and its HA language."""

    __slots__ = ("alias", "label", "language")

    def __init__(self, alias: str, label: str, language: str) -> None:
        """Store the alias the sanotts package knows and how to present it."""
        self.alias = alias
        self.label = label
        self.language = language


# Mirrors pypkg/sanotts/tables/voices.json in this repository, converted to the
# hyphenated language tags Home Assistant uses (the package writes en_US).
# tools/check_sanotts_ha_voices.py asserts the two stay in step, so a voice
# added to the package but not here is a test failure rather than a surprise.
VOICES: Final[tuple[VoiceInfo, ...]] = (
    VoiceInfo("amy", "Amy — English (1.46M)", "en-US"),
    VoiceInfo("amy-1p1m", "Amy small — English (1.08M)", "en-US"),
    VoiceInfo("amy-1p8m", "Amy large — English (1.8M)", "en-US"),
    VoiceInfo("hfc", "HFC — English (1.8M)", "en-US"),
    VoiceInfo("kristin", "Kristin — English (1.4M)", "en-US"),
    VoiceInfo("vi", "Vietnamese (1.46M)", "vi-VN"),
    VoiceInfo("id", "Indonesian (1.46M)", "id-ID"),
)

VOICES_BY_ALIAS: Final[dict[str, VoiceInfo]] = {v.alias: v for v in VOICES}

# Deterministic order, so the entity's language list is stable across restarts.
SUPPORTED_LANGUAGES: Final[list[str]] = sorted({v.language for v in VOICES})

DEFAULT_VOICE: Final = "amy"
