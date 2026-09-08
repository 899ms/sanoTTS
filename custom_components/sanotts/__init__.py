"""The sanoTTS text-to-speech integration.

Synthesis runs entirely on the Home Assistant host through the `sanotts` PyPI
package (numpy only, no torch and no onnxruntime). Nothing is sent anywhere:
the only network access this integration ever performs is the one-time voice
download in the config flow, and even that is skipped when a local voice
directory is configured.
"""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

PLATFORMS: list[Platform] = [Platform.TTS]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up sanoTTS from a config entry."""
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
