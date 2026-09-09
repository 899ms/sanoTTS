"""Config flow for the sanoTTS integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.selector import (
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)

from .const import CONF_VOICE, CONF_VOICE_DIR, DEFAULT_VOICE, DOMAIN, VOICES

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_VOICE, default=DEFAULT_VOICE): SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=v.alias, label=v.label) for v in VOICES
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        ),
        vol.Optional(CONF_VOICE_DIR): str,
    }
)


def _load_voice(voice: str, voice_dir: str | None) -> None:
    """Load the voice once, to fail here rather than at the first spoken word.

    Runs in an executor: for a named voice this downloads roughly 2-3 MB into
    ~/.cache/sanotts on first use, which also warms the cache for the entity.
    """
    # Imported lazily so a missing requirement surfaces as a flow error rather
    # than breaking integration discovery at startup.
    from sanotts import Synthesizer  # noqa: PLC0415

    Synthesizer(None if voice_dir else voice, voice_dir=voice_dir)


class SanoTTSConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for sanoTTS."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Pick a voice and verify it actually loads."""
        errors: dict[str, str] = {}

        if user_input is not None:
            voice: str = user_input[CONF_VOICE]
            voice_dir: str | None = (user_input.get(CONF_VOICE_DIR) or "").strip() or None

            self._async_abort_entries_match(
                {CONF_VOICE: voice, CONF_VOICE_DIR: voice_dir}
            )

            if voice_dir is not None and not Path(voice_dir).is_dir():
                errors[CONF_VOICE_DIR] = "voice_dir_not_found"
            else:
                try:
                    await self.hass.async_add_executor_job(
                        _load_voice, voice, voice_dir
                    )
                except ImportError:
                    _LOGGER.exception("The sanotts package failed to import")
                    errors["base"] = "package_missing"
                except OSError:
                    # urllib raises URLError (an OSError) when offline, and the
                    # package raises VoicePackError for a bad directory; both
                    # are actionable by the user, so neither is swallowed.
                    _LOGGER.exception("Could not reach the voice download host")
                    errors["base"] = "cannot_download"
                except Exception:  # noqa: BLE001 - surfaced to the user below
                    _LOGGER.exception("Could not load sanoTTS voice %s", voice)
                    errors["base"] = "cannot_load_voice"

            if not errors:
                data: dict[str, Any] = {CONF_VOICE: voice}
                if voice_dir is not None:
                    data[CONF_VOICE_DIR] = voice_dir
                return self.async_create_entry(title=f"sanoTTS ({voice})", data=data)

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )
