"""Text-to-speech entity backed by the local sanotts package."""

from __future__ import annotations

import io
import logging
import re
import wave
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from homeassistant.components import tts
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_VOICE,
    CONF_VOICE_DIR,
    DEFAULT_VOICE,
    DOMAIN,
    OPT_LENGTH_SCALE,
    SUPPORTED_LANGUAGES,
    VOICES,
    VOICES_BY_ALIAS,
)

_LOGGER = logging.getLogger(__name__)

# Sentence boundaries for streaming: Latin and CJK terminators, taking any
# trailing quote or bracket with them so >>He said "go."<< stays one piece.
_SENTENCE_RE = re.compile("[^.!?\u2026\u3002\uff01\uff1f\n]*"
                          "[.!?\u2026\u3002\uff01\uff1f\n]+"
                          "[\"'\u201d\u2019)\\]]*\\s*")

# A caller can stream a long run with no terminator at all -- an LLM mid-thought
# is the usual case. Past this many pending characters we cut at a word boundary
# rather than let the audio stall waiting for a full stop that never comes. With
# no word boundary in the window either we keep buffering, because cutting inside
# a word would mispronounce it; that degrades to the non-streaming behaviour,
# which is the right way to fail.
_MAX_PENDING_CHARS = 240


def _take_sentences(buffer: str) -> tuple[list[str], str]:
    """Split off every complete sentence, returning them and the remainder."""
    sentences: list[str] = []
    end = 0
    for match in _SENTENCE_RE.finditer(buffer):
        sentences.append(match.group(0))
        end = match.end()
    remainder = buffer[end:]
    if len(remainder) > _MAX_PENDING_CHARS:
        cut = remainder.rfind(" ", 0, _MAX_PENDING_CHARS)
        if cut > 0:
            sentences.append(remainder[:cut])
            remainder = remainder[cut:]
    return sentences, remainder


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sanoTTS entity from a config entry."""
    async_add_entities([SanoTTSEntity(config_entry)])


def _pcm_bytes(audio: np.ndarray) -> bytes:
    """Convert a float32 mono waveform in [-1, 1] to 16-bit little-endian PCM.

    The package already clips to [-1, 1]; the clip here is belt and braces so a
    future change upstream cannot turn into wraparound crackle on the speaker.
    """
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _wav_header(sample_rate: int) -> bytes:
    """A 16-bit mono WAV header declaring zero frames.

    Zero frames is how Home Assistant's own streaming TTS engines signal that
    the length is not known ahead of time (see the wyoming integration); raw
    PCM chunks follow it.
    """
    with io.BytesIO() as buffer:
        wav_file: wave.Wave_write = wave.open(buffer, "wb")
        with wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
        buffer.seek(0)
        return buffer.getvalue()


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Pack a float32 mono waveform as a complete 16-bit PCM WAV file."""
    return _wav_header(sample_rate) + _pcm_bytes(audio)


class SanoTTSEntity(tts.TextToSpeechEntity):
    """Speak text using a sanoTTS voice, entirely on this machine."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialise the entity without loading any voice yet."""
        self._config_entry = config_entry
        self._default_voice: str = config_entry.data.get(CONF_VOICE, DEFAULT_VOICE)
        self._voice_dir: str | None = config_entry.data.get(CONF_VOICE_DIR)
        # One loaded Synthesizer per voice alias. Loading a pack reads several
        # MB off disk and builds numpy arrays, so it is done once and reused.
        self._synthesizers: dict[str, Any] = {}

        self._attr_unique_id = config_entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, config_entry.entry_id)},
            entry_type=DeviceEntryType.SERVICE,
            manufacturer="Ampixa",
            model="sanoTTS",
            name=config_entry.title,
        )

        default = VOICES_BY_ALIAS.get(self._default_voice)
        self._attr_default_language = default.language if default else "en-US"
        self._attr_supported_languages = SUPPORTED_LANGUAGES
        self._attr_supported_options = [tts.ATTR_VOICE, OPT_LENGTH_SCALE]

    def async_get_supported_voices(self, language: str) -> list[tts.Voice] | None:
        """List the voices available for a language."""
        voices = [
            tts.Voice(voice_id=v.alias, name=v.label)
            for v in VOICES
            if v.language == language
        ]
        return voices or None

    def _get_synthesizer(self, voice: str) -> Any:
        """Return a loaded Synthesizer for `voice`, loading it on first use."""
        if (synth := self._synthesizers.get(voice)) is not None:
            return synth

        # Imported here rather than at module scope so that a broken or missing
        # requirement is reported when someone actually asks for speech, with
        # the voice name in the message, instead of failing platform setup.
        try:
            from sanotts import Synthesizer  # noqa: PLC0415
        except ImportError as err:
            raise HomeAssistantError(
                "The 'sanotts' package is not installed. Reload the sanoTTS "
                "integration so Home Assistant can install its requirements."
            ) from err

        _LOGGER.debug("Loading sanoTTS voice %s", voice)
        try:
            # A configured local directory always wins: it is the fully offline
            # path, and the alias would otherwise trigger a download.
            synth = Synthesizer(
                None if self._voice_dir else voice, voice_dir=self._voice_dir
            )
        except Exception as err:
            raise HomeAssistantError(
                f"Could not load sanoTTS voice {voice!r}: {err}"
            ) from err

        self._synthesizers[voice] = synth
        return synth

    def _resolve_voice(self, language: str, options: dict[str, Any]) -> str:
        """Choose the voice: an explicit option, else one matching the language."""
        requested = options.get(tts.ATTR_VOICE)
        if requested:
            if requested not in VOICES_BY_ALIAS:
                raise HomeAssistantError(
                    f"Unknown sanoTTS voice {requested!r}. Available voices: "
                    + ", ".join(sorted(VOICES_BY_ALIAS))
                )
            return str(requested)

        # No explicit voice: keep the configured default when it speaks the
        # requested language, otherwise fall back to any voice that does.
        default = VOICES_BY_ALIAS.get(self._default_voice)
        if default is not None and default.language == language:
            return default.alias

        for voice in VOICES:
            if voice.language == language:
                return voice.alias

        raise HomeAssistantError(
            f"No sanoTTS voice supports language {language!r}. Supported: "
            + ", ".join(SUPPORTED_LANGUAGES)
        )

    @staticmethod
    def _resolve_length_scale(options: dict[str, Any]) -> float | None:
        """Validate the optional speaking-rate override.

        None means "use whatever the voice package ships with", which is not
        always 1.0 — so an unset option must not be turned into one.
        """
        if (raw := options.get(OPT_LENGTH_SCALE)) is None:
            return None
        try:
            scale = float(raw)
        except (TypeError, ValueError) as err:
            raise HomeAssistantError(
                f"{OPT_LENGTH_SCALE} must be a number, got {raw!r}"
            ) from err
        if scale <= 0.0:
            raise HomeAssistantError(
                f"{OPT_LENGTH_SCALE} must be greater than 0, got {scale}"
            )
        return scale

    def _synthesize(
        self, voice: str, message: str, length_scale: float | None
    ) -> tuple[np.ndarray, int]:
        """Load the voice if needed and synthesize. Blocking; run in executor."""
        synth = self._get_synthesizer(voice)
        try:
            result = synth.synthesize(message, duration_length_scale=length_scale)
        except Exception as err:
            # Includes FrontendError, which is what an unphonemizable string
            # raises. Never return (None, None) here: that shows up in the UI
            # as a silent failure with nothing in the log to act on.
            raise HomeAssistantError(
                f"sanoTTS failed to synthesize with voice {voice!r}: {err}"
            ) from err
        return result.audio, result.sample_rate

    def get_tts_audio(
        self, message: str, language: str, options: dict[str, Any] | None = None
    ) -> tts.TtsAudioType:
        """Synthesize `message` in one shot. Runs in an executor via the base class."""
        opts = options or {}
        voice = self._resolve_voice(language, opts)
        length_scale = self._resolve_length_scale(opts)
        audio, sample_rate = self._synthesize(voice, message, length_scale)
        return "wav", _to_wav(audio, sample_rate)

    async def async_stream_tts_audio(
        self, request: tts.TTSAudioRequest
    ) -> tts.TTSAudioResponse:
        """Synthesize sentence by sentence as the text arrives.

        Overriding this is what makes async_supports_streaming_input() true, so
        Home Assistant feeds us an LLM's output as it is generated instead of
        waiting for the whole reply. The listener hears sentence one while
        sentence two is still being written.

        Synthesis is CPU-bound numpy, so every call goes to the executor; the
        event loop must never block on it.
        """
        voice = self._resolve_voice(request.language, request.options)
        length_scale = self._resolve_length_scale(request.options)

        async def data_gen() -> AsyncGenerator[bytes]:
            header_sent = False
            pending = ""
            spoke = False

            async for text_chunk in request.message_gen:
                pending += text_chunk
                sentences, pending = _take_sentences(pending)
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    audio, sample_rate = await self.hass.async_add_executor_job(
                        self._synthesize, voice, sentence, length_scale
                    )
                    if not header_sent:
                        # The rate comes from the voice pack, so the header can
                        # only be written once something has been synthesized.
                        yield _wav_header(sample_rate)
                        header_sent = True
                    yield _pcm_bytes(audio)
                    spoke = True

            if pending.strip():
                audio, sample_rate = await self.hass.async_add_executor_job(
                    self._synthesize, voice, pending, length_scale
                )
                if not header_sent:
                    yield _wav_header(sample_rate)
                    header_sent = True
                yield _pcm_bytes(audio)
                spoke = True

            if not spoke:
                raise HomeAssistantError(
                    "sanoTTS was asked to speak text with nothing speakable in it"
                )

        return tts.TTSAudioResponse("wav", data_gen())
