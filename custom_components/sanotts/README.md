# sanoTTS for Home Assistant

A text-to-speech engine for Home Assistant that runs on the Home Assistant
machine itself. No cloud service, no API key, and no separate add-on container.

It plugs into Assist like any other TTS engine, so you can pick it in a voice
pipeline or call `tts.speak` from an automation.

It **streams**. Home Assistant hands a conversation agent's reply over as the
agent writes it, and this speaks each sentence as it completes rather than
waiting for the whole reply. Measured on an M-series Mac with the voice already
loaded: first audio at 0.03 s streaming versus 0.11 s one-shot, for 4.85 s of
speech. The gap that matters in practice is larger than that, because one-shot
also waits for the language model to finish writing.

## Is this the right thing for you?

Home Assistant already has good local TTS: the **Piper** add-on, which speaks
over Wyoming and is the default for local voice pipelines. Piper sounds better
than these voices. Use Piper if it runs well for you.

This integration is worth having when one of these is true:

- **Your Home Assistant host is small.** A whole voice here is 2–3 MB and the
  runtime is numpy only, with no PyTorch, no onnxruntime and no add-on
  container. That fits places a Piper add-on does not, including a Home
  Assistant Core install on a low-end box.
- **You cannot run add-ons at all.** Home Assistant Core and Container installs
  have no add-on store; this is a plain custom integration, so it works there.
- **You want a second engine** for announcements while Piper handles the
  pipeline, or you want Vietnamese or Indonesian voices.

These voices are small models. They are clearly intelligible and clearly
synthetic. That is the trade.

## Install

**HACS** (custom repository): add `https://github.com/Ampixa/sanoTTS` as an
Integration, install **sanoTTS**, restart Home Assistant.

**Manually:** copy this directory to `config/custom_components/sanotts/` on the
Home Assistant host and restart.

Then go to **Settings → Devices & services → Add integration → sanoTTS** and
pick a voice.

Home Assistant installs the [`sanotts`](https://pypi.org/project/sanotts/)
package (numpy, phonemizer-fork, espeakng-loader) on first setup.

## Voices

| Voice | Language | Parameters |
| --- | --- | --- |
| `amy` | English (US) | 1.46 M |
| `amy-1p1m` | English (US) | 1.08 M |
| `amy-1p8m` | English (US) | 1.8 M |
| `hfc` | English (US) | 1.8 M |
| `kristin` | English (US) | 1.4 M |
| `vi` | Vietnamese | 1.46 M |
| `id` | Indonesian | 1.46 M |

The voice chosen during setup is the default. Any voice can be selected per
call, and Assist shows the ones matching the pipeline language.

## Staying fully offline

Named voices are downloaded once, during setup, from the project's GitHub
release into the Home Assistant cache. After that, nothing touches the network.

To avoid even that download, unpack a voice package onto the Home Assistant
host and give its absolute path in the **Local voice directory** field. When
that is set the directory is used and nothing is fetched.

## Using it

Pick **sanoTTS** as the text-to-speech engine in
**Settings → Voice assistants → your pipeline**, or call it directly:

```yaml
action: tts.speak
target:
  entity_id: tts.sanotts
data:
  media_player_entity_id: media_player.kitchen
  message: The garage door has been open for ten minutes.
```

Two options are supported per call:

```yaml
action: tts.speak
target:
  entity_id: tts.sanotts
data:
  media_player_entity_id: media_player.kitchen
  message: Slower, and in a different voice.
  options:
    voice: kristin
    length_scale: 1.2      # > 1 is slower, < 1 is faster
```

Leave `length_scale` out to use the pacing the voice ships with, which is not
always 1.0.

## Notes

- Synthesis is CPU-bound and runs in Home Assistant's executor, so it does not
  block the event loop. The first call for a voice also loads it from disk.
- Audio is returned as 16-bit mono WAV at the voice's own sample rate
  (22.05 kHz for the current voices).
- Loading and synthesis failures raise a `HomeAssistantError` with the voice
  name and the underlying cause, and are logged. Nothing fails silently.

## Related

The same models also run on an ESP32-S3 and in the browser. See the
[project README](../../README.md).
