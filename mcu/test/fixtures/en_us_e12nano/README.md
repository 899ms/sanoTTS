# en_US E12-nano Golden Fixture

The 294,642-parameter nano stack as it ships: duration student (22,858) +
acoustic student (128,102) + TinyVocos decoder (143,682), mel-100 interface,
24 kHz, Kokoro `af_heart` teacher lineage, decoder checkpoint
`e12-nano-decoder/checkpoint_step225000.pt`.

Produced in one shot by `tools/export_e12_nano_q8.py`; consumed by
`mcu/test/nano_golden_main.c` and audited by
`tools/audit_e12_nano_parameters.py`. Regenerate or replace the complete
fixture as one versioned contract; never edit an individual binary in place.

## Contents

| file | what |
| --- | --- |
| `front_q8.bin` | duration + acoustic students, int8 + f32 |
| `model_q8.bin` | decoder, int8 + f32 |
| `rows.txt` | one line per row: `row_id tokens frames samples noise_seed` |
| `rNN_ids.bin` | int32 phoneme ids |
| `rNN_durs.bin` | int32 frame counts, frozen from the FLOAT duration student |
| `rNN_audio.bin` | float32 reference waveform from the float PyTorch stack |
| `e2e_uniform.bin` | first 64 `torch.rand` values for row 0's seed |
| `e2e_noise.bin` | float32 `[4, T]` decoder noise for row 0 |

The 8 rows are `packs-regen/eval8`, the same rows every quantisation
measurement on this stack uses.

## Why durations are frozen

The gate is waveform correlation. A one-frame drift in any token would shift
every later sample and make the number meaningless, so the fixture carries the
float duration student's own output and the runtime is told to use it
(`snt_nano_config.dur_override`). The int8 duration student runs anyway; only
its output is discarded. This is exactly what `en_us_r7` does.

## Why the noise is part of the contract

The decoder is noise-fed: four Gaussian channels enter through a learned
adapter. `render_fullstack_tiny.py` seeds them with
`sha256(row_id).digest()[:8]` read big-endian, so `rows.txt` carries that seed
per row and the C regenerates the draw rather than shipping it.

`e2e_uniform.bin` and `e2e_noise.bin` split that contract in two, because they
are held to different standards:

* the **uniform stream** (MT19937 + tempering + the 24-bit uniform) must be
  **bit-exact** against `torch.rand`. It is;
* the **Box-Muller output** is held to `|delta| <= 1e-5`, because `logf`,
  `cosf` and `sinf` are not bit-identical between the C library the test links
  and the one PyTorch was built against. Measured on this fixture: 427 of 1660
  draws differ, by at most 1.9e-6.

## Verify

```bash
cd mcu/test/fixtures/en_us_e12nano
shasum -a 256 -c SHA256SUMS
cd ../../../..
make -C mcu test-nano
python3 tools/audit_e12_nano_parameters.py
```
