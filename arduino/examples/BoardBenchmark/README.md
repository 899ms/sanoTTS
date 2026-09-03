# BoardBenchmark

Flash one sketch, get one block of numbers, paste it into an issue. No DAC,
no SD card, no filesystem — the only peripheral is Serial.

Runs the **294,642-parameter** voice. Needs ~500 KB flash and **~100 KB of
free RAM in one contiguous block**. 22 boards are compile-verified
(arduino-cli 1.5.2); two are measured on hardware:

| Board | RTF | |
|---|---:|---|
| ESP32-S3 | **0.41** | 2.4× faster than real time (SIMD, automatic) |
| ESP32 classic | 2.17 | scalar; LX6 has no vector unit |

Yours is probably not on that list yet — that is what you are here for.

## 1. Install the library (all boards)

1. Download **[SanoTTS.zip](https://github.com/Ampixa/sanoTTS/releases/download/arduino-lib-0.1.0/SanoTTS.zip)** (1.1 MB).
2. Arduino IDE → **Sketch → Include Library → Add .ZIP Library…** → pick it.
3. **File → Examples → SanoTTS → BoardBenchmark**.

Use that zip, not GitHub's green *Code → Download ZIP*: `library.properties`
lives inside `arduino/` rather than at the archive root, so the IDE rejects
the source download. The release asset is rooted correctly.

## 2. Select your board

Add the board-manager URL under **File → Preferences → Additional boards
manager URLs**, install the core from **Tools → Board → Boards Manager**, then
pick the board.

### ESP32-S3 / ESP32-C3 / ESP32

- URL: `https://espressif.github.io/arduino-esp32/package_esp32_index.json`
- Core: **esp32 by Espressif Systems** (3.3.x)
- Board: `ESP32S3 Dev Module`, `ESP32C3 Dev Module`, or `ESP32 Dev Module`
- Default partition scheme is fine (sketch is ~80% of the 1.3 MB app slot).

### Teensy 4.1 / 4.0 / 3.6

- URL: `https://www.pjrc.com/teensy/package_teensy_index.json`
- Core: **Teensy** (1.62.x)
- Board: `Teensy 4.1`, `Teensy 4.0`, or `Teensy 3.6`
- Upload uses Teensy Loader, which the core installs. Press the board's
  button if it does not auto-reboot.

### Raspberry Pi Pico / Pico 2 (RP2040 / RP2350)

- URL: `https://github.com/earlephilhower/arduino-pico/releases/download/global/package_rp2040_index.json`
- Core: **Raspberry Pi Pico/RP2040/RP2350 by Earle Philhower** (6.x)
- Board: `Raspberry Pi Pico` or `Raspberry Pi Pico 2`
- First upload: hold **BOOTSEL** while plugging in, then upload. After that,
  the IDE can reset it over USB.

### STM32 Nucleo (H743ZI2, H745ZI-Q / H755ZI-Q, F767ZI, F429ZI)

- URL: `https://raw.githubusercontent.com/stm32duino/BoardManagerFiles/main/package_stmicroelectronics_index.json`
- Core: **STM32 MCU based boards** (3.x)
- Board: `Nucleo-144`, then **Tools → Board part number → …**
- **Nucleo-H755ZI-Q:** the core has no H755 entry. Select **Nucleo H745ZI-Q** —
  it uses the `H745Z(G-I)T_H755ZIT` variant, which is your die.
- **Tools → U(S)ART support: `Enabled (generic 'Serial')`**
- **Tools → USB support: `None`.** If you pick *CDC (generic 'Serial'
  supersede U(S)ART)*, `Serial` moves to the user USB connector and the
  ST-Link COM port stays silent.
- Upload needs **STM32CubeProgrammer** installed (the core calls it).
- **Opening the serial monitor does not reset a Nucleo.** The sketch starts
  the moment flashing finishes, so its first report is printed before you can
  open the port. It repeats every 12 s until you press a key, so just wait —
  or press Enter to run it immediately.
- 512 KB-flash parts (e.g. Nucleo-F411RE) do not fit; the linker will say so.

### Arduino GIGA R1 / Portenta H7 / Nicla Vision / Opta / Nano 33 BLE / Nano RP2040 Connect

- No extra URL needed — install **Arduino Mbed OS \<family\> Boards** from
  Boards Manager (4.6.x).
- **Portenta H7 only:** the default flash split gives the M7 just 1 MB and the
  sketch overflows it by 132,264 bytes. Set **Tools → Flash split →
  `1.5MB M7 + 0.5MB M4`** (or `2MB M7 + M4 in SDRAM`) and it links.
- These cores define no `F_CPU`, so `cpu_hz` prints `unknown`. That is
  expected and harmless — RTF is measured, not derived from the clock. Put
  your board's clock in the issue instead.
- **Nano 33 BLE** fits, but only just: 87% of its 983 KB. It reports about
  160 KB of free RAM, so it runs on a middle rung of the arena ladder rather
  than the top one — that is fine, and `arena_bytes` will say which.

### Something else

If it has ≥1 MB of flash available to the sketch, ≥128 KB RAM and an Arduino
core, try it. It either links or the linker tells you exactly why not.

## Known not to fit

Flash, not speed, is the wall. These fail at link with a clear message:

| Board | Flash short by |
|---|---|
| Arduino UNO R4 WiFi | 576,884 B |
| Arduino MKR Zero (SAMD21) | 1 MB+ |
| Teensy 3.5 | 287,872 B |
| Nucleo-F411RE | 264,168 B |
| Portenta H7 at default 50/50 split | 132,264 B (fix above) |

Anything with 256–512 KB of flash is out. There is no build flag that shrinks
the weights.

## 3. Run it, and hear it

1. **Upload.**
2. **Tools → Serial Monitor**, set **115200 baud**.
3. If the screen is blank, **wait 12 seconds** — the report repeats until you
   press a key. Boards whose USB-serial bridge does not reset the MCU (every
   ST-Link Nucleo, most external programmers) finish their first run before
   you can open the monitor. Pressing Enter also runs it immediately.

Takes a few seconds on fast boards, up to a minute on slow ones.

### Hearing the audio

The benchmark checks its output and throws it away. To actually listen to it,
press **`w`** in the serial monitor: the board streams the utterance out as a
base64 WAV over the same USB cable. Save it with:

```bash
pip install pyserial
python3 extras/wav_from_serial.py <port> out.wav
```

That script sends the `w` for you and writes a playable file. No DAC, no I2S,
no SD card, no wiring.

Verified end to end on an ESP32-S3: the captured WAV correlates **1.000000**
with the host reference across all 65,024 samples.

## 4. What to share

Copy the whole block between `---- REPORT ----` and `---- END REPORT ----`:

```
---- REPORT (paste this whole block) ----
board:        Teensy 4.1
model:        en_us_e12nano
cpu_hz:       600000000
arena_bytes:  106512
rc:           0
frames:       255
samples:      65024
compared:     34304
arena_peak:   98224
audio_s:      2.9489
elapsed_s:    ...
RTF:          ...
eff_MMAC_s:   ...
golden_corr:  1.00...
rms_ratio:    0.99...
verdict:      PASS
---- END REPORT ----
```

Paste it here: **[New board report](https://github.com/Ampixa/sanoTTS/issues/new?template=board-report.yml)**

`verdict: FAIL` is still worth posting — it means the port contract is wrong
somewhere, which we can fix. `RTF` is the headline: below 1.0 is faster than
playback.

## arduino-cli (optional)

Same thing without the IDE:

```bash
arduino-cli config set library.enable_unsafe_install true
arduino-cli lib install --zip-path SanoTTS.zip
```

```bash
SKETCH=~/Arduino/libraries/SanoTTS/examples/BoardBenchmark
arduino-cli compile --upload -p <port> --fqbn <fqbn> $SKETCH
arduino-cli monitor -p <port> -c baudrate=115200
```

| Board | fqbn |
|---|---|
| ESP32-S3 | `esp32:esp32:esp32s3` |
| ESP32-C3 | `esp32:esp32:esp32c3` |
| ESP32 | `esp32:esp32:esp32` |
| Teensy 4.1 | `teensy:avr:teensy41` |
| Teensy 4.0 | `teensy:avr:teensy40` |
| Teensy 3.6 | `teensy:avr:teensy36` |
| Pico | `rp2040:rp2040:rpipico` |
| Pico 2 | `rp2040:rp2040:rpipico2` |
| Nucleo-H743ZI2 | `STMicroelectronics:stm32:Nucleo_144:pnum=NUCLEO_H743ZI2` |
| Nucleo-F767ZI | `STMicroelectronics:stm32:Nucleo_144:pnum=NUCLEO_F767ZI` |
| Nucleo-F429ZI | `STMicroelectronics:stm32:Nucleo_144:pnum=NUCLEO_F429ZI` |
| GIGA R1 WiFi | `arduino:mbed_giga:giga` |
| Portenta H7 | `arduino:mbed_portenta:envie_m7:split=75_25` |
| Nicla Vision | `arduino:mbed_nicla:nicla_vision` |
| Opta | `arduino:mbed_opta:opta` |
| Nano 33 BLE | `arduino:mbed_nano:nano33ble` |
| Nano RP2040 Connect | `arduino:mbed_nano:nanorp2040connect` |
| ESP32-P4 / S2 / C6 / H2 | `esp32:esp32:esp32p4` / `esp32s2` / `esp32c6` / `esp32h2` |
| Teensy MicroMod | `teensy:avr:teensyMM` |

## Regenerating the data header

`sanotts_bench_data.h` is checked in; you do not need this. To rebuild it
from the fixture, from `arduino/`:

```bash
python3 extras/gen_board_benchmark_data.py \
  --fixture ../mcu/test/fixtures/en_us_r7 \
  --out examples/BoardBenchmark/sanotts_bench_data.h
./extras/bench_host_check.sh
```
