# BoardBenchmark

Flash one sketch, get one block of numbers, paste it into an issue. No DAC,
no SD card, no filesystem — the only peripheral is Serial.

Needs ~820 KB flash and 88 KB free RAM. Every board below has been
compile-verified (arduino-cli 1.5.2). None has been measured yet — that is
what you are here for.

## 1. Install the library (all boards)

```bash
git clone https://github.com/Ampixa/sanoTTS
```

Copy the `arduino/` folder into your sketchbook `libraries/` as `SanoTTS`:

| OS | destination |
|---|---|
| Windows | `Documents\Arduino\libraries\SanoTTS` |
| macOS | `~/Documents/Arduino/libraries/SanoTTS` |
| Linux | `~/Arduino/libraries/SanoTTS` |

Restart the Arduino IDE. **File → Examples → SanoTTS → BoardBenchmark**.

Do not use *Add .ZIP Library* on the GitHub download — `library.properties`
is inside `arduino/`, not at the zip root, and the IDE will reject it.

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

### STM32 Nucleo-H743ZI2 (and other ≥1 MB-flash STM32)

- URL: `https://raw.githubusercontent.com/stm32duino/BoardManagerFiles/main/package_stmicroelectronics_index.json`
- Core: **STM32 MCU based boards** (3.x)
- Board: `Nucleo-144`, then **Tools → Board part number → Nucleo H743ZI2**
- Upload needs **STM32CubeProgrammer** installed (the core calls it).
- Serial is the ST-Link virtual COM port. Leave *U(S)ART support* at its
  default, *Enabled (generic Serial)*.
- 512 KB-flash parts (e.g. Nucleo-F411RE) do not fit; the linker will say so.

### Something else

If it has ≥1 MB flash, ≥128 KB RAM and an Arduino core, try it. It either
links or the linker tells you exactly why not.

## 3. Run it

1. **Upload.**
2. **Tools → Serial Monitor**, set **115200 baud**.
3. If the screen is blank, **press Enter** in the monitor — the benchmark
   re-runs on any keypress.

Takes a few seconds on fast boards, up to a minute on slow ones.

## 4. What to share

Copy the whole block between `---- REPORT ----` and `---- END REPORT ----`:

```
---- REPORT (paste this whole block) ----
board:        Teensy 4.1
cpu_hz:       600000000
arena_bytes:  327696
rc:           0
frames:       134
samples:      34304
compared:     34304
audio_s:      1.5557
elapsed_s:    ...
RTF:          ...
eff_MMAC_s:   ...
golden_corr:  0.98...
rms_ratio:    0.93...
verdict:      PASS
---- END REPORT ----
```

Paste it here: **[New board report](https://github.com/Ampixa/sanoTTS/issues/new?template=board-report.yml)**

`verdict: FAIL` is still worth posting — it means the port contract is wrong
somewhere, which we can fix. `RTF` is the headline: below 1.0 is faster than
playback.

## arduino-cli (optional)

Same thing without the IDE. Library goes in `libraries/SanoTTS` under your
sketchbook as above.

```bash
arduino-cli compile --upload -p <port> --fqbn <fqbn> libraries/SanoTTS/examples/BoardBenchmark
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

## Regenerating the data header

`sanotts_bench_data.h` is checked in; you do not need this. To rebuild it
from the fixture, from `arduino/`:

```bash
python3 extras/gen_board_benchmark_data.py \
  --fixture ../mcu/test/fixtures/en_us_r7 \
  --out examples/BoardBenchmark/sanotts_bench_data.h
./extras/bench_host_check.sh
```
