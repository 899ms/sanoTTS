# Board results

saanoTTS on real silicon. **Two boards are measured on the Arduino path** --
an ESP32-S3 at **0.41 xRT, 2.4x faster than real time**, and a classic ESP32
at 2.17 xRT. Both below. Everything else here is either a projection or an empty row
waiting for someone with the hardware.

If you have one of these boards, `arduino/examples/BoardBenchmark` is a single
sketch with no peripherals — no DAC, no SD card, no filesystem — that prints a
report block. [Open an issue with it.](../../issues/new?template=board-report.yml)

Install: download
[SanoTTS.zip](https://github.com/Ampixa/sanoTTS/releases/download/arduino-lib-0.1.0/SanoTTS.zip),
then Arduino IDE → *Sketch → Include Library → Add .ZIP Library…* →
*File → Examples → SanoTTS → BoardBenchmark*.

## What the numbers mean

- **RTF** — seconds of compute per second of audio. `RTF < 1.0` is faster than
  playback. This is the interactive/offline line.
- **eff MMAC/s** — effective int8 throughput, `(MACs per second of audio) /
  RTF`. The workload constant belongs to the stack, not the project: 19 for
  the 294k nano, 45 for the 567k R7. The sketch prints `n/a` rather than
  borrow another graph's figure.
- **corr / rms_ratio** — correctness against the shipped golden fixture.
  Gates are `corr > 0.98` and `0.80 < rms_ratio < 1.25`. **A speed number
  without a passing correctness gate is not a result** — correlation alone is
  scale-invariant, and a build with a defective integer iFFT once scored
  corr 0.989 while emitting samples ~500× hot.

## Requirements

The shipped example runs **en_us_e12nano** (294,642 params), the only nano
lineage with measurements on real silicon.

- **Flash: ~500 KB** (399 KB of embedded weights + reference, plus code).
- **Free heap: 136 KB minimum** for the embedded row (415 frames, 4.81 s).
  The arena is `46.5 KB fixed + 195.7 B/frame` (measured, r2 0.9998), so a
  shorter utterance needs less. The sketch walks a ladder from 320 KB down to
  the floor and takes the largest one `malloc` gives it; output is identical
  at every rung, so the extra space is speed, not correctness.
- `arena_peak` measured **128,944 B** -- identical on the host, on the Arduino
  build and in the ESP-IDF port. It is in every board report.

## Measured

Correctness gates: `corr > 0.98`, `0.80 < rms_ratio < 1.25`. **A speed number
without a passing correctness gate is not a result.**

### ESP32-S3, Arduino library -- 2026-09-04

The library now assembles the Xtensa LX7 PIE SIMD int8 kernels automatically
when the target is an ESP32-S3. Nothing to configure.

| Build | RTF | vs real time | eff MMAC/s | corr |
|---|---:|---:|---:|---:|
| **SIMD (shipped default)** | **0.3828** | **2.6x faster** | 49.6 | 0.994664 |
| SIMD + `SANOTTS_ESP32_DUALCORE` | 0.3863 | 2.6x | 49.2 | 0.994664 |
| scalar C (previous default) | 1.5825 | 0.63x | 12.0 | 0.994664 |
| scalar + `SANOTTS_ESP32_DUALCORE` | 1.5856 | 0.63x | 12.0 | 0.994664 |

**SIMD is a 4.13x speedup and crosses the real-time line.** All four builds
emit bit-identical audio -- correlation is 0.994664 in every one, and equal to
the host. The vector unit changes the speed and nothing else.

**The second core does nothing for this workload.** Dual-core is marginally
*slower* in both pairs (0.2% and 0.9%, within run-to-run spread). The flag
remains available but is not worth setting here.

Model comparison, same board and same scalar kernels, only the model differing:

| Model | Params | RTF (scalar) | corr | Flash data |
|---|---:|---:|---:|---:|
| **en_us_e12nano** | 294,642 | **1.5825** | 0.994664 | 399 KB |
| en_us_r7 | 567,008 | 3.6948 | 0.989048 | 731 KB |

The 294k model is 2.33x faster, 1.83x smaller and correlates better. Host and
device agree exactly on it (`arena_peak` 128,944 B on host, on the Arduino
build and in the ESP-IDF port).

### Both chips, short row (255 frames, 2.95 s) -- 2026-09-04

| Board | MCU | Kernels | RTF | vs real time | eff MMAC/s | corr |
|---|---|---|---:|---:|---:|---:|
| ESP32-S3 | Xtensa LX7 | **PIE SIMD** | **0.4107** | **2.4x faster** | 46.3 | 1.000000 |
| ESP32 classic | Xtensa LX6 | scalar | 2.1720 | 0.46x | 8.7 | 1.000000 |

Both reproduce the host reference **exactly**, which is the point of this
gate: an LX6 running scalar C and an LX7 running hand-written vector assembly
produce bit-identical audio. `arena_peak` is 98,224 B on both, and on the
host.

The classic ESP32 could not run the 415-frame row at all -- 250,040 B free but
a largest contiguous block of 110,580 B against a 128,944 B requirement.
Total free heap badly overstates what these parts hand out at once. The
255-frame row needs 98,224 B and fits with ~12 KB spare.

### ESP-IDF ports with SIMD kernels### ESP-IDF ports with SIMD kernels

Not comparable to the rows above -- these use the PIE assembly and esp-nn,
which the portable Arduino library does not ship.

| Board | MCU | Model | RTF | corr | Source |
|---|---|---|---:|---:|---|
| ESP32-S3 | Xtensa LX7, PIE SIMD | e12nano | **0.185** | 0.9848 | 2026-08-22 |
| ESP32-S3 | Xtensa LX7, PIE SIMD | r7 | 0.22 | 0.985 | earlier |
| ESP32-C3 | RV32IMC, scalar | r7 | 5.72 | pass | earlier |

**The largest speed lever is SIMD, not the model** -- 4.13x, measured above.
The Arduino library now gets 0.383 xRT; the IDF port reaches 0.185 with
esp-nn's tuned single-row dot and further tuning the library does not vendor,
so 2.1x is still on the table.

Residency matters as much: PIE vector loads against flash-XIP silently return
garbage (corr 0.011), so the runtime stages weights into the arena and
dispatches SIMD only for internal SRAM. Backed by PSRAM instead, the IDF port
measures 1.059 rather than 0.185 -- a 5.7x penalty from where weights live.

## Build-verified boards

22 targets compile and link with the 294k example (arduino-cli 1.5.2).
Building is not measuring -- only the ESP32-S3 rows above are measured.

- **Espressif** -- ESP32-S3, C3, classic, P4, S2, C6, H2
- **Teensy** -- 4.1, 4.0, MicroMod, 3.6, **3.5**
- **Raspberry Pi** -- Pico (RP2040), Pico 2 (RP2350)
- **STM32** -- Nucleo-H743ZI2, F767ZI, F429ZI, **F411RE**
- **Arduino** -- GIGA R1 WiFi, Portenta H7, Nicla Vision, Opta, Nano 33 BLE,
  Nano RP2040 Connect
- **Adafruit SAMD51** -- **Metro M4**, **Feather M4**

Bold entries are newly possible at 294k; they could not fit the 567k model.
Portenta H7 no longer needs a non-default flash split, and Nano 33 BLE drops
from 87% of flash to 52%.

### Compiles but will not run

**Nucleo-F411RE** links at 85% of flash but leaves only ~48 KB of RAM, under
the 136 KB floor, so it prints `FATAL: could not allocate` instead of a
number. A shorter fixture row would bring it into range.

### Known not to fit

| Board | Flash | Note |
|---|---|---|
| Arduino UNO R4 WiFi | 256 KB | `.text` will not fit |
| Arduino MKR Zero | 256 KB | far short |
| Adafruit Grand Central M4 | -- | fqbn not present in adafruit:samd 1.7.17 |

## Wanted — highest value first

| Board | MCU | Class | Prediction | Status |
|---|---|---|---|---|
| **Teensy 4.0 / 4.1** | i.MX RT1062, Cortex-M7 | D | 0.2–0.5× RT | **wanted** |
| **Nucleo-H743 / H7 family** | STM32H7, Cortex-M7 | D | 0.2–0.5× RT | **wanted** |
| i.MX RT1060 EVK | Cortex-M7 | D | 0.2–0.5× RT | wanted |
| Alif Ensemble E7 | Cortex-M55 + Helium | V | <0.1× RT | wanted |
| Renesas RA8M1 / RA8D1 | Cortex-M85 + Helium | V | <0.1× RT | wanted |
| ESP32-P4 | RV32 + vendor SIMD | V | real-time | wanted |
| Raspberry Pi Pico / Pico 2 | RP2040 / RP2350 | S | offline | **builds, never measured** |
| Adafruit Metro / Feather M4 | SAMD51, Cortex-M4F | S | offline | **newly fits at 294k** |
| Teensy 3.5 | Cortex-M4F 120 MHz | S | offline | **newly fits at 294k** |
| ESP32 (classic) | Xtensa LX6, no PIE | S/D | between C3 and S3 | wanted |
| Any Cortex-M4F ≥168 MHz | — | S | offline / short utterances | wanted |

One caveat for Teensy 4.x reporters: `malloc` there returns RAM2 (OCRAM),
which is slower than the DTCM the core uses for ordinary variables. A Teensy
number is therefore a floor, not a ceiling, for what an M7 can do.

**Cortex-M7 is the most valuable gap.** It is the one class with no measured
port at all, and it is exactly where the real-time boundary is predicted to
sit — so a Teensy 4.x result decides whether that boundary is real.

## Why a FAIL is still worth posting

A reproducible failure tells us the port contract is wrong somewhere, which is
more actionable than a missing row. Post it with whatever you had to change.
