# Board results

saanoTTS on real silicon. **We have measured two chips.** Everything else here
is either a projection or an empty row waiting for someone with the hardware.

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
- **eff MMAC/s** — effective int8 throughput, `45 / RTF`. The workload is a
  measured constant of ~45 MMAC/s per second of audio.
- **corr / rms_ratio** — correctness against the shipped golden fixture.
  Gates are `corr > 0.98` and `0.80 < rms_ratio < 1.25`. **A speed number
  without a passing correctness gate is not a result** — correlation alone is
  scale-invariant, and a build with a defective integer iFFT once scored
  corr 0.989 while emitting samples ~500× hot.

## Requirements

- **Flash: ~820 KB** (731 KB of embedded weights plus code). This is a hard
  floor — a 512 KB-flash part cannot hold the sketch at all. Verified: the
  Nucleo-F411RE link fails with `region 'FLASH' overflowed by 263976 bytes`.
- **Free heap: 88 KB minimum.** The sketch walks a ladder from 320 KB down to
  88 KB and takes the largest arena `malloc` will give it. Measured on the
  host against this fixture, output is **bit-identical at every rung** —
  corr 0.989148, rms_ratio 0.935022 from 88 KB all the way to 320 KB. Below
  88 KB the runtime reports `ARENA OOM` and stops.
- Spare arena is **speed, not correctness**: it holds opportunistic
  weight-residency buffers that keep decoder weights out of flash. So an RTF
  measured with a 96 KB arena is not comparable to one measured with 320 KB
  even on the same chip. That is why every report includes `arena_bytes`.

## Build-verified boards

22 targets compile and link today (arduino-cli 1.5.2). Building is not
measuring — every one of these still needs someone to flash it and post
numbers.

**Espressif** (`esp32:esp32` 3.3.11) — ESP32-S3, ESP32-C3, ESP32 classic,
ESP32-P4, ESP32-S2, ESP32-C6, ESP32-H2

**Teensy** (`teensy:avr` 1.62.0) — 4.1, 4.0, MicroMod, 3.6

**Raspberry Pi** (`rp2040:rp2040` 6.0.0) — Pico (RP2040), Pico 2 (RP2350)

**STM32** (`STMicroelectronics:stm32` 3.0.0) — Nucleo-H743ZI2, Nucleo-F767ZI,
Nucleo-F429ZI

**Arduino brand** (`arduino:mbed_*` 4.6.0) — GIGA R1 WiFi, Portenta H7
(needs a non-default flash split, below), Nicla Vision, Opta, Nano 33 BLE,
Nano RP2040 Connect

Notes:

- **Portenta H7** defaults to a 50/50 M7/M4 flash split, leaving the sketch
  1 MB and overflowing by 132,264 bytes. `split=75_25` or `split=100_0` links.
- The **Arduino mbed cores define no `F_CPU`**, so `cpu_hz` reports `unknown`
  there. RTF is measured directly, so this costs nothing.

## Known not to fit

Flash is the wall, not speed. Measured link failures:

| Board | Flash | Short by |
|---|---|---|
| Arduino UNO R4 WiFi | 256 KB | 576,884 B |
| Arduino MKR Zero (SAMD21) | 256 KB | over 1 MB |
| Teensy 3.5 | 512 KB | 287,872 B |
| Nucleo-F411RE | 512 KB | 264,168 B |

Anything in the 256–512 KB flash class is out; there is no flag that shrinks
the weights.

## Measured

| Board | MCU | Clock | RTF | eff MMAC/s | corr | Kernels | Source |
|---|---|---:|---:|---:|---:|---|---|
| ESP32-S3 devkit | Xtensa LX7 ×2, PIE SIMD | 240 MHz | **0.22** | ~205 | 0.985 | PIE asm + esp-nn | ours |
| ESP32-C3 | RV32IMC, no FPU | 160 MHz | **5.72** | ~7.9 | pass | scalar ref | ours |
| host (POSIX) | — | — | — | — | 0.989148 | scalar ref | CI gate |

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
