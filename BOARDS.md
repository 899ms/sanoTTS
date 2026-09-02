# Board results

saanoTTS on real silicon. **We have measured two chips.** Everything else here
is either a projection or an empty row waiting for someone with the hardware.

If you have one of these boards, `arduino/examples/BoardBenchmark` is a single
sketch with no peripherals — no DAC, no SD card, no filesystem — that prints a
report block. [Open an issue with it.](../../issues/new?template=board-report.yml)

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

These all compile and link today (arduino-cli 1.5.2, cores as noted). Building
is not measuring — every one of these still needs someone to flash it and post
the numbers.

| Target | Core | Links | Note |
|---|---|---|---|
| ESP32-S3 | esp32:esp32 3.3.11 | yes | 79% flash, ~249 KB heap free |
| ESP32-C3 | esp32:esp32 3.3.11 | yes | 80% flash, ~257 KB heap free |
| ESP32 (classic) | esp32:esp32 3.3.11 | yes | 79% flash, ~249 KB heap free |
| Teensy 4.1 | teensy:avr 1.62.0 | yes | 757 KB in flash, 512 KB RAM2 free |
| Teensy 4.0 | teensy:avr 1.62.0 | yes | same, 1.2 MB flash left for files |
| Teensy 3.6 | teensy:avr 1.62.0 | yes | 814 KB of 1 MB flash |
| RP2040 (Pico) | rp2040:rp2040 6.0.0 | yes | 39% flash, ~192 KB heap free |
| RP2350 (Pico 2) | rp2040:rp2040 6.0.0 | yes | 19% flash, ~449 KB heap free |
| Nucleo-H743ZI2 | STM32 3.0.0 | yes | 38% flash, ~465 KB heap free |
| Nucleo-F411RE | STM32 3.0.0 | **no** | 512 KB flash — too small, by design |

Two portability traps were found and fixed this way, so you should not hit
them:

- **Teensy 4.x** routes plain `.rodata` into *DTCM*, not flash, so the weight
  tables overflowed 512 KB of tightly-coupled RAM by 291,104 bytes. The
  generated header now marks every array `PROGMEM`, which Teensyduino defines
  as `section(".progmem")` and the linker script keeps in FLASH. On the other
  cores `PROGMEM` is empty, so it costs nothing.
- **arduino-pico and stm32duino** both pass `-DBOARD_NAME="<board>"` on the
  compiler command line, which collided with a variable of that name in the
  sketch and broke every one of their targets with `expected unqualified-id
  before string constant`. The sketch now uses `SANOTTS_BOARD` and
  auto-detects the board name on all four core families.

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
