# saanotts-mcu — portable neural TTS runtime for microcontrollers

One C99 core, one small port API, any MCU. The full sanoTTS stack
(phoneme IDs -> duration -> acoustic -> iSTFT decoder -> PCM) in int8,
validated bit-exact against the PyTorch reference on every platform via
embedded golden vectors.

To our knowledge this is the first **real-time** full neural TTS stack
(text -> PCM) to run on a general-purpose microcontroller with **no
dedicated neural accelerator**: real-time on the ESP32-S3 using only its
int8 SIMD (0.22x real-time, 4.5x faster than playback). Prior on-device
neural TTS either required an NPU (TinyTTS: Cortex-M55 + Ethos-U55) or ran
vocoder-only (TinyVocos: ARM Cortex-M, mel->waveform). The same core runs
the complete stack offline / near-real-time on RISC-V (ESP32-C3); the C3 is
not a real-time target. We make no "smallest model" or unqualified "first"
claim -- the interesting results are the measured per-tier numbers below and
the honest evaluation behind them.

## Measured workload (the numbers that define feasibility)

| Requirement | whole-utterance | streaming (planned) |
| --- | --- | --- |
| compute | ~45 MMAC/s of int8 per second of audio | same |
| weights (int8, R7 en_US stack) | 640 KB flash, SRAM-resident copies staged | same |
| RAM working set | ~300 KB | ~130-160 KB |
| float ops | glue only (scales, norm, iSTFT); FPU strongly recommended | same |

Reference point: ESP32-S3 (dual LX7 @ 240 MHz, PIE int8 SIMD) runs this
at **0.22x real-time** (4.5x faster than playback), output correlation
0.985 vs the float reference.

## MCU classes

**Tier V — vector int8 (real-time, margin).**
128-bit int8 SIMD + >=400 KB SRAM. The three kernels map to native
vector MACs (16+ MAC/cycle).
- ESP32-S3 (Xtensa PIE): **reference port, measured 0.22x RT**
- Cortex-M55/M85 (Helium/MVE): Renesas RA8, Alif Ensemble — projected
  <0.1x RT at 400-480 MHz via CMSIS-NN or hand MVE
- ESP32-P4 (RV32 + vendor SIMD): pending esp-nn support check

**Tier D — dual-MAC DSP (real-time, tight).**
SMLAD-class int16/int8 dual MACs, >=512 KB SRAM.
- Cortex-M7 @ 480-600 MHz (STM32H7, Teensy 4.x, i.MX RT): projected
  0.2-0.5x RT with SXTB16+SMLAD int8 kernels
- Cortex-M4 @ >=168 MHz: offline / near-RT for short utterances

**Tier S — scalar (near-RT to offline, correctness-identical).**
Any 32-bit MCU with the RAM floor. Reference C kernels, no assembly.
- ESP32-C3 class (RV32IMC 160 MHz, no FPU): projected 1.5-2x RT with
  the int16-activation refactor; today's float glue makes it slower
- Anything else that can hold the working set

**Tier N — NPU offload (quality scaling, future).**
Ethos-U55 (Alif, Renesas RA8P1, Himax WE2), ST Neural-ART (STM32N6).
256 MAC/cycle class: run a *bigger, better* model in real-time instead
of a faster small one. Needs static activation scales + vendor graph
compiler; the iSTFT and orchestration stay on this runtime.

## Library design

```
mcu/
  include/snt_port.h   <- THE port API: 3 kernels + 4 shims. Port = this.
  include/snt_tts.h    <- public API
  src/snt_tts.c        <- the entire pipeline, platform-free C99
  src/snt_kernels_ref.c<- scalar reference kernels (Tier S default)
  ports/host/          <- POSIX port (CI + golden gate)
  ports/esp32s3/       <- PIE asm kernels + FreeRTOS worker + IDF glue
  ports/wasm/          <- WebAssembly port: full stack in the browser, no server
  test/golden_main.c   <- bit-exactness gate vs PyTorch golden vectors
  test/fixtures/       <- small versioned model/golden contract used by CI
```

Host verification is self-contained:

```bash
make -C mcu test
```

The default fixture is `test/fixtures/en_us_r7`. The historical
`test/golden_c3` path is a compatibility link to the same bytes.

Runs in a browser too: `ports/wasm/` compiles the same core to WebAssembly
(`bash mcu/ports/wasm/build.sh`), and `web/index.html` synthesizes a full
utterance client-side and shows the golden correlation computed live in-page.
The WASM golden gate (`node mcu/ports/wasm/verify_node.mjs`) reproduces the
host result (corr 0.987 vs the PyTorch reference). Browser speed is a desktop-
CPU figure (~50-60x real time), not an MCU measurement.

Port API (complete):
- `int32_t snt_dot_s8(const int8_t *a, const int8_t *b, int len)`
- `void snt_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
   int rows, int len)` — weights row-contiguous, n16-padded, 16B tail pad
- `int snt_weights_resident(const void *p)` — may SIMD read p? (SRAM test)
- `void snt_par_run(snt_par_fn f, int n, void *ctx)` — optional 2nd core;
   default runs serial. All parallel sections are column-disjoint with
   barriers; a port never needs to know the model.
- `SNT_NOW_US()` — profiling only
- memory: the CALLER hands the library one arena buffer; the library
  never allocates. Deterministic bump layout, documented peak per model.

Model format: per-output-channel symmetric int8 weight blobs + generated
offset headers (zero parse, zero copy from flash-mapped storage).
Activations: per-frame symmetric dynamic int8 (no calibration shipping),
frozen calibrated GroupNorm statistics baked at export.

Correctness contract: every port must pass `test/golden_main.c` with
correlation >= 0.98 against the shipped golden audio; the scalar
reference kernels define the exact integer semantics SIMD must match.

## The piperlite lineage, in int8 end to end

`snt_tts.c` above is the R7 lineage. The piperlite lineage is a separate,
larger stack — a `token_context` acoustic student feeding a 192-channel latent
into an upsampling waveform decoder — and it now runs entirely from int8
weights:

```
include/snt_front_f32.h  src/snt_front_f32.c   <- front half, fp32 reference
include/snt_front_q8.h   src/snt_front_q8.c    <- front half, int8
include/snt_piperlite.h  src/snt_piperlite.c   <- decoder, fp32 reference
include/snt_piperlite_q8.h src/snt_piperlite_q8.c <- decoder, int8
test/piperlite_e2e_main.c                      <- ids -> PCM, both stacks
```

Unlike `snt_tts.c`, these read every dimension **from the blob**, not from a
generated header. `src/model/front_q8_meta.h` is why: its `#define`s pinned
that runtime to one model, so the int8 front that existed could not be pointed
at a second voice. `snt_front_q8_init()` takes dims, the output-adapter shape
and the activation-clip table out of `front_meta_q8.bin`, checks every slot's
size against the shape the dims imply, checks the widest fan-in against the
int32 accumulator, and refuses anything that does not add up.

```bash
make -C mcu test-front            # fp32 front vs PyTorch (exact durations)
make -C mcu test-front-q8         # int8 front vs the same goldens
make -C mcu test-front-q8-f32act  # int8 weights, fp32 activations
make -C mcu test-front-q8-negative# the blob refusals
make -C mcu test-piperlite        # fp32 decoder vs PyTorch
make -C mcu test-piperlite-q8     # int8 decoder
make -C mcu e2e                   # int8 front + int8 decoder -> wavs
```

Blobs come from `tools/export_front_golden.py` then `tools/export_front_q8.py`
(front) and `tools/export_piperlite_golden.py` then
`tools/export_piperlite_q8.py` (decoder), into the same directory. Both int8
exporters want a calibration pack; `tools/make_front_latent_pack.py` builds one
from arbitrary text using the voice's own front, so a shipped
`roota.raw-fp16.v1` package is enough to get here — `export_front_q8.py
--package` repacks one in memory.

Measured on three voices (`experiments/evidence/piperlite-int8-20260916.json`):
latent correlation 0.99996–0.99998 against the fp32 reference, end-to-end
waveform correlation 0.9939–0.9995, Whisper CER unchanged within noise, and the
whole stack 3.9x smaller than its fp32 export (amy: 1,479,204 B vs 5,818,120 B).

Two things to know before changing the front exporter. First, the duration
student does NOT reproduce PyTorch's frame counts exactly in int8 and cannot be
made to — its head is `exp()` then round-half-to-even, a hard decision boundary
— so `front_q8_golden_test.c` gates a tolerance (<= 1 frame per token, <= 1% of
the total) and the latent correlation is measured on the *golden* durations so
a one-frame shift cannot masquerade as quantisation error. Second, the
MSE-optimal clip search the decoder exporter uses is actively harmful here:
it clipped amy's duration planes to 0.56x of their range for no latent gain and
cost four tokens their frame count. `export_front_q8.py` defaults to
`--calib-mode max`.

## Hard-won portability rules (measured, not theoretical)

1. SIMD reads require resident operands — flash-XIP vector loads return
   garbage silently on ESP32-S3 (corr 0.011 at full speed).
2. Residency is about WHERE bytes live, not model size: a 36k-param
   stage cost 3x a 74 KB matrix because its weights sat in flash.
3. libm is not free: lroundf ~50 cycles, float div ~40, doubles are
   software-emulated on FPUs without double support (~300 ms of pure
   instrumentation cost). The core ships division-free fast math.
4. Idle busy-wait workers poison the memory bus (~10% on everything);
   parallel workers must block, not spin.
5. malloc at 95% utilization is bin-packing roulette; the arena is the
   only allocation strategy that survives.
