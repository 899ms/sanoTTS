#!/usr/bin/env python3
"""gen_board_benchmark_data.py -- embed everything BoardBenchmark needs.

examples/BoardBenchmark is deliberately self-contained: no filesystem, no SD
card, no partition scheme, no DAC.  Someone bringing up a board we have never
tested should be able to flash one sketch and read a number off the serial
console.  That means the weights, the phoneme ids and the reference waveform
all have to live in flash.

This writes ONE header holding:
  * front_q8.bin  (~280 KB) and model_q8.bin (~400 KB), 16-byte aligned
    because the kernel contract requires it;
  * the golden phoneme ids;
  * the golden reference waveform as int16, so the sketch can prove
    correctness before it reports a speed.  int16 costs half the flash of the
    float32 fixture and changes the correlation by <1e-4, because correlation
    is invariant to scale and near-invariant to 16-bit quantisation.

Every array is marked PROGMEM.  On most cores that macro is empty and the
data lands in .rodata, which is already flash-resident.  On Teensy 4.x it is
NOT empty: cores/teensy4/avr/pgmspace.h defines it as
section(".progmem"), and imxrt1062.ld routes plain .rodata to *DTCM* while
routing .progmem to FLASH.  Without the annotation the linker tries to copy
all ~731 KB into 512 KB of DTCM and fails ("region `DTCM' overflowed by
291104 bytes" -- measured).  Flash is memory-mapped on every core we target,
so ordinary pointer reads still work; no pgm_read_* accessors are needed.

Flash cost is ~820 KB, which fits Teensy 4.x (8 MB), RP2040 (2 MB) and the
ESP32 family.  It will NOT fit a 512 KB-flash part -- those need the
LittleFS route in examples/SpeakGolden instead.

Two fixture layouts are supported and detected automatically:

  R7 line (snt_tts.c)   -- e2e_ids.bin / e2e_durs.bin / e2e_audio.bin
  nano line (snt_nano.c) -- rows.txt plus rNN_ids/durs/audio.bin, and a
                            per-row sha256(row_id) decoder seed

The nano decoder is noise-fed, so its output is only reproducible with the
row's exact seed; rows.txt carries it and it is emitted as
SANOTTS_BENCH_SEED. The generated header sets SANOTTS_BENCH_NANO so the
sketch selects the matching runtime.

The reference waveform is truncated to --ref-samples (default 34,304, about
1.56 s). Correlation over that prefix gates the build just as well as the
whole utterance, and the full waveform is still synthesized and timed -- but
a nano row is 4.8-7.3 s, and shipping all of it as int16 would add ~210 KB of
flash for nothing.

Usage (from the arduino/ directory):
  python3 extras/gen_board_benchmark_data.py \
    --fixture ../mcu/test/fixtures/en_us_e12nano \
    --out examples/BoardBenchmark/sanotts_bench_data.h
"""
from __future__ import annotations

import argparse
from pathlib import Path

import struct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="Directory holding front_q8.bin, model_q8.bin, e2e_ids.bin, e2e_audio.bin")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--row", type=int, default=0,
                        help="nano fixtures only: which row to embed (default 0, the shortest)")
    parser.add_argument("--ref-samples", type=int, default=34_304,
                        help="how many reference samples to embed for the correlation gate")
    parser.add_argument("--model-name", default=None,
                        help="label reported by the sketch; defaults to the fixture directory name")
    return parser.parse_args()


def load_fixture(fixture, row_index):
    """Return (front, model, ids, durs, audio, seed, is_nano).

    seed is None on the R7 line, which has no noise input.
    """
    front = (fixture / "front_q8.bin").read_bytes()
    model = (fixture / "model_q8.bin").read_bytes()
    rows_file = fixture / "rows.txt"
    if not rows_file.exists():
        return (front, model,
                (fixture / "e2e_ids.bin").read_bytes(),
                (fixture / "e2e_durs.bin").read_bytes(),
                (fixture / "e2e_audio.bin").read_bytes(),
                None, False)

    rows = [line.split() for line in rows_file.read_text().split("\n") if line.strip()]
    if not 0 <= row_index < len(rows):
        raise SystemExit(f"--row {row_index} out of range; {rows_file} has {len(rows)} rows")
    row_id, _tokens, frames, samples, seed = rows[row_index][:5]
    prefix = f"r{row_index:02d}_"
    ids = (fixture / f"{prefix}ids.bin").read_bytes()
    durs = (fixture / f"{prefix}durs.bin").read_bytes()
    audio = (fixture / f"{prefix}audio.bin").read_bytes()
    if len(audio) // 4 != int(samples):
        raise SystemExit(f"{prefix}audio.bin holds {len(audio)//4} samples, rows.txt says {samples}")
    print(f"  nano row {row_index} = {row_id}: {frames} frames, {samples} samples, seed {seed}")
    return front, model, ids, durs, audio, int(seed), True


# int8 MACs per second of audio, per stack. This is NOT a shared constant:
# it is a property of the graph, and using one stack's figure for another
# silently reports a wrong throughput. Sources:
#   en_us_r7        45.0  -- docs/mcu-classes-and-porting.md section 1
#   en_us_e12nano   19.0  -- 138,507,952 MACs over r06 (160,768 samples,
#                            7.291 s) from the ESP32-S3 residency counters,
#                            experiments/evidence/serial-logs/, 2026-08-22
# A lineage that is not listed emits 0, and the sketch then prints "n/a"
# rather than a number derived from someone else's graph.
MMAC_PER_SECOND = {
    "en_us_r7": 45.0,
    "en_us_e12nano": 19.0,
}


def emit_bytes(handle, name: str, blob: bytes) -> None:
    handle.write(f"/* {len(blob)} bytes */\n")
    handle.write(f"alignas(16) static const unsigned char {name}[{len(blob)}] PROGMEM = {{\n")
    for offset in range(0, len(blob), 16):
        row = ", ".join(f"0x{b:02x}" for b in blob[offset : offset + 16])
        handle.write(f"  {row},\n")
    handle.write("};\n\n")


def main() -> None:
    args = parse_args()
    fixture = args.fixture
    front, model, ids_raw, durs_raw, audio_raw, seed, is_nano = load_fixture(fixture, args.row)
    model_name = args.model_name or fixture.name

    if len(ids_raw) % 4 or len(audio_raw) % 4:
        raise SystemExit("e2e_ids.bin / e2e_audio.bin must be 4-byte records")
    ids = list(struct.unpack(f"<{len(ids_raw)//4}i", ids_raw))
    durs = list(struct.unpack(f"<{len(durs_raw)//4}i", durs_raw))
    if len(durs) != len(ids):
        raise SystemExit(f"durations {len(durs)} != ids {len(ids)}")
    audio = list(struct.unpack(f"<{len(audio_raw)//4}f", audio_raw))
    if not audio:
        raise SystemExit("empty reference audio")
    total_samples = len(audio)
    if args.ref_samples > 0:
        audio = audio[: args.ref_samples]

    peak = max(abs(v) for v in audio)
    if peak <= 0.0:
        raise SystemExit("reference audio is silent")
    # Normalise to full int16 scale; correlation is scale-invariant so this
    # loses nothing and keeps the quantisation error minimal.
    scale = 32767.0 / peak
    pcm16 = [max(-32768, min(32767, int(round(v * scale)))) for v in audio]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("/* Auto-generated by extras/gen_board_benchmark_data.py"
                     " -- do not edit by hand.\n"
                     f" * Source fixture: {fixture}\n"
                     " * Weights, golden phoneme ids and the int16 reference waveform,\n"
                     " * so examples/BoardBenchmark needs no filesystem and no peripherals.\n"
                     " */\n#pragma once\n#include <stdint.h>\n#include <stddef.h>\n\n"
                     "/* Teensy 4.x needs PROGMEM to keep this out of DTCM (see the\n"
                     " * generator's docstring). Cores that do not define it, and host\n"
                     " * builds, get a harmless empty macro. */\n"
                     "#ifndef PROGMEM\n#define PROGMEM\n#endif\n\n")
        handle.write(f"#define SANOTTS_BENCH_SAMPLE_RATE {args.sample_rate}\n")
        handle.write(f"#define SANOTTS_BENCH_N_IDS {len(ids)}\n")
        handle.write(f"#define SANOTTS_BENCH_N_REF {len(pcm16)}\n")
        # Correlation is scale-invariant, so the golden gate ALSO checks the
        # RMS ratio -- a build with a defective integer iFFT once scored
        # corr 0.989 while emitting samples ~500x hot. Peak-normalising the
        # reference would destroy that check, so ship the scale back with it:
        # reference_float = SANOTTS_BENCH_REF[i] / SANOTTS_BENCH_REF_SCALE.
        handle.write(f"#define SANOTTS_BENCH_REF_SCALE {scale:.9f}f\n")
        handle.write(f'#define SANOTTS_BENCH_MODEL "{model_name}"\n')
        # Total synthesized length, which is what RTF is computed over. The
        # embedded reference is only the correlation prefix, so the sketch
        # must not infer audio duration from SANOTTS_BENCH_N_REF.
        handle.write(f"#define SANOTTS_BENCH_N_SAMPLES {total_samples}\n")
        handle.write(f"#define SANOTTS_BENCH_MMAC_PER_S "
                     f"{MMAC_PER_SECOND.get(model_name, 0.0):.1f}f\n")
        if is_nano:
            # The nano decoder is noise-fed: without this exact seed the output
            # is a different (valid) waveform and the correlation is meaningless.
            handle.write("#define SANOTTS_BENCH_NANO 1\n")
            handle.write(f"#define SANOTTS_BENCH_SEED {seed}ULL\n")
        handle.write("\n")

        emit_bytes(handle, "SANOTTS_BENCH_FRONT_Q8", front)
        emit_bytes(handle, "SANOTTS_BENCH_MODEL_Q8", model)

        handle.write(f"static const int32_t SANOTTS_BENCH_IDS[{len(ids)}] PROGMEM = {{\n")
        for offset in range(0, len(ids), 12):
            handle.write("  " + ", ".join(str(v) for v in ids[offset : offset + 12]) + ",\n")
        handle.write("};\n\n")

        # The golden durations are passed as dur_override so every board
        # synthesizes the SAME number of frames. Without them the duration
        # model predicts its own timing, the output length drifts, and both
        # the correlation gate and the cross-board timing become meaningless.
        handle.write(f"static const int32_t SANOTTS_BENCH_DURS[{len(durs)}] PROGMEM = {{\n")
        for offset in range(0, len(durs), 12):
            handle.write("  " + ", ".join(str(v) for v in durs[offset : offset + 12]) + ",\n")
        handle.write("};\n\n")

        handle.write("/* Golden reference, int16, peak-normalised. Correlation only. */\n")
        handle.write(f"static const int16_t SANOTTS_BENCH_REF[{len(pcm16)}] PROGMEM = {{\n")
        for offset in range(0, len(pcm16), 16):
            handle.write("  " + ", ".join(str(v) for v in pcm16[offset : offset + 16]) + ",\n")
        handle.write("};\n")

    total = args.out.stat().st_size
    print(f"wrote {args.out} ({total/1024:.0f} KB source)")
    print(f"  front  {len(front):>7} bytes")
    print(f"  model  {len(model):>7} bytes")
    print(f"  ids    {len(ids):>7} entries")
    print(f"  ref    {len(pcm16):>7} samples ({len(pcm16)/args.sample_rate:.2f} s @ {args.sample_rate} Hz)"
          f" of {total_samples} synthesized ({total_samples/args.sample_rate:.2f} s)")
    print(f"  flash  ~{(len(front)+len(model)+len(pcm16)*2+len(ids)*4)/1024:.0f} KB of const data")


if __name__ == "__main__":
    main()
