# en_us_e12nano_short — short benchmark row

A 255-frame (2.95 s) prefix of `en_us_e12nano` row `000001_i200`, cut on a
token boundary at 51 of its 73 phonemes. Same weights, same
`sha256(row_id)` noise seed.

## Why it exists

The arena is ~46.5 KB fixed + 196 B/frame and must be **one contiguous
block**. The full 415-frame row needs 128,944 B, which several real boards
will not hand out in a single piece — a classic ESP32 (ESP32-D0WD-V3)
reports 250,040 B free but a largest block of 110,580 B, and fails. This row
peaks at **98,224 B**, which fits with ~12 KB to spare.

## What the reference IS, and is NOT

`r00_audio.bin` here is the **host C runtime's** output, not the float
PyTorch model's. A new input needs a new PyTorch run, and this row was
derived without one.

That makes it a **port** gate, not a **model** gate. It answers "does this
board's build reproduce the reference C implementation on this input?" and
catches wrong kernels, bad alignment, SIMD reading unstageable memory, and
endianness — every class of porting bug. It cannot catch a wrong model.

The model gate is unchanged and still lives next door:

```bash
make -C mcu test-nano   # en_us_e12nano vs PyTorch, threshold 0.98
```

The chain is: PyTorch gates the host runtime, the host runtime gates the
board. The runtime that produced this file passes that first gate at
min corr 0.984263 over 8 rows.

## Regenerating

```bash
make -C mcu nano_short_row
./mcu/nano_short_row mcu/test/fixtures/en_us_e12nano 0 256 \
    mcu/test/fixtures/en_us_e12nano_short
```

Deterministic: repeated runs are byte-identical.
