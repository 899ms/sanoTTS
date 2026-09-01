# en_us_r227f32 Golden Fixture

The 2,272,145-parameter release stack (sanotts-tiny-227) with FLOAT32 weight rows (--weights f32): duration 131,652 + e6-mel100-fullcorpus acoustic 681,227 + e8h-tinyvocos-noisefed decoder 1,459,266 (step 200000); the browser build's blob, exported on k2 2026-09-02

Layout, contract and verification are identical to
`en_us_e12nano` (see its README): blobs + frozen durations +
float PyTorch reference waveforms + the split noise contract.
Assembled by `tools/make_nano_fixture.sh` from a
`tools/export_e12_nano_q8.py` export; the export report is
`export-report.json` beside the original export.

Operators (from `mcu/models/en_us_r227f32/nano_q8_meta.h`):
norm_type=0 (0=LayerNorm 1=DyT), act_type=0
(0=GELU 1=ReLU), decoder width 192.

```bash
cd mcu/test/fixtures/en_us_r227f32 && shasum -a 256 -c SHA256SUMS && cd -
make -C mcu test-nano-wf32 NANO_GOLDEN=test/fixtures/en_us_r227f32 NANO_MODEL=models/en_us_r227f32
```

## Why this fixture is float32

The same export as int8 rows (`--weights int8`, the device format) was gated
first and FAILED: 8 rows, mean corr 0.967099, **MIN corr 0.951106** (row
000002_i400), rms ratio 0.989-1.002 -- below the 0.98 gate every nano lineage
is held to. Float activations over the int8 rows (`make test-nano-f32act`)
recovered only to 0.962587 minimum, so the loss is in the weight rows, not the
activation quantiser. With float32 rows this fixture measures **1.000000** on
every row (`make test-nano-wf32`). The int8 fixture was therefore not kept; the
browser voice `heart` ships this float blob (`web/voices/heart/`).
