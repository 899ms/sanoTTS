# ESP32-S3 serial captures for the E12-nano

Raw `cap_nano.py` output from COM5, one file per flashed firmware, in the order
they were taken. Nothing in these files is edited; each begins with the ROM
boot banner and ends with `END`.

Read `00-BASELINE-labelled-COM5.log` and `06-FINAL-labelled-COM5.log` as the
pair that matters: they were produced by the SAME harness binary logic, so the
before/after comparison is like-for-like.

Every measured value in those two logs is prefixed `DEVICE:`. Constants that
were compiled into the binary from a host run are prefixed `HOST-REF:` and are
printed in their own block, never on the same line as a measured value. That
separation is deliberate: a reference number in parentheses beside a measured
one is exactly how a host constant gets misread as a silicon result.

| file | firmware under test |
|---|---|
| `00-BASELINE-labelled-COM5.log` | pre-optimisation `snt_nano.c` + pre-optimisation port, labelled harness |
| `01-baseline-COM5.log`          | same source, earlier capture (independent flash+capture cycle) |
| `02-staging-COM5.log`           | + weight residency staging |
| `03-memory-COM5.log`            | + tiled resblock ring, shared ax/x plane |
| `04-parallel-head-spec-COM5.log`| + parallel head-int8/spectrum, precomputed FFT twiddles |
| `05-parallel-trunk-fft-COM5.log`| + tiled dual-core trunk, paired-frame IFFT |
| `06-FINAL-labelled-COM5.log`    | + tiled embed/stem-LN, labelled harness |

Board: ESP32-S3 rev 2, 2 cores, 240 MHz, 8 MB octal PSRAM, ESP-IDF v6.0.1.
Protocol: one discarded warm-up, then 5 timed runs, median reported, spread
shown. Correlation is computed on-chip from the chip's own PCM in a separate
untimed run.

The espeak SPIFFS at 0x390000 was re-read from the board after the last flash
and hashes to `F7B35C123548F463DBA0670E5C659B278400C985CE88F59F26B41B37664743E5`,
identical to both the pre-flash and post-flash backups in
`C:\esp\board_backup_20260822\`.
