# ESP32-C3 flash backup: INCOMPLETE — do not flash this board

Attempted 2026-08-22 on COM7 after the user re-plugged the C3.
**No firmware was written to the C3.** The backup did not finish, so the
precondition for flashing was never met.

## Chip, confirmed by esptool before any write

    ESP32-C3 (QFN32) revision v0.4
    Wi-Fi, BT 5 (LE), Single Core, 160MHz, Embedded Flash 4MB (XMC)
    Crystal 40MHz   MAC ec:da:3b:18:6a:30
    Flash manufacturer 0x20 device 0x4016, detected size 4MB

## Partition table, read from 0x8000 (3072 B) and decoded

| label    | type | subtype | offset    | size      | end       |
|----------|------|---------|-----------|-----------|-----------|
| nvs      | data | 0x2     | 0x9000    | 24576     | 0xf000    |
| phy_init | data | 0x1     | 0xf000    | 4096      | 0x10000   |
| factory  | app  | 0x0     | 0x10000   | 3145728   | 0x310000  |

There is no SPIFFS on this board — unlike the S3, which carries the espeak
data partition at 0x390000. Only the two regions above exist, so only those
were dumped.

## What was captured

| artifact | bytes | sha256 |
|---|---|---|
| `c3_boot.bin` (0x0–0x10000, bootloader + parttable + nvs + phy_init) | 65,536 | `31D4339D055055F8F52123EA259099A8B5B5A660120893DCB01294E17480E872` |
| `c3_parttable.bin` (0x8000, standalone copy) | 3,072 | `73C0B5C3E5FCBA3A151CC70C453C93DD5F4798899E7F2F8CCA76DA1F32FFC501` |

App region chunks captured, 5 of 12 (1,310,720 B of 3,145,728 B, 41.7%):

| chunk | bytes | sha256 |
|---|---|---|
| `app_00010000.bin` | 262,144 | `3DA8DE8A0E12A46AD86DAD8E1D56A10C3393AAEFEB21CD3CF9574AAA4A1BD7BB` |
| `app_00050000.bin` | 262,144 | `AE6D97ED1B66FFD68325F85AE1CF339CF8A006E534BF157346F542F1AA9AE7AD` |
| `app_00090000.bin` | 262,144 | `1A7B1A09666FFF93509F6D54C80FB91AB34D68214D094CF9AA51D90FD0A5AB18` |
| `app_000d0000.bin` | 262,144 | `D5CA2E6F3D83D5AEA937E956542A948D749963822986A04C19AD27E803CE56B0` |
| `app_00110000.bin` | 262,144 | `A9DFC664D147F274C462E44D47A75A16A1C87B859568DB367BC84D4716BFFF09` |

Missing: 0x150000, 0x190000, 0x1d0000, 0x210000, 0x250000, 0x290000, 0x2d0000.

## How it failed

The read of the chunk at 0x150000 died mid-transfer and every retry after it
failed to open the port:

    A fatal error occurred: Could not open COM7, the port is busy or doesn't exist.
    (Cannot configure port, something went wrong. Original message:
     PermissionError(13, 'A device attached to the system is not functioning.', None, 31))

Five retries in the script, then four further full passes of the script (20
more attempts) all failed at the same address. `Get-PnpDevice` reported COM7
as `Status OK` throughout the first failures, which is the stale-entry
behaviour this board has shown before — the entry exists, the device does not
answer. A disable/enable cycle on the device instance did not recover it, and
COM7 then disappeared from the port list entirely (`FileNotFoundError`).

Final state: all three CH340 entries (COM3, COM6, COM7) read `Unknown`; only
the S3's CH343 on COM5 is `OK`. Polled for a further ~100 s with no recovery.

**The C3 needs another physical re-plug.** The dump script
(`C:\esp\dump_c3.ps1`) is resumable: it skips any chunk already present at the
correct length, so re-running it after a re-plug will continue from 0x150000
rather than starting over.

## Ready to go when the board returns

The firmware is built and waiting; only the backup blocks it.

- `mcu/ports/esp32c3/nano_app/` — C3 harness, single core, no PSRAM, no
  1-core/2-core pair, `DEVICE:`/`HOST-REF:` labelling, r00 and r06 both on the
  internal-SRAM arena, per-stage breakdown with float glue isolated.
- `mcu/ports/esp32c3/snt_port_esp32c3.c` — operand residency counters added.
  On RV32IMC these separate SRAM-resident operands from flash-XIP ones; they
  are NOT a SIMD dispatch rate, because there is no vector unit and both
  branches run the same scalar loop. The harness label says so explicitly.
- Built on the host: `C:\esp\build_nano_c3.bat`, EXITCODE 0,
  `nano_c3.bin` 1,612,224 B, fits the 3,145,728 B factory partition (49% free).
