#!/usr/bin/env python3
"""wav_from_serial.py -- capture BoardBenchmark's base64 WAV into a file.

A benchmark that prints only numbers gives you no way to hear whether the
thing actually works.  Press 'w' in the serial monitor (or let this script
send it) and the board streams the synthesized utterance out as a base64
RIFF/WAV; this saves it as a file you can play.

No DAC, no I2S, no SD card, no wiring -- the audio comes down the same USB
cable you flashed over.

Usage:
  python3 extras/wav_from_serial.py /dev/ttyUSB0 out.wav
  python3 extras/wav_from_serial.py COM5 out.wav --baud 115200

Needs pyserial (`pip install pyserial`).
"""
from __future__ import annotations

import argparse
import base64
import sys
import time


BEGIN = "---- WAV BEGIN (base64) ----"
END = "---- WAV END ----"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("port", help="serial port, e.g. /dev/ttyUSB0 or COM5")
    parser.add_argument("out", help="output .wav path")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="give up after this many seconds")
    parser.add_argument("--no-trigger", action="store_true",
                        help="do not send 'w'; wait for a dump already in flight")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import serial  # noqa: PLC0415
    except ImportError:
        print("pyserial is required:  pip install pyserial", file=sys.stderr)
        return 2

    with serial.Serial(args.port, args.baud, timeout=1) as port:
        if not args.no_trigger:
            # The board runs its benchmark at boot; 'w' asks for the audio.
            time.sleep(0.3)
            port.write(b"w\n")
            port.flush()

        started = time.time()
        collecting = False
        chunks: list[str] = []
        pending = b""
        while time.time() - started < args.timeout:
            pending += port.read(4096)
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode("utf-8", "replace").strip()
                if line == BEGIN:
                    collecting, chunks = True, []
                    print("receiving...", file=sys.stderr)
                elif line == END:
                    if not chunks:
                        print("empty payload", file=sys.stderr)
                        return 1
                    data = base64.b64decode("".join(chunks))
                    if len(data) < 44 or data[:4] != b"RIFF":
                        print(f"not a RIFF file ({len(data)} bytes)", file=sys.stderr)
                        return 1
                    with open(args.out, "wb") as handle:
                        handle.write(data)
                    seconds = (len(data) - 44) / 2 / 22050
                    print(f"wrote {args.out}  {len(data)} bytes, {seconds:.2f} s")
                    return 0
                elif collecting and line:
                    chunks.append(line)
                elif not collecting:
                    print(line, file=sys.stderr)

    print("timed out waiting for the WAV; is the sketch running?", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
