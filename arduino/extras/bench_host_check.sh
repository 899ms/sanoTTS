#!/usr/bin/env bash
# extras/bench_host_check.sh -- compile and run examples/BoardBenchmark's
# ACTUAL sketch source on this workstation, and fail unless it reports PASS.
#
# Why this exists: the only thing standing between a stranger with a Teensy
# and a wasted evening is whether this sketch works on the first flash. We
# cannot flash every board, but we can prove the sketch's own logic -- arena
# allocation, the golden correlation gate, the RMS gate, the REPORT block --
# against the same fixture mcu/test/golden_main.c uses. A shim supplies the
# handful of Arduino symbols the sketch touches; everything else is the real
# thing, including the generated data header.
#
# Reference (mcu/test/golden_main.c, fixture en_us_r7):
#   frames 134, samples 34304, corr 0.989148, rms_ratio 0.935022
#
# Usage: ./extras/bench_host_check.sh
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
lib="$here/.."
src="$lib/src"
sketch="$lib/examples/BoardBenchmark"
CC=${CC:-cc}
CXX=${CXX:-c++}

if [ ! -f "$sketch/sanotts_bench_data.h" ]; then
  echo "missing $sketch/sanotts_bench_data.h"
  echo "generate it first, from the arduino/ directory:"
  echo "  python3 extras/gen_board_benchmark_data.py \\"
  echo "    --fixture ../mcu/test/fixtures/en_us_r7 \\"
  echo "    --out examples/BoardBenchmark/sanotts_bench_data.h"
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

CFLAGS="-O2 -std=c99 -Wall -I$src -DFSD_FAST_MATH"
CXXFLAGS="-O2 -std=c++11 -Wall -I$src -I$here -DFSD_FAST_MATH"
# Which arena rungs to exercise. The runtime keeps opportunistic weight-
# residency buffers in whatever arena space is left over, so a board with less
# RAM takes a different code path -- and that path has to produce the same
# audio, or a small-RAM reporter's numbers mean nothing. 90112 is just above
# the 88 KB floor; 88064 is the floor itself.
CAPS=${CAPS:-"0 327680 196608 131072 98304 90112"}

echo "== compiling the runtime (C) =="
for f in snt_tts snt_kernels_ref snt_port_default; do
  "$CC" $CFLAGS -c "$src/$f.c" -o "$tmp/$f.o"
done

for cap in $CAPS; do
  if [ "$cap" = "0" ]; then
    label="uncapped"; capflag=""
  else
    label="cap ${cap} B"; capflag="-DSANOTTS_BENCH_MAX_ARENA=${cap}"
  fi
  echo
  echo "== BoardBenchmark.ino, ${label} =="
  "$CXX" $CXXFLAGS $capflag -x c++ -c "$here/bench_host_main.cpp" -o "$tmp/bench_main.o"
  "$CXX" "$tmp"/snt_tts.o "$tmp"/snt_kernels_ref.o "$tmp"/snt_port_default.o \
    "$tmp"/bench_main.o -lm -o "$tmp/bench_host"
  out="$tmp/out.txt"
  "$tmp/bench_host" > "$out"
  grep -E "arena_bytes|frames:|samples:|golden_corr|rms_ratio|verdict" "$out" | sed 's/^/  /'

  grep -q "verdict:      PASS" "$out"  || { echo "FAIL (${label}): sketch did not report PASS"; exit 1; }
  grep -q "frames:       134" "$out"   || { echo "FAIL (${label}): expected 134 frames"; exit 1; }
  grep -q "samples:      34304" "$out" || { echo "FAIL (${label}): expected 34304 samples"; exit 1; }
  grep -q "golden_corr:  0.9891" "$out" || { echo "FAIL (${label}): correlation drifted from 0.989148"; exit 1; }
  grep -q "rms_ratio:    0.935" "$out" || { echo "FAIL (${label}): rms_ratio drifted from 0.935022"; exit 1; }
done
echo
echo "OK: the shipped sketch reproduces golden_main.c at every arena rung"
