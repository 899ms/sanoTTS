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
# Expectations follow the stack the header holds. R7: 134 frames / 34,304
# samples / corr 0.989148. nano row 0: 415 frames / 105,984 samples, and the
# correlation is over the embedded 34,304-sample prefix.
if grep -q "define SANOTTS_BENCH_NANO" "$sketch/sanotts_bench_data.h"; then
  EXPECT_FRAMES=${EXPECT_FRAMES:-415}
  EXPECT_SAMPLES=${EXPECT_SAMPLES:-105984}
  EXPECT_CORR=${EXPECT_CORR:-0.98}
  CAPS=${CAPS:-"0 327680 196608 147456 139264"}
else
  EXPECT_FRAMES=${EXPECT_FRAMES:-134}
  EXPECT_SAMPLES=${EXPECT_SAMPLES:-34304}
  EXPECT_CORR=${EXPECT_CORR:-0.9891}
  CAPS=${CAPS:-"0 327680 196608 131072 98304 90112"}
fi

# Which stack the generated header holds decides which runtime to link.
if grep -q "define SANOTTS_BENCH_NANO" "$sketch/sanotts_bench_data.h"; then
  RUNTIME="snt_nano snt_kernels_ref snt_port_default"
  echo "== nano stack (snt_nano.c) =="
else
  RUNTIME="snt_tts snt_kernels_ref snt_port_default"
  echo "== R7 stack (snt_tts.c) =="
fi
echo "== compiling the runtime (C) =="
for f in $RUNTIME; do
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
  OBJS=""
  for f in $RUNTIME; do OBJS="$OBJS $tmp/$f.o"; done
  "$CXX" $OBJS "$tmp"/bench_main.o -lm -o "$tmp/bench_host"
  out="$tmp/out.txt"
  "$tmp/bench_host" > "$out"
  grep -E "model:|arena_bytes|arena_peak|frames:|samples:|golden_corr|rms_ratio|verdict" "$out" | sed 's/^/  /'

  grep -q "verdict:      PASS" "$out" || { echo "FAIL (${label}): sketch did not report PASS"; exit 1; }
  grep -q "frames:       $EXPECT_FRAMES" "$out"   || { echo "FAIL (${label}): expected $EXPECT_FRAMES frames"; exit 1; }
  grep -q "samples:      $EXPECT_SAMPLES" "$out" || { echo "FAIL (${label}): expected $EXPECT_SAMPLES samples"; exit 1; }
  # Numeric, not a string prefix: the gate is "at least this correlated",
  # and a prefix match rejects a result that is BETTER than expected.
  got=$(sed -n 's/^golden_corr:  *//p' "$out")
  awk -v g="$got" -v w="$EXPECT_CORR" 'BEGIN{exit !(g+0 >= w+0)}' || {
    echo "FAIL (${label}): correlation $got below $EXPECT_CORR"; exit 1; }
  rms=$(sed -n 's/^rms_ratio:  *//p' "$out")
  awk -v r="$rms" 'BEGIN{exit !(r+0 > 0.80 && r+0 < 1.25)}' || {
    echo "FAIL (${label}): rms_ratio $rms outside 0.80-1.25"; exit 1; }
done
echo
echo "OK: the shipped sketch reproduces golden_main.c at every arena rung"
