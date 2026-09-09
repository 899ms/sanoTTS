#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ampixa
# build_ios.sh -- produce SanoTTS.xcframework (device + simulator, arm64).
#
# Drop the result into an Xcode project or an SPM binaryTarget, add
# mobile/swift/SanoTTS.swift, and you have a voice.
#
# Usage:  ./mobile/build_ios.sh [out-dir]     (run from the repo root)
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/build/ios}"
model="${SANOTTS_MODEL:-$root/mcu/models/en_us_e12nano}"

SDK=$(xcrun --sdk iphoneos --show-sdk-path)
SIM=$(xcrun --sdk iphonesimulator --show-sdk-path)
INC="-I$root/mobile -I$root/mcu/include -I$root/mcu/src -I$model -DFSD_FAST_MATH"
SRCS="$root/mobile/sanotts.c $root/mcu/src/snt_nano.c $root/mcu/src/snt_kernels_ref.c $root/mcu/ports/host/snt_port_host.c"

rm -rf "$out"; mkdir -p "$out/device" "$out/sim" "$out/include"
for f in $SRCS; do
  xcrun clang -target arm64-apple-ios15.0 -isysroot "$SDK" -O2 -std=c99 -fembed-bitcode-marker \
    $INC -c "$f" -o "$out/device/$(basename "$f" .c).o"
  xcrun clang -target arm64-apple-ios15.0-simulator -isysroot "$SIM" -O2 -std=c99 \
    $INC -c "$f" -o "$out/sim/$(basename "$f" .c).o"
done
xcrun libtool -static -o "$out/libsanotts-ios.a" "$out"/device/*.o
xcrun libtool -static -o "$out/libsanotts-sim.a" "$out"/sim/*.o
cp "$root/mobile/sanotts.h" "$out/include/"
rm -rf "$out/SanoTTS.xcframework"
xcodebuild -create-xcframework \
  -library "$out/libsanotts-ios.a" -headers "$out/include" \
  -library "$out/libsanotts-sim.a" -headers "$out/include" \
  -output "$out/SanoTTS.xcframework" >/dev/null
echo "built $out/SanoTTS.xcframework"
ls "$out/SanoTTS.xcframework"
