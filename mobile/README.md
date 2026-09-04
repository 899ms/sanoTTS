# sanoTTS on iOS, Android and Flutter

A ~300 KB neural voice that runs on the phone. No network, no API key, no
per-character billing.

One C API — [`sanotts.h`](sanotts.h) — with three thin bindings over it:

| Platform | Binding | Build |
|---|---|---|
| iOS / macOS | [`swift/SanoTTS.swift`](swift/SanoTTS.swift) | [`build_ios.sh`](build_ios.sh) → `SanoTTS.xcframework` |
| Android | [`kotlin/SanoTTS.kt`](kotlin/SanoTTS.kt) + [`kotlin/sanotts_jni.c`](kotlin/sanotts_jni.c) | [`CMakeLists.txt`](CMakeLists.txt) → `libsanotts.so` |
| Flutter / Dart | [`dart/sanotts.dart`](dart/sanotts.dart) | the same `.xcframework` / `.so` |

## Why a second API

`mcu/include/snt_nano.h` is an embedded interface: the caller owns an arena,
supplies both weight blobs, a noise seed and per-phoneme durations, and takes
PCM through a streaming callback. That is right on a microcontroller and
miserable to bind from Swift, Kotlin or Dart.

`sanotts.h` keeps the same runtime and changes nothing about the maths. It
owns the arena, buffers the waveform, grows the arena on demand for long
utterances, and exposes an opaque handle plus plain scalars — the shape FFI
consumes without glue.

```c
sanotts *tts = sanotts_open("front_q8.bin", "model_q8.bin");
float *pcm; int n;
sanotts_speak(tts, ids, n_ids, &pcm, &n);   /* 24 kHz mono float */
sanotts_free_pcm(pcm);
sanotts_close(tts);
```

## Read this before you plan the app

**It takes phoneme ids, not text.** There is no grapheme-to-phoneme in this
library. The browser demo phonemizes with espeak-ng compiled to WebAssembly
(`web/snt_g2p.wasm`); on a phone you have three options:

1. **Build espeak-ng for the platform.** It is plain C and cross-compiles for
   arm64 on both OSes. This is the honest path to arbitrary text and the one
   real piece of work in a full app.
2. **Precompute** phoneme ids at build time, if the app speaks a fixed set of
   phrases. Zero runtime cost, no dependency.
3. **Phonemize server-side** and send ids down. Defeats most of the point of
   on-device synthesis.

**The licence is GPL-3.0**, inherited from piper and espeak-ng. GPLv3 and the
App Store have a long-standing conflict — Apple's terms impose installation
and usage restrictions the FSF reads as violating GPL §6 and §12, and apps
have been removed over exactly this. Settle this before building, not after.
Dropping on-device espeak-ng removes half the copyleft; the rest depends on
what your runtime and weights derive from.

## Performance is not a problem

Measured, same runtime, portable scalar C:

| Target | RTF |
|---|---:|
| Apple arm64 (host) | ~0.003 — roughly 380× real time |
| Cortex-M7 @ 480 MHz | 0.35 |
| ESP32-S3 @ 240 MHz, SIMD | 0.41 |

If a Cortex-M7 clears real time by 3×, a phone is not going to struggle. Do
not write NEON kernels; you would be optimising something that costs about
3 ms per second of audio.

Working memory is one allocation, `46.5 KB + ~196 B per frame` — about 98 KB
for a 3 s utterance. `sanotts_arena_bytes()` reports the current size.

## iOS

```bash
./mobile/build_ios.sh            # -> build/ios/SanoTTS.xcframework
```

Add the xcframework and `swift/SanoTTS.swift` to your target, ship
`front_q8.bin` and `model_q8.bin` as bundle resources, then:

```swift
let tts = try SanoTTS(
    frontURL: Bundle.main.url(forResource: "front_q8", withExtension: "bin")!,
    decoderURL: Bundle.main.url(forResource: "model_q8", withExtension: "bin")!)
let buffer = try tts.buffer(phonemeIDs: ids)   // AVAudioPCMBuffer, 24 kHz

let engine = AVAudioEngine(), player = AVAudioPlayerNode()
engine.attach(player)
engine.connect(player, to: engine.mainMixerNode, format: buffer.format)
try engine.start()
player.scheduleBuffer(buffer); player.play()
```

*Verified: device and simulator arm64 slices build against the iOS 18.2 SDK
and link to a 52 KB static library. Not yet run on a physical device.*

## Android

```gradle
android {
    externalNativeBuild { cmake { path "../../mobile/CMakeLists.txt" } }
}
```

Assets are compressed in the APK and native code cannot open them directly,
so copy the two `.bin` files to `filesDir` on first run and pass those paths.

```kotlin
val tts = SanoTTS.open("$filesDir/front_q8.bin", "$filesDir/model_q8.bin")
val pcm = tts.synthesizePcm16(ids)
AudioTrack.Builder()
    .setAudioFormat(AudioFormat.Builder()
        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
        .setSampleRate(tts.sampleRate)
        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
    .setBufferSizeInBytes(pcm.size * 2)
    .build().apply { play(); write(pcm, 0, pcm.size) }
tts.close()
```

*Not verified: no Android NDK on the machine this was written on. The CMake
target does configure and build a shared library for the host, so the source
set and include paths are right, but nobody has built it for an Android ABI.
If you do, please open an issue either way.*

## Flutter

Bundle the xcframework (iOS) or let the CMake target build `libsanotts.so`
(Android), then:

```dart
final tts = SanoTts.open(frontPath: front, decoderPath: decoder);
final pcm = tts.synthesize(ids);      // Float32List, 24 kHz mono
tts.close();
```

`dart/sanotts.dart` imports nothing from Flutter, so it also runs in plain
Dart and in tests. It needs `package:ffi`.

*Not verified: no Flutter toolchain on the machine this was written on.*

## Which weights

Both voices in the [`voices-v2`](https://github.com/Ampixa/sanoTTS/releases/tag/voices-v2)
release run on this API, and both are 24 kHz:

| Voice | Params | Weights | Size |
|---|---:|---|---:|
| `heart-nano` | 294,279 | int8 | 316 KB |
| `heart` | 2,272,145 | f32 | 8.1 MB |

`heart` needs `-DSNT_NANO_W_F32`, since the header makes an int8/f32 mismatch
a compile error rather than a silent misread.

The lineage also picks the operators — LayerNorm vs DyT, GELU vs ReLU — from
`mcu/models/<lineage>/nano_q8_meta.h`, so set `SANOTTS_MODEL` to match the
weights you ship. Mixing them produces confident nonsense, not an error.

## Threading

A handle is not thread-safe. Give each thread its own, or serialise. Prefer
reusing one handle: it holds the arena and both weight blobs, so a second
handle costs another copy of everything.
