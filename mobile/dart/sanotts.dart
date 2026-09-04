// Dart FFI binding for the C API in sanotts.h. Works in Flutter and in plain
// Dart; nothing here imports Flutter.
//
// The C layer was shaped for exactly this: an opaque handle plus scalars, so
// there is no marshalling beyond Utf8 for paths and one copy of the result.

import 'dart:ffi';
import 'dart:io' show Platform;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

typedef _OpenC = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _Open = Pointer<Void> Function(Pointer<Utf8>, Pointer<Utf8>);
typedef _SpeakC = Int32 Function(
    Pointer<Void>, Pointer<Int32>, Int32, Pointer<Pointer<Float>>, Pointer<Int32>);
typedef _Speak = int Function(
    Pointer<Void>, Pointer<Int32>, int, Pointer<Pointer<Float>>, Pointer<Int32>);
typedef _VoidHandleC = Void Function(Pointer<Void>);
typedef _VoidHandle = void Function(Pointer<Void>);
typedef _FreePcmC = Void Function(Pointer<Float>);
typedef _FreePcm = void Function(Pointer<Float>);
typedef _ErrC = Pointer<Utf8> Function(Pointer<Void>);
typedef _Err = Pointer<Utf8> Function(Pointer<Void>);
typedef _RateC = Int32 Function(Pointer<Void>);
typedef _Rate = int Function(Pointer<Void>);
typedef _SeedC = Void Function(Pointer<Void>, Uint64);
typedef _Seed = void Function(Pointer<Void>, int);

DynamicLibrary _open() {
  if (Platform.isIOS || Platform.isMacOS) return DynamicLibrary.process();
  if (Platform.isAndroid) return DynamicLibrary.open('libsanotts.so');
  if (Platform.isLinux) return DynamicLibrary.open('libsanotts.so');
  if (Platform.isWindows) return DynamicLibrary.open('sanotts.dll');
  throw UnsupportedError('sanoTTS: unsupported platform');
}

class SanoTtsException implements Exception {
  final String message;
  SanoTtsException(this.message);
  @override
  String toString() => 'SanoTtsException: $message';
}

class SanoTts {
  static final DynamicLibrary _lib = _open();
  static final _openFn = _lib.lookupFunction<_OpenC, _Open>('sanotts_open');
  static final _speakFn = _lib.lookupFunction<_SpeakC, _Speak>('sanotts_speak');
  static final _closeFn = _lib.lookupFunction<_VoidHandleC, _VoidHandle>('sanotts_close');
  static final _freeFn = _lib.lookupFunction<_FreePcmC, _FreePcm>('sanotts_free_pcm');
  static final _errFn = _lib.lookupFunction<_ErrC, _Err>('sanotts_last_error');
  static final _rateFn = _lib.lookupFunction<_RateC, _Rate>('sanotts_sample_rate');
  static final _seedFn = _lib.lookupFunction<_SeedC, _Seed>('sanotts_set_seed');

  Pointer<Void> _handle;
  SanoTts._(this._handle);

  /// Weights are usually copied out of assets to a writable directory first;
  /// pass those paths.
  factory SanoTts.open({required String frontPath, required String decoderPath}) {
    final f = frontPath.toNativeUtf8();
    final d = decoderPath.toNativeUtf8();
    try {
      final h = _openFn(f, d);
      if (h == nullptr) throw SanoTtsException('could not allocate handle');
      final err = _errFn(h).toDartString();
      if (err != 'ok') {
        _closeFn(h);
        throw SanoTtsException(err);
      }
      return SanoTts._(h);
    } finally {
      calloc.free(f);
      calloc.free(d);
    }
  }

  int get sampleRate => _rateFn(_handle);

  /// The decoder is noise-fed; the seed is fixed by default so a given input
  /// always renders identically.
  void setSeed(int seed) => _seedFn(_handle, seed);

  /// Phoneme ids, not text -- this package does no G2P. See mobile/README.md.
  Float32List synthesize(List<int> phonemeIds) {
    final ids = calloc<Int32>(phonemeIds.length);
    final outPcm = calloc<Pointer<Float>>();
    final outLen = calloc<Int32>();
    try {
      for (var i = 0; i < phonemeIds.length; i++) {
        ids[i] = phonemeIds[i];
      }
      final rc = _speakFn(_handle, ids, phonemeIds.length, outPcm, outLen);
      if (rc != 0 || outPcm.value == nullptr) {
        throw SanoTtsException(_errFn(_handle).toDartString());
      }
      final n = outLen.value;
      // Copy before freeing: the native buffer is ours only until free.
      final out = Float32List.fromList(outPcm.value.asTypedList(n));
      _freeFn(outPcm.value);
      return out;
    } finally {
      calloc.free(ids);
      calloc.free(outPcm);
      calloc.free(outLen);
    }
  }

  void close() {
    if (_handle != nullptr) {
      _closeFn(_handle);
      _handle = nullptr;
    }
  }
}
