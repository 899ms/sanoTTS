// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ampixa
//
// The sanoTTS inference runtime is MIT; see LICENSE.MIT for the exact file
// list and why the split is sound. The repository as a whole is GPL-3.0,
// because the grapheme-to-phoneme layer embeds espeak-ng. This file does not.
package com.ampixa.sanotts

/**
 * Kotlin wrapper over the C API in sanotts.h, through the JNI shims in
 * mobile/kotlin/sanotts_jni.c.
 *
 * Closeable rather than finalized: the handle owns an arena and both weight
 * blobs, which is real memory, and Android gives no promises about when a
 * finalizer runs.
 */
class SanoTTS private constructor(private var handle: Long) : AutoCloseable {

    companion object {
        init { System.loadLibrary("sanotts") }

        /**
         * Weights normally ship as assets. Android cannot hand a file
         * descriptor for a compressed asset to native code, so copy them to
         * filesDir once at first run and pass those paths.
         */
        @JvmStatic
        fun open(frontPath: String, decoderPath: String): SanoTTS {
            val h = nativeOpen(frontPath, decoderPath)
            if (h == 0L) throw IllegalStateException("could not allocate handle")
            val err = nativeLastError(h)
            if (err != "ok") { nativeClose(h); throw IllegalStateException(err) }
            return SanoTTS(h)
        }

        // No @JvmStatic: these belong to the companion object, so their JNI
        // names are Java_com_ampixa_sanotts_SanoTTS_00024Companion_<method>
        // ($ escaped as _00024). @JvmStatic additionally emits a static
        // bridge on SanoTTS, which does not change the native symbol but does
        // invite the reader to expect Java_..._SanoTTS_<method>. The shims in
        // sanotts_jni.c export exactly the names above; that was checked
        // against the built .so with llvm-nm, because a mismatch compiles
        // cleanly and only fails at runtime.
        private external fun nativeOpen(front: String, dec: String): Long
        private external fun nativeClose(handle: Long)
        private external fun nativeLastError(handle: Long): String
        private external fun nativeSpeak(handle: Long, ids: IntArray): FloatArray?
        private external fun nativeSampleRate(handle: Long): Int
        private external fun nativeSetSeed(handle: Long, seed: Long)
    }

    val sampleRate: Int get() = nativeSampleRate(handle)

    fun setSeed(seed: Long) = nativeSetSeed(handle, seed)

    /** Phoneme ids, not text -- this library does no G2P. */
    fun synthesize(phonemeIds: IntArray): FloatArray =
        nativeSpeak(handle, phonemeIds)
            ?: throw IllegalStateException(nativeLastError(handle))

    /** Convert to the 16-bit PCM an AudioTrack wants. */
    fun synthesizePcm16(phonemeIds: IntArray): ShortArray {
        val f = synthesize(phonemeIds)
        return ShortArray(f.size) { i ->
            (f[i].coerceIn(-1f, 1f) * 32767f).toInt().toShort()
        }
    }

    override fun close() {
        if (handle != 0L) { nativeClose(handle); handle = 0L }
    }
}
