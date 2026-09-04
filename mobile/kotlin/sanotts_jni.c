/* sanotts_jni.c -- JNI shims for mobile/kotlin/SanoTTS.kt.
 *
 * Deliberately thin: every one of these maps to exactly one call in
 * sanotts.h. The handle crosses as a jlong, which is the ordinary way to
 * carry an opaque native pointer through Kotlin without a wrapper object per
 * call.
 */
#include <jni.h>
#include <stdlib.h>

#include "sanotts.h"

#define NS Java_com_ampixa_sanotts_SanoTTS_00024Companion

JNIEXPORT jlong JNICALL NS_nativeOpen(JNIEnv *env, jobject self,
                                      jstring front, jstring dec) {
    (void)self;
    const char *f = (*env)->GetStringUTFChars(env, front, NULL);
    const char *d = (*env)->GetStringUTFChars(env, dec, NULL);
    sanotts *t = sanotts_open(f, d);
    (*env)->ReleaseStringUTFChars(env, front, f);
    (*env)->ReleaseStringUTFChars(env, dec, d);
    return (jlong)(intptr_t)t;
}

JNIEXPORT void JNICALL NS_nativeClose(JNIEnv *env, jobject self, jlong h) {
    (void)env; (void)self;
    sanotts_close((sanotts *)(intptr_t)h);
}

JNIEXPORT jstring JNICALL NS_nativeLastError(JNIEnv *env, jobject self, jlong h) {
    (void)self;
    return (*env)->NewStringUTF(env, sanotts_last_error((sanotts *)(intptr_t)h));
}

JNIEXPORT jint JNICALL NS_nativeSampleRate(JNIEnv *env, jobject self, jlong h) {
    (void)env; (void)self;
    return (jint)sanotts_sample_rate((sanotts *)(intptr_t)h);
}

JNIEXPORT void JNICALL NS_nativeSetSeed(JNIEnv *env, jobject self, jlong h, jlong seed) {
    (void)env; (void)self;
    sanotts_set_seed((sanotts *)(intptr_t)h, (uint64_t)seed);
}

/* Returns null on failure; Kotlin then reads nativeLastError. Returning null
 * rather than throwing keeps the error text in one place. */
JNIEXPORT jfloatArray JNICALL NS_nativeSpeak(JNIEnv *env, jobject self,
                                             jlong h, jintArray ids) {
    (void)self;
    sanotts *t = (sanotts *)(intptr_t)h;
    const jsize n = (*env)->GetArrayLength(env, ids);
    jint *raw = (*env)->GetIntArrayElements(env, ids, NULL);

    float *pcm = NULL;
    int n_out = 0;
    /* jint is int32_t on every Android ABI, so this cast is a formality. */
    const int rc = sanotts_speak(t, (const int32_t *)raw, (int)n, &pcm, &n_out);
    (*env)->ReleaseIntArrayElements(env, ids, raw, JNI_ABORT);
    if (rc != 0 || !pcm) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, n_out);
    if (out) (*env)->SetFloatArrayRegion(env, out, 0, n_out, pcm);
    sanotts_free_pcm(pcm);
    return out;
}
