/* nano_glue_bench.c -- what the nano's float glue actually costs.
 *
 * mcu/INT16_CHAIN.md records the finding that decided R7's Tier-S work: on the
 * ESP32-C3 it was float glue, not multiply-accumulates, that dominated the
 * time (soft-float ~200 cycles/op). The nano adds glue R7 does not pay --
 * per second of audio at 93.75 frames/s:
 *
 *     exp        511 per frame  =  47,906 /s   (bins 1..511; DC and Nyquist zeroed)
 *   sincos       513 per frame  =  48,094 /s   (one call, two table probes)
 *   LayerNorm      6 per frame  =     562 /s   (stem + 4 blocks + final, 48 ch)
 *
 * so if the glue is a meaningful share on a host with hardware float, it will
 * be a much larger share on a part without one, and any projection that
 * scales the whole runtime by one factor is optimistic.
 *
 * Method: time each primitive in isolation over inputs drawn from the ranges
 * the real pipeline produces, then multiply by the per-second call counts.
 * This does NOT perturb the shipped loop (unlike an ablation build, where the
 * compiler can hoist a stubbed call and flatter the result). The stage timer
 * in the PROF build measures the same work end to end; the two are reported
 * side by side and should agree.
 *
 * The accumulator is printed so nothing can be optimised away.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"
#include "snt_port.h"

#define N_EXP 2000000
#define N_SC 2000000
#define N_LN 300000
#define DIMC 48

/* Same policy as snt_nano.c so the numbers describe the shipped build. */
#ifdef SNT_NANO_FAST_MATH
static float g_exp2_lut[65];
static int g_exp2_init = 0;
static inline float bench_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    if (!g_exp2_init) {
        for (int i = 0; i <= 64; i++) g_exp2_lut[i] = powf(2.0f, i / 64.0f);
        g_exp2_init = 1;
    }
    float y = x * 1.44269504088896f;
    float fi = floorf(y);
    float f = y - fi;
    int idx = (int)(f * 64.0f);
    float pf = g_exp2_lut[idx] + (g_exp2_lut[idx + 1] - g_exp2_lut[idx]) * (f * 64.0f - idx);
    union { float fv; int iv; } u;
    u.iv = (int)((fi + 127.0f) * 8388608.0f);
    return u.fv * pf;
}
static inline float bench_rsqrt(float x) {
    union { float fv; int iv; } u;
    u.fv = x;
    u.iv = 0x5f3759df - (u.iv >> 1);
    float r = u.fv;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    return r;
}
#define GLUE_LABEL "SNT_NANO_FAST_MATH"
#else
static inline float bench_exp(float x) { return expf(x); }
static inline float bench_rsqrt(float x) { return 1.0f / sqrtf(x); }
#define GLUE_LABEL "libm (exact)"
#endif

static void bench_layernorm(float *dst, const float *src, const float *g,
                            const float *b, int C) {
    float sum = 0.0f;
    for (int c = 0; c < C; c++) sum += src[c];
    float mean = sum * (1.0f / (float)C);
    float var = 0.0f;
    for (int c = 0; c < C; c++) { float d = src[c] - mean; var += d * d; }
    var *= (1.0f / (float)C);
    float inv = bench_rsqrt(var + 1e-6f);
    for (int c = 0; c < C; c++) dst[c] = (src[c] - mean) * inv * g[c] + b[c];
}

int main(void) {
    printf("# nano float-glue microbenchmark   math path: %s\n", GLUE_LABEL);

    /* inputs from the ranges the real pipeline produces:
     *   magnitude logits land in roughly [-12, 4.6] (4.6 = ln of the mag clip)
     *   phase spans the measured |phi| <= 30.33 */
    static float xs[4096], ps[4096];
    for (int i = 0; i < 4096; i++) {
        xs[i] = -12.0f + 16.6f * (float)i / 4095.0f;
        ps[i] = -30.33f + 60.66f * (float)i / 4095.0f;
    }
    static float ln_in[DIMC], ln_g[DIMC], ln_b[DIMC], ln_out[DIMC];
    for (int c = 0; c < DIMC; c++) {
        ln_in[c] = 0.7f * sinf(0.31f * c) + 0.2f;
        ln_g[c] = 1.0f + 0.01f * c;
        ln_b[c] = 0.001f * c;
    }

    double acc = 0.0;
    int64_t t0, t1;

    /* warm the tables and the caches */
    for (int i = 0; i < 100000; i++) acc += bench_exp(xs[i & 4095]);
    { float c, s; for (int i = 0; i < 100000; i++) { snt_nano_sincos(ps[i & 4095], &c, &s); acc += c + s; } }
    for (int i = 0; i < 20000; i++) { bench_layernorm(ln_out, ln_in, ln_g, ln_b, DIMC); acc += ln_out[0]; }

    /* SERIALIZED: every call feeds one accumulator, so latency cannot be
     * hidden. This is the pessimistic bound and the better model for an
     * in-order scalar core (ESP32-C3 class). */
    t0 = snt_now_us();
    for (int i = 0; i < N_EXP; i++) acc += bench_exp(xs[i & 4095]);
    t1 = snt_now_us();
    double ns_exp_ser = (double)(t1 - t0) * 1000.0 / N_EXP;

    t0 = snt_now_us();
    { float c, s; for (int i = 0; i < N_SC; i++) { snt_nano_sincos(ps[i & 4095], &c, &s); acc += c + s; } }
    t1 = snt_now_us();
    double ns_sc_ser = (double)(t1 - t0) * 1000.0 / N_SC;

    /* PIPELINED: independent results written to an array, as the real spec
     * loop does (each bin writes its own fre[k]/fim[k]). An out-of-order host
     * overlaps these; an in-order core cannot. */
    static float out_a[4096], out_b[4096];
    t0 = snt_now_us();
    for (int i = 0; i < N_EXP; i++) out_a[i & 4095] = bench_exp(xs[i & 4095]);
    t1 = snt_now_us();
    double ns_exp_pipe = (double)(t1 - t0) * 1000.0 / N_EXP;

    t0 = snt_now_us();
    for (int i = 0; i < N_SC; i++)
        snt_nano_sincos(ps[i & 4095], &out_a[i & 4095], &out_b[i & 4095]);
    t1 = snt_now_us();
    double ns_sc_pipe = (double)(t1 - t0) * 1000.0 / N_SC;
    for (int i = 0; i < 4096; i++) acc += out_a[i] + out_b[i];

    double ns_exp = ns_exp_pipe, ns_sc = ns_sc_pipe;

    t0 = snt_now_us();
    for (int i = 0; i < N_LN; i++) {
        ln_in[i & (DIMC - 1)] += 1e-7f;
        bench_layernorm(ln_out, ln_in, ln_g, ln_b, DIMC);
        acc += ln_out[0];
    }
    t1 = snt_now_us();
    double ns_ln = (double)(t1 - t0) * 1000.0 / N_LN;

    /* per second of generated audio at 93.75 mel frames/s */
    const double fps = 93.75;
    double calls_exp = 511.0 * fps;
    /* ONE snt_nano_sincos call yields both cos and sin, so the per-second
     * call count is 513*fps, not 2*513*fps. The 96k figure in the brief counts
     * cos and sin separately -- it is the number of TABLE PROBES, and there
     * are two per call. Getting this wrong doubles the apparent glue cost. */
    double calls_sc = 513.0 * fps;
    double calls_ln = 6.0 * fps;
    double us_exp = ns_exp * calls_exp / 1000.0;
    double us_sc = ns_sc * calls_sc / 1000.0;
    double us_ln = ns_ln * calls_ln / 1000.0;

    printf("%-12s %10s %10s %14s %13s %13s\n", "primitive", "ns/call",
           "ns/call", "calls/s-audio", "us/s-audio", "us/s-audio");
    printf("%-12s %10s %10s %14s %13s %13s\n", "", "(pipe)", "(serial)", "",
           "(pipe)", "(serial)");
    printf("%-12s %10.2f %10.2f %14.0f %13.1f %13.1f\n", "exp", ns_exp_pipe,
           ns_exp_ser, calls_exp, us_exp, ns_exp_ser * calls_exp / 1000.0);
    printf("%-12s %10.2f %10.2f %14.0f %13.1f %13.1f\n", "sincos", ns_sc_pipe,
           ns_sc_ser, calls_sc, us_sc, ns_sc_ser * calls_sc / 1000.0);
    printf("%-12s %10.2f %10.2f %14.0f %13.1f %13.1f\n", "layernorm48", ns_ln,
           ns_ln, calls_ln, us_ln, us_ln);
    printf("%-12s %10s %10s %14s %13.1f %13.1f\n", "GLUE TOTAL", "", "", "",
           us_exp + us_sc + us_ln,
           (ns_exp_ser * calls_exp + ns_sc_ser * calls_sc) / 1000.0 + us_ln);
    printf("# checksum %.6g (printed so nothing folds away)\n", acc);
    return 0;
}
