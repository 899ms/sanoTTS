/* front_q8_golden_test.c -- int8 gate for the piperlite front half.
 *
 * For each golden dir (tools/export_front_golden.py then
 * tools/export_front_q8.py into the SAME dir): load front_meta_q8.bin +
 * front_weights_q8.bin, run the int8 duration + latent path on ids.bin and
 * gate against the fp32 PyTorch goldens already in the dir:
 *
 *   - durations: compared token by token against durations.bin at length
 *     scale 1.0 (and durations_ls125.bin at 1.25). The fp32 port gates these
 *     EXACT, because it runs the same arithmetic; int8 does not, and the
 *     honest gate is a tolerance. The duration head ends in exp() then
 *     round-half-to-even -- a hard decision boundary -- so quantisation noise
 *     there does not degrade, it flips a whole frame. This gate allows a
 *     drift of at most DUR_TOL frames on any token and GATE_FRAME_DRIFT of
 *     the total frame count, and prints the exact-match count either way.
 *     (Measured on amy: int8 weights + a 12-bit activation lane reproduce
 *     157/157 tokens at ls 1.0 and miss 3 at ls 1.25, each by one frame.)
 *   - latent: Pearson corr vs latent.bin, computed on the GOLDEN durations so
 *     the number isolates the acoustic student's quantisation error from any
 *     duration drift -- otherwise a one-frame shift would dominate a
 *     correlation that is supposed to be measuring quantisation. Gate
 *     GATE_CORR.
 *
 * Per-stage goldens (dur_log.bin, tok_ctx.bin, ...) are diffed with -s.
 *
 *   ./front_q8_golden_test [-s] golden_front/amy golden_front/hindi
 * Exit 0 iff every dir passes both gates.
 * (The harness mallocs; the runtime's hot path does not.)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_front_q8.h"

#ifndef GATE_CORR
#define GATE_CORR 0.99
#endif
#ifndef DUR_TOL
#define DUR_TOL 1                /* frames, per token */
#endif
#ifndef GATE_FRAME_DRIFT
#define GATE_FRAME_DRIFT 0.01    /* of the fp32 total */
#endif

static void *xload(const char *dir, const char *name, size_t *bytes,
                   int required) {
    char path[512];
    FILE *fh;
    long sz;
    void *buf;
    snprintf(path, sizeof path, "%s/%s", dir, name);
    fh = fopen(path, "rb");
    if (!fh) {
        if (required) { fprintf(stderr, "missing %s\n", path); exit(1); }
        if (bytes) *bytes = 0;
        return NULL;
    }
    fseek(fh, 0, SEEK_END);
    sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    buf = malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) {
        fprintf(stderr, "short read %s\n", path);
        exit(1);
    }
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static void diff_stats(const float *a, const float *b, long n,
                       double *corr, double *maxabs) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mad = 0;
    long i;
    for (i = 0; i < n; i++) {
        double x = a[i], y = b[i], d = fabs(x - y);
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
        if (d > mad) mad = d;
    }
    *corr = (sab - sa * sb / (double)n) /
            sqrt((saa - sa * sa / (double)n) * (sbb - sb * sb / (double)n) +
                 1e-30);
    *maxabs = mad;
}

typedef struct { const char *dir; } StageCtx;

static void stage_tap(const char *name, const float *data, int ch, int len,
                      void *user) {
    StageCtx *ctx = (StageCtx *)user;
    char fname[128];
    size_t nb = 0;
    float *gold;
    long n = (long)ch * len;
    double corr, mad;
    snprintf(fname, sizeof fname, "%s.bin", name);
    gold = (float *)xload(ctx->dir, fname, &nb, 0);
    if (!gold) return;
    if ((long)(nb / 4) != n) {
        printf("  stage %-14s SIZE MISMATCH: C %ld vs golden %zu floats\n",
               name, n, nb / 4);
        free(gold);
        return;
    }
    diff_stats(data, gold, n, &corr, &mad);
    printf("  stage %-14s corr %.9f  max|diff| %.3e  (%d x %d)\n",
           name, corr, mad, ch, len);
    free(gold);
}

/* Returns 0 if the drift is inside tolerance, 1 otherwise. */
static int check_durations(const char *label, const int32_t *pred,
                           const int32_t *gold, int n) {
    long mism = 0, i, first = -1, worst = 0, sp = 0, sg = 0;
    double drift;
    int ok;
    for (i = 0; i < n; i++) {
        long d = (long)pred[i] - (long)gold[i];
        sp += pred[i];
        sg += gold[i];
        if (d) {
            if (first < 0) first = i;
            mism++;
            if (d < 0) d = -d;
            if (d > worst) worst = d;
        }
    }
    drift = sg ? (double)(sp - sg) / (double)sg : 0.0;
    ok = worst <= DUR_TOL && fabs(drift) <= GATE_FRAME_DRIFT;
    printf("durations %-8s %ld/%d tokens exact, max|drift| %ld frame(s), "
           "total %ld vs %ld (%+.3f%%)  %s\n",
           label, n - mism, n, worst, sp, sg, 100.0 * drift,
           ok ? "PASS" : "FAIL");
    if (mism)
        printf("            first difference at token %ld: int8 %d vs fp32 %d\n",
               first, pred[first], gold[first]);
    return ok ? 0 : 1;
}

static int run_dir(const char *dir, int verbose_stages) {
    size_t meta_nb, w_nb, ids_nb, dur_nb, dur125_nb, lat_nb;
    void *meta = xload(dir, "front_meta_q8.bin", &meta_nb, 1);
    int8_t *weights = (int8_t *)xload(dir, "front_weights_q8.bin", &w_nb, 1);
    int32_t *ids = (int32_t *)xload(dir, "ids.bin", &ids_nb, 1);
    int32_t *gold_dur = (int32_t *)xload(dir, "durations.bin", &dur_nb, 1);
    int32_t *gold_dur125 =
        (int32_t *)xload(dir, "durations_ls125.bin", &dur125_nb, 0);
    float *gold_lat = (float *)xload(dir, "latent.bin", &lat_nb, 1);
    snt_front_q8_model m;
    StageCtx ctx;
    int32_t *dur, *dur125;
    float *latent;
    void *arena;
    size_t arena_n;
    int n_tokens, rc, failed = 0;
    long frames, gold_frames, i;
    double corr, mad;

    printf("== %s\n", dir);
    rc = snt_front_q8_init(&m, meta, meta_nb, weights, w_nb);
    if (rc != 0) {
        fprintf(stderr, "snt_front_q8_init failed: %d\n", rc);
        return 1;
    }
    n_tokens = (int)(ids_nb / 4);
    if (n_tokens <= 0 || dur_nb != ids_nb) {
        fprintf(stderr, "ids.bin/durations.bin token count mismatch\n");
        return 1;
    }
    if ((lat_nb / 4) % (size_t)m.a_out != 0) {
        fprintf(stderr, "latent.bin size %zu not divisible by C %d\n",
                lat_nb / 4, m.a_out);
        return 1;
    }
    gold_frames = (long)(lat_nb / 4 / (size_t)m.a_out);
    for (i = 0, frames = 0; i < n_tokens; i++) frames += gold_dur[i];
    if (frames != gold_frames) {
        fprintf(stderr, "durations.bin sums to %ld, latent.bin has %ld frames\n",
                frames, gold_frames);
        return 1;
    }

    /* Attach the tap before any arena is sized: the tap needs scratch. */
    if (verbose_stages) {
        ctx.dir = dir;
        m.stage_cb = stage_tap;
        m.stage_user = &ctx;
    }
    dur = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    dur125 = (int32_t *)malloc((size_t)n_tokens * sizeof(int32_t));
    arena_n = snt_front_q8_duration_arena_bytes(&m, n_tokens);
    arena = malloc(arena_n);
    if (!dur || !dur125 || !arena) { fprintf(stderr, "oom\n"); exit(1); }

    if (snt_front_q8_durations(&m, ids, n_tokens, 1.0f, dur, arena,
                               arena_n) <= 0) {
        fprintf(stderr, "snt_front_q8_durations failed\n");
        return 1;
    }
    failed += check_durations("ls=1.0", dur, gold_dur, n_tokens);
    if (gold_dur125) {
        if (snt_front_q8_durations(&m, ids, n_tokens, 1.25f, dur125, arena,
                                   arena_n) <= 0 || dur125_nb != ids_nb) {
            fprintf(stderr, "length_scale 1.25 run failed\n");
            return 1;
        }
        failed += check_durations("ls=1.25", dur125, gold_dur125, n_tokens);
    }
    free(arena);

    /* Latent on the GOLDEN durations: this isolates acoustic quantisation
     * error from any duration drift, so the number compares like for like. */
    arena_n = snt_front_q8_latent_arena_bytes(&m, n_tokens, gold_frames);
    arena = malloc(arena_n);
    latent = (float *)malloc((size_t)m.a_out * (size_t)gold_frames * sizeof(float));
    if (!arena || !latent) { fprintf(stderr, "oom\n"); exit(1); }
    rc = snt_front_q8_latent(&m, ids, gold_dur, n_tokens, gold_frames, latent,
                             arena, arena_n);
    if (rc != 0) {
        fprintf(stderr, "snt_front_q8_latent failed: %d\n", rc);
        return 1;
    }
    diff_stats(latent, gold_lat, (long)m.a_out * gold_frames, &corr, &mad);
    printf("latent corr %.9f  max|diff| %.3e  (%d ch x %ld frames)  %s\n",
           corr, mad, m.a_out, gold_frames, corr > GATE_CORR ? "PASS" : "FAIL");
    if (!(corr > GATE_CORR)) failed++;
    {   /* size, against the fp32 export sitting in the same dir */
        size_t f32_nb = 0, f32_meta_nb = 0;
        void *a = xload(dir, "front_weights_f32.bin", &f32_nb, 0);
        void *b = xload(dir, "meta.bin", &f32_meta_nb, 0);
        free(a); free(b);
        if (f32_nb)
            printf("size int8 %zu B (weights %zu + meta %zu)  vs fp32 %zu B  "
                   "= %.2fx smaller\n",
                   w_nb + meta_nb, w_nb, meta_nb, f32_nb + f32_meta_nb,
                   (double)(f32_nb + f32_meta_nb) / (double)(w_nb + meta_nb));
        else
            printf("size int8 %zu B (weights %zu + meta %zu)\n",
                   w_nb + meta_nb, w_nb, meta_nb);
    }

    free(meta); free(weights); free(ids); free(gold_dur); free(gold_dur125);
    free(gold_lat); free(dur); free(dur125); free(arena); free(latent);
    return failed ? 1 : 0;
}

int main(int argc, char **argv) {
    int i, failures = 0, ran = 0;
    int verbose = 0, argstart = 1;
    if (argc > 1 && strcmp(argv[1], "-s") == 0) { verbose = 1; argstart = 2; }
    if (argc <= argstart) {
        fprintf(stderr, "usage: %s [-s] <golden-dir> [more dirs...]\n"
                        "  -s  also diff per-stage goldens\n", argv[0]);
        return 2;
    }
    for (i = argstart; i < argc; i++) {
        failures += run_dir(argv[i], verbose);
        ran++;
    }
    printf("%d/%d dirs passed (gates: duration drift <= %d frame/token and "
           "%.1f%% of total, latent corr > %.3f)\n",
           ran - failures, ran, DUR_TOL, 100.0 * GATE_FRAME_DRIFT,
           (double)GATE_CORR);
    return failures ? 1 : 0;
}
