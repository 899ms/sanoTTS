/* r7_bench_main.c -- host-C runtime timing for the shipped R7 stack.
 *
 * The calibration anchor. R7 is the only stack here with both a host-C figure
 * and measured silicon figures (ESP32-S3 0.22x RT, ESP32-C3 5.7x RT, July
 * 2026), so timing it on the same machine, in the same session, under the same
 * background load as test/nano_bench_main.c turns the nano's host number into a
 * device projection with a measured anchor instead of a predicted MMAC ratio.
 *
 * Identical protocol to the nano bench: N repetitions, first --warmup
 * discarded, median of the rest. Timing is snt_stats.elapsed_us (the
 * synthesize call only). The PCM callback is the same counter, so neither
 * stack is credited with work the other pays for.
 *
 * R7 renders at 22.05 kHz; the nano at 24 kHz. Both are divided by their own
 * sample rate, so the xRT figures are directly comparable even though R7 runs
 * 86.13 mel frames per second of audio against the nano's 93.75.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_tts.h"

#define MAX_REPS 129
#define R7_SR 22050.0

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) exit(1);
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static long g_sink;
static int count_cb(const float *pcm, int n, void *user) {
    (void)user;
    for (int i = 0; i < n; i++) g_sink += (long)(pcm[i] != 0.0f);
    return 0;
}

static int cmp_ll(const void *a, const void *b) {
    long long x = *(const long long *)a, y = *(const long long *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "test/fixtures/en_us_r7";
    int reps = argc > 2 ? atoi(argv[2]) : 21;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    if (reps < 1 || reps > MAX_REPS) { fprintf(stderr, "reps 1..%d\n", MAX_REPS); return 1; }
    if (warmup < 0 || warmup >= reps) { fprintf(stderr, "warmup < reps\n"); return 1; }

    size_t nb;
    void *front = xload(dir, "front_q8.bin", NULL);
    void *dec = xload(dir, "model_q8.bin", NULL);
    int32_t *ids = (int32_t *)xload(dir, "e2e_ids.bin", &nb);
    int n_ids = (int)(nb / 4);
    int32_t *durs = (int32_t *)xload(dir, "e2e_durs.bin", NULL);

    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    snt_config cfg = {front, dec, arena, sizeof arena, durs};
    snt_stats st;
    long long samples_us[MAX_REPS];
    int kept = 0;

    printf("# R7 host-C timing   reps=%d (first %d discarded)  fixture=%s\n",
           reps, warmup, dir);
    for (int i = 0; i < reps; i++) {
        memset(&st, 0, sizeof st);
        if (snt_synthesize(&cfg, ids, n_ids, count_cb, NULL, &st) != 0) {
            fprintf(stderr, "synthesize failed\n");
            return 1;
        }
        if (i >= warmup) samples_us[kept++] = st.elapsed_us;
    }
    qsort(samples_us, (size_t)kept, sizeof(long long), cmp_ll);
    double med = (kept % 2) ? (double)samples_us[kept / 2]
                            : 0.5 * (samples_us[kept / 2 - 1] + samples_us[kept / 2]);
    double audio_s = (double)st.samples / R7_SR;
    printf("%-14s %6s %9s %10s %10s %10s %9s\n",
           "row", "frames", "audio_s", "med_us", "min_us", "max_us", "xRT");
    printf("%-14s %6d %9.4f %10.0f %10lld %10lld %9.5f\n", "en_us_r7", st.frames,
           audio_s, med, samples_us[0], samples_us[kept - 1], med / 1e6 / audio_s);
    printf("\n# R7 host-C xRT = %.5f\n", med / 1e6 / audio_s);
    printf("# frames %d, %.2f ms per second of audio\n", st.frames,
           med / 1000.0 / audio_s);
    (void)g_sink;
    return 0;
}
