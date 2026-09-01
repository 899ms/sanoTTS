/* nano_bench_main.c -- host-C runtime timing for the E12-nano stack.
 *
 * A HOST NUMBER IS NOT A DEVICE NUMBER. This program exists to produce one half
 * of a ratio: the same protocol is run against the R7 runtime
 * (test/r7_bench_main.c) on the same machine in the same session, and R7 is the
 * only stack in this repo with BOTH a host-C figure and measured silicon
 * figures. The ratio, applied to R7's measured device numbers, is a better
 * grounded projection than scaling by predicted MMACs. It is still a
 * projection.
 *
 * Protocol: N repetitions per row, the first --warmup discarded, median of the
 * rest reported per row, then the medians summed over rows. Median rather than
 * mean because a scheduler preemption produces a one-sided outlier.
 *
 * Timing is snt_nano_stats.elapsed_us, i.e. the synthesize call only -- fixture
 * loading and correlation accumulation are outside it. The PCM callback is a
 * counter, matching what the R7 bench does, so neither stack is credited with
 * work the other pays for.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"

#define MAX_ROWS 32
#define MAX_REPS 129
#define NANO_SR 24000.0            /* the nano renders at 24 kHz */

#ifdef SNT_NANO_PROF
void snt_nano_prof_reset(void);
void snt_nano_prof_report(int frames, double audio_seconds);
#endif

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

typedef struct { char row_id[64]; int tokens, frames, samples; unsigned long long seed; } RowMeta;

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "test/fixtures/en_us_e12nano";
    int reps = argc > 2 ? atoi(argv[2]) : 21;
    int warmup = argc > 3 ? atoi(argv[3]) : 3;
    if (reps < 1 || reps > MAX_REPS) { fprintf(stderr, "reps 1..%d\n", MAX_REPS); return 1; }
    if (warmup < 0 || warmup >= reps) { fprintf(stderr, "warmup < reps\n"); return 1; }

    void *front = xload(dir, "front_q8.bin", NULL);
    void *dec = xload(dir, "model_q8.bin", NULL);
    char path[512];
    snprintf(path, sizeof path, "%s/rows.txt", dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "missing %s\n", path); return 1; }
    RowMeta rows[MAX_ROWS];
    int n_rows = 0;
    while (n_rows < MAX_ROWS &&
           fscanf(mf, "%63s %d %d %d %llu", rows[n_rows].row_id, &rows[n_rows].tokens,
                  &rows[n_rows].frames, &rows[n_rows].samples, &rows[n_rows].seed) == 5)
        n_rows++;
    fclose(mf);
    if (!n_rows) return 1;

    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    double total_med_us = 0.0, total_audio_s = 0.0;
    double worst_rtf = 0.0, best_rtf = 1e30;
    int total_frames = 0;

    printf("# E12-nano host-C timing   reps=%d (first %d discarded)  fixture=%s\n",
           reps, warmup, dir);
    printf("%-14s %6s %9s %10s %10s %10s %9s\n",
           "row", "frames", "audio_s", "med_us", "min_us", "max_us", "xRT");
    for (int r = 0; r < n_rows; r++) {
        char name[64];
        size_t nb;
        snprintf(name, sizeof name, "r%02d_ids.bin", r);
        int32_t *ids = (int32_t *)xload(dir, name, &nb);
        int n_ids = (int)(nb / 4);
        snprintf(name, sizeof name, "r%02d_durs.bin", r);
        int32_t *durs = (int32_t *)xload(dir, name, NULL);

        snt_nano_config cfg;
        cfg.front_blob = front;
        cfg.dec_blob = dec;
        cfg.arena = arena;
        cfg.arena_size = sizeof arena;
        cfg.dur_override = durs;
        cfg.noise_seed = (uint64_t)rows[r].seed;

        long long samples_us[MAX_REPS];
        snt_nano_stats st;
        int kept = 0;
        for (int i = 0; i < reps; i++) {
            memset(&st, 0, sizeof st);
#ifdef SNT_NANO_PROF
            if (i == reps - 1) snt_nano_prof_reset();
#endif
            if (snt_nano_synthesize(&cfg, ids, n_ids, count_cb, NULL, &st) != 0) {
                fprintf(stderr, "%s: synthesize failed\n", rows[r].row_id);
                return 1;
            }
            if (i >= warmup) samples_us[kept++] = st.elapsed_us;
        }
        qsort(samples_us, (size_t)kept, sizeof(long long), cmp_ll);
        double med = (kept % 2) ? (double)samples_us[kept / 2]
                                : 0.5 * (samples_us[kept / 2 - 1] + samples_us[kept / 2]);
        double audio_s = (double)st.samples / NANO_SR;
        double rtf = med / 1e6 / audio_s;
        if (rtf > worst_rtf) worst_rtf = rtf;
        if (rtf < best_rtf) best_rtf = rtf;
        printf("%-14s %6d %9.4f %10.0f %10lld %10lld %9.5f\n", rows[r].row_id,
               st.frames, audio_s, med, samples_us[0], samples_us[kept - 1], rtf);
        total_med_us += med;
        total_audio_s += audio_s;
        total_frames += st.frames;
#ifdef SNT_NANO_PROF
        if (r == 0) {
            printf("\n# stage breakdown, row %s, PROF build (timers add a few %%;\n"
                   "# read the SHARES here, take absolute xRT from the non-PROF build)\n",
                   rows[r].row_id);
            snt_nano_prof_report(st.frames, audio_s);
            printf("\n");
        }
#endif
        free(ids);
        free(durs);
    }
    printf("\n# aggregate over %d rows: %.4f s of audio, %.1f ms of compute\n",
           n_rows, total_audio_s, total_med_us / 1000.0);
    printf("# NANO host-C xRT = %.5f   (per-row spread %.5f .. %.5f)\n",
           total_med_us / 1e6 / total_audio_s, best_rtf, worst_rtf);
    printf("# frames %d, %.2f ms per second of audio\n", total_frames,
           total_med_us / 1000.0 / total_audio_s);
    (void)g_sink;
    return 0;
}
