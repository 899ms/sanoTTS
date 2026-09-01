/* nano_dev_main.c -- on-silicon gate for the E12-nano (294,642 params).
 *
 * Device analogue of mcu/test/nano_golden_main.c. Same fixture, same fixed
 * sha256(row_id) seed, same 0.98 correlation gate. Nothing here is tuned to
 * make the board pass; a failure printed honestly is the point.
 *
 * Measurement protocol (the one the R7 line was held to):
 *   1 discarded warm-up, then 5 timed runs, median reported, spread shown.
 * Timed runs use a counting-only PCM callback so the number is synthesis cost
 * and not the cost of whatever a consumer does with the samples. Correlation
 * is computed in a separate, untimed run -- the graph is deterministic under a
 * fixed seed, so it is the same audio.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_private/esp_clk.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "snt_nano.h"

extern void snt_port_esp32s3_start_worker(void);
extern void snt_nano_prof_set(int on);
extern void snt_nano_prof_reset(void);
extern void snt_nano_prof_report(int frames, double audio_seconds);
extern int snt_weights_resident(const void *p);

extern const uint8_t front_start[] asm("_binary_front_q8_bin_start");
extern const uint8_t model_start[] asm("_binary_model_q8_bin_start");
extern const uint8_t ids_start[] asm("_binary_r00_ids_bin_start");
extern const uint8_t ids_end[] asm("_binary_r00_ids_bin_end");
extern const uint8_t durs_start[] asm("_binary_r00_durs_bin_start");
extern const uint8_t audio_start[] asm("_binary_r00_audio_bin_start");
extern const uint8_t audio_end[] asm("_binary_r00_audio_bin_end");
extern const uint8_t uni_start[] asm("_binary_e2e_uniform_bin_start");
extern const uint8_t uni_end[] asm("_binary_e2e_uniform_bin_end");
extern const uint8_t noi_start[] asm("_binary_e2e_noise_bin_start");
extern const uint8_t noi_end[] asm("_binary_e2e_noise_bin_end");
extern const uint8_t ids6_start[] asm("_binary_r06_ids_bin_start");
extern const uint8_t ids6_end[] asm("_binary_r06_ids_bin_end");
extern const uint8_t durs6_start[] asm("_binary_r06_durs_bin_start");
extern const uint8_t audio6_start[] asm("_binary_r06_audio_bin_start");
extern const uint8_t audio6_end[] asm("_binary_r06_audio_bin_end");

extern int64_t g_mv_macs_simd, g_mv_macs_scalar;
extern int64_t g_mv_calls_simd, g_mv_calls_scalar;
extern void snt_port_res_reset(void);

/* rows.txt line 1 of the fixture: id, tokens, frames, samples, seed */
#define ROW_ID      "000001_i200"
#define ROW_TOKENS  73
#define ROW_FRAMES  415
#define ROW_SAMPLES 105984
#define ROW_SEED    2236265385529901705ULL
#define SAMPLE_RATE 24000.0
#define GATE        0.98
#define NOISE_TOL   1e-5
#define SINCOS_TOL  1e-4
#define N_TIMED     5
/* rows.txt line 7: the row with the WORST host correlation (0.984762), i.e.
 * the row the host gate's MIN is actually taken from. At 278,816 B of arena it
 * used to be larger than the biggest contiguous internal-SRAM block on this
 * board and could only be measured PSRAM-backed. It is measured BOTH ways now:
 * whether it fits internal SRAM is the whole test of the memory work. */
#define ROW6_ID      "000007_i1400"
#define ROW6_FRAMES  629
#define ROW6_SAMPLES 160768
#define ROW6_SEED    14351038112024617452ULL

/* Newlib-nano's printf drops %f. Rather than depend on the sdkconfig knob,
 * every number that matters is formatted from integers. */
static const char *f6(double v, char *buf) {
    int neg = v < 0;
    if (neg) v = -v;
    long long scaled = (long long)(v * 1000000.0 + 0.5);
    sprintf(buf, "%s%lld.%06lld", neg ? "-" : "", scaled / 1000000,
            scaled % 1000000);
    return buf;
}
static const char *f3(double v, char *buf) {
    int neg = v < 0;
    if (neg) v = -v;
    long long scaled = (long long)(v * 1000.0 + 0.5);
    sprintf(buf, "%s%lld.%03lld", neg ? "-" : "", scaled / 1000, scaled % 1000);
    return buf;
}

/* ---- PCM sinks --------------------------------------------------------- */

static long g_count;
static int count_cb(const float *pcm, int n, void *user) {
    (void)pcm; (void)user;
    g_count += n;
    return 0;
}

typedef struct {
    const float *gold;
    size_t n_gold, pos;
    double sa, sb, saa, sbb, sab;
} CorrSink;

static int corr_cb(const float *pcm, int n, void *user) {
    CorrSink *c = (CorrSink *)user;
    for (int i = 0; i < n && c->pos < c->n_gold; i++, c->pos++) {
        double a = pcm[i], b = c->gold[c->pos];
        c->sa += a; c->sb += b;
        c->saa += a * a; c->sbb += b * b; c->sab += a * b;
    }
    return 0;
}

/* ---- component checks -------------------------------------------------- */

static int check_components(void) {
    char b1[48];
    int bad = 0;

    uint64_t derived = 0;
    snt_nano_sha256_seed(ROW_ID, &derived);
    printf("DEVICE: seed    : sha256(\"%s\")[:8] = %llu, fixture says %llu %s\n",
           ROW_ID, (unsigned long long)derived, (unsigned long long)ROW_SEED,
           derived == ROW_SEED ? "OK" : "-- SHA-256 PORT IS WRONG");
    bad |= (derived != ROW_SEED);

    int nu = (int)((uni_end - uni_start) / sizeof(float));
    const float *uref = (const float *)uni_start;
    float *ugot = malloc((size_t)nu * sizeof(float));
    if (!ugot) return 1;
    snt_nano_uniform_stream(ROW_SEED, nu, ugot);
    int diff = 0;
    for (int i = 0; i < nu; i++) if (ugot[i] != uref[i]) diff++;
    printf("uniform : %d values, %d differ %s\n", nu, diff,
           diff == 0 ? "-- BIT-EXACT vs torch.rand" : "-- MT19937 PORT IS WRONG");
    bad |= (diff != 0);
    free(ugot);

    int nn = (int)((noi_end - noi_start) / sizeof(float));
    const float *nref = (const float *)noi_start;
    float *ngot = malloc((size_t)nn * sizeof(float));
    if (!ngot) return 1;
    if (snt_nano_seeded_noise(ROW_SEED, 4, nn / 4, ngot) != 0) {
        printf("noise   : generator refused %d values\n", nn);
        free(ngot);
        return 1;
    }
    diff = 0;
    double worst = 0.0;
    for (int i = 0; i < nn; i++) {
        double d = fabs((double)ngot[i] - (double)nref[i]);
        if (d > worst) worst = d;
        if (ngot[i] != nref[i]) diff++;
    }
    printf("noise   : %d values, %d differ, max |delta| %s (tol 1e-5) %s\n",
           nn, diff, f6(worst, b1), worst <= NOISE_TOL ? "OK" : "OUT OF BOUND");
    bad |= (worst > NOISE_TOL);
    free(ngot);

    /* 100k points, not the host harness's 400k: same phase range, a quarter of
     * the serial-log wait. Stated so the two numbers are not confused. */
    worst = 0.0;
    float worst_phi = 0.0f, phi_max = 32.0f;
    for (int i = 0; i <= 100000; i++) {
        float phi = -phi_max + 2.0f * phi_max * (float)i / 100000.0f;
        float c, s;
        snt_nano_sincos(phi, &c, &s);
        double dc = fabs((double)c - cos((double)phi));
        double ds = fabs((double)s - sin((double)phi));
        double d = dc > ds ? dc : ds;
        if (d > worst) { worst = d; worst_phi = phi; }
    }
    printf("sincos  : max |error| %s over |phi| <= 32 (worst at %s, tol 1e-4) %s\n",
           f6(worst, b1), f3((double)worst_phi, (char[24]){0}),
           worst <= SINCOS_TOL ? "OK" : "OUT OF BOUND");
    bad |= (worst > SINCOS_TOL);
    return bad;
}

/* ---- timing ------------------------------------------------------------ */

static int cmp_i64(const void *a, const void *b) {
    int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static void run_timed(snt_nano_config *cfg, const int32_t *ids, int n_ids,
                      const char *label, double audio_sec) {
    char b1[48], b2[48], b3[48];
    snt_nano_stats st;
    int64_t us[N_TIMED];

    g_count = 0;
    memset(&st, 0, sizeof st);
    int rc = snt_nano_synthesize(cfg, ids, n_ids, count_cb, NULL, &st);
    printf("DEVICE: [%s] warm-up: rc=%d frames=%d samples=%d %lld us (DISCARDED)\n",
           label, rc, st.frames, st.samples, (long long)st.elapsed_us);
    if (rc != 0) return;

    for (int i = 0; i < N_TIMED; i++) {
        g_count = 0;
        memset(&st, 0, sizeof st);
        rc = snt_nano_synthesize(cfg, ids, n_ids, count_cb, NULL, &st);
        us[i] = st.elapsed_us;
        printf("DEVICE: [%s] run %d: rc=%d frames=%d samples=%d arena_peak=%u  %lld us  RTF %s\n",
               label, i + 1, rc, st.frames, st.samples,
               (unsigned)st.arena_peak, (long long)st.elapsed_us,
               f6((double)st.elapsed_us / 1e6 / audio_sec, b1));
    }
    int64_t sorted[N_TIMED];
    memcpy(sorted, us, sizeof us);
    qsort(sorted, N_TIMED, sizeof(int64_t), cmp_i64);
    int64_t med = sorted[N_TIMED / 2];
    printf("DEVICE: [%s] MEDIAN %lld us  (min %lld, max %lld, spread %s%%)\n",
           label, (long long)med, (long long)sorted[0],
           (long long)sorted[N_TIMED - 1],
           f3(100.0 * (double)(sorted[N_TIMED - 1] - sorted[0]) / (double)med, b2));
    printf("DEVICE: [%s] audio %s s @ 24000 Hz  ==>  RTF = %s   (%s x real time)\n",
           label, f6(audio_sec, b1), f6((double)med / 1e6 / audio_sec, b2),
           f3(audio_sec / ((double)med / 1e6), b3));
}


/* ---- one (row, arena) configuration ------------------------------------ */

static double run_row(const char *tag, const int32_t *ids, int n_ids,
                      const int32_t *durs, const float *gold, size_t n_gold,
                      uint64_t seed, int exp_frames, int exp_samples,
                      void *arena, size_t arena_size, const char *where,
                      int do_profile) {
    char b1[48], b2[48];
    double audio_sec = (double)exp_samples / SAMPLE_RATE;

    printf("\n======== %s :: arena %u B in %s (resident=%d) ========\n",
           tag, (unsigned)arena_size, where, snt_weights_resident(arena));

    snt_nano_config cfg;
    cfg.front_blob = front_start;
    cfg.dec_blob = model_start;
    cfg.arena = arena;
    cfg.arena_size = arena_size;
    cfg.dur_override = durs;
    cfg.noise_seed = seed;

    snt_nano_prof_set(0);
    run_timed(&cfg, ids, n_ids, tag, audio_sec);

    CorrSink sink;
    memset(&sink, 0, sizeof sink);
    sink.gold = gold;
    sink.n_gold = n_gold;
    snt_nano_stats st;
    memset(&st, 0, sizeof st);
    snt_port_res_reset();
    int rc = snt_nano_synthesize(&cfg, ids, n_ids, corr_cb, &sink, &st);
    if (rc != 0 || sink.pos == 0) {
        /* An arena that cannot hold the row is a RESULT, not a setup problem:
         * say so instead of printing a correlation over zero samples. */
        printf("DEVICE: [%s] rc=%d, %u samples produced -- NO CORRELATION "
               "COMPUTED (rc=-2 is ERR_OOM: the row does not fit this arena)\n",
               tag, rc, (unsigned)sink.pos);
        return -2.0;
    }
    double n = (double)sink.pos;
    double cov = sink.sab - sink.sa * sink.sb / n;
    double cr = cov / sqrt((sink.saa - sink.sa * sink.sa / n) *
                           (sink.sbb - sink.sb * sink.sb / n) + 1e-30);
    printf("DEVICE: [%s] rc=%d frames=%d (fixture %d) samples=%d (fixture %d) arena_peak=%u\n",
           tag, rc, st.frames, exp_frames, st.samples, exp_samples,
           (unsigned)st.arena_peak);
    printf("DEVICE: [%s] CORRELATION = %s  rms_ratio = %s  ==> GATE %s (0.98)\n",
           tag, f6(cr, b1), f6(sqrt(sink.saa / n) / sqrt(sink.sbb / n), b2),
           cr > GATE ? "PASS" : "FAIL");

    int64_t tot = g_mv_macs_simd + g_mv_macs_scalar;
    printf("DEVICE: [%s] int8 matvec residency: SIMD %lld MACs / %lld calls (%s%%), "
           "SCALAR %lld MACs / %lld calls (%s%%)\n", tag,
           (long long)g_mv_macs_simd, (long long)g_mv_calls_simd,
           f3(tot ? 100.0 * (double)g_mv_macs_simd / (double)tot : 0.0, b1),
           (long long)g_mv_macs_scalar, (long long)g_mv_calls_scalar,
           f3(tot ? 100.0 * (double)g_mv_macs_scalar / (double)tot : 0.0, b2));

    if (do_profile) {
        printf("\n---- PER-STAGE BREAKDOWN, %s (profiling ON; totals run high, "
               "read the SHARES) ----\n", tag);
        snt_nano_prof_set(1);
        snt_nano_prof_reset();
        g_count = 0;
        memset(&st, 0, sizeof st);
        rc = snt_nano_synthesize(&cfg, ids, n_ids, count_cb, NULL, &st);
        printf("profiled run: rc=%d %lld us wall\n", rc, (long long)st.elapsed_us);
        snt_nano_prof_report(st.frames, audio_sec);
        snt_nano_prof_set(0);
    }
    return cr;
}

/* ---- main -------------------------------------------------------------- */

void app_main(void) {
    char b1[48], b2[48];
    vTaskDelay(pdMS_TO_TICKS(1500));   /* let the monitor attach */

    printf("\n\n================ E12-nano on ESP32-S3 ================\n");
    printf("RAW SERIAL CAPTURE. Lines marked DEVICE: are measured on this\n"
           "chip in this run. Lines marked HOST-REF: are constants compiled\n"
           "into the binary and are NOT device measurements.\n");
    esp_chip_info_t ci;
    esp_chip_info(&ci);
    printf("chip    : %d core(s), rev %d, %d MHz\n", ci.cores, ci.revision,
           (int)(esp_clk_cpu_freq() / 1000000));
    printf("psram   : %u bytes\n", (unsigned)esp_psram_get_size());
    printf("heap    : internal free %u, largest internal block %u, "
           "spiram free %u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("blobs   : front @ %p, model @ %p (flash XIP, resident=%d/%d)\n",
           front_start, model_start, snt_weights_resident(front_start),
           snt_weights_resident(model_start));

    /* Arena: internal SRAM if it fits, PSRAM only as a stated fallback.
     * This is NOT a cosmetic choice -- snt_weights_resident() only reports
     * true for internal SRAM, so a PSRAM arena silently disables every PIE
     * SIMD kernel (the core stages weights INTO the arena). An SRAM-resident
     * result and a PSRAM-backed result are different claims. */
    size_t want = 320 * 1024;
    size_t largest = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (want > largest - 16384) want = largest - 16384;
    void *arena = heap_caps_aligned_alloc(16, want, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    const char *arena_where = "INTERNAL SRAM";
    if (!arena) {
        want = 320 * 1024;
        arena = heap_caps_aligned_alloc(16, want, MALLOC_CAP_SPIRAM);
        arena_where = "PSRAM (SIMD DISABLED -- see snt_weights_resident)";
    }
    if (!arena) { printf("FATAL: no arena\n"); return; }
    printf("arena   : %u bytes @ %p in %s (resident=%d)\n", (unsigned)want,
           arena, arena_where, snt_weights_resident(arena));
    printf("======================================================\n\n");

    int bad_components = check_components();

    const int32_t *ids = (const int32_t *)ids_start;
    int n_ids = (int)((ids_end - ids_start) / 4);
    const int32_t *durs = (const int32_t *)durs_start;
    const float *gold = (const float *)audio_start;
    size_t n_gold = (size_t)(audio_end - audio_start) / 4;
    printf("\nfixture : row %s, %d ids (rows.txt says %d), %u golden samples "
           "(rows.txt says %d)\n", ROW_ID, n_ids, ROW_TOKENS,
           (unsigned)n_gold, ROW_SAMPLES);
    printf("seeding : cfg.noise_seed = %llu = sha256(\"%s\")[:8], IDENTICAL to "
           "the fixture\n", (unsigned long long)ROW_SEED, ROW_ID);

    /* Single vs dual core on the SRAM row first, then the arena comparisons. */
    double audio_sec0 = (double)ROW_SAMPLES / SAMPLE_RATE;
    snt_nano_config cfg;
    cfg.front_blob = front_start;
    cfg.dec_blob = model_start;
    cfg.arena = arena;
    cfg.arena_size = want;
    cfg.dur_override = durs;
    cfg.noise_seed = ROW_SEED;
    snt_nano_prof_set(0);
    printf("\n---- TIMING r00, SINGLE core, SRAM arena ----\n");
    run_timed(&cfg, ids, n_ids, "r00/SRAM/1core", audio_sec0);

    snt_port_esp32s3_start_worker();
    vTaskDelay(pdMS_TO_TICKS(50));

    double cr0 = run_row("r00/SRAM/2core", ids, n_ids, durs, gold, n_gold,
                         ROW_SEED, ROW_FRAMES, ROW_SAMPLES, arena, want,
                         arena_where, 1);

    /* Same row, PSRAM arena: holds the row fixed and isolates the PSRAM
     * penalty -- which, because snt_weights_resident() is false for PSRAM,
     * also prices what the PIE SIMD path is currently worth. */
    size_t psz = 320 * 1024;
    void *parena = heap_caps_aligned_alloc(16, psz, MALLOC_CAP_SPIRAM);
    double cr0p = -2.0, cr6 = -2.0, cr6p = -2.0;
    if (parena) {
        cr0p = run_row("r00/PSRAM", ids, n_ids, durs, gold, n_gold, ROW_SEED,
                       ROW_FRAMES, ROW_SAMPLES, parena, psz,
                       "PSRAM (resident false -> SIMD off)", 0);
        const int32_t *ids6 = (const int32_t *)ids6_start;
        int n_ids6 = (int)((ids6_end - ids6_start) / 4);
        cr6p = run_row("r06/PSRAM", ids6, n_ids6, (const int32_t *)durs6_start,
                       (const float *)audio6_start,
                       (size_t)(audio6_end - audio6_start) / 4, ROW6_SEED,
                       ROW6_FRAMES, ROW6_SAMPLES, parena, psz,
                       "PSRAM (resident false -> SIMD off)", 0);
    } else {
        printf("\nPSRAM arena allocation FAILED -- PSRAM comparisons skipped\n");
    }

    /* r06 in INTERNAL SRAM. This is the row that did not fit before the memory
     * work; if the arena request below is honoured and the peak comes back
     * under it, the row is running fully SIMD like r00. */
    {
        const int32_t *ids6 = (const int32_t *)ids6_start;
        int n_ids6 = (int)((ids6_end - ids6_start) / 4);
        cr6 = run_row("r06/SRAM/2core", ids6, n_ids6, (const int32_t *)durs6_start,
                      (const float *)audio6_start,
                      (size_t)(audio6_end - audio6_start) / 4, ROW6_SEED,
                      ROW6_FRAMES, ROW6_SAMPLES, arena, want, arena_where, 1);
    }

    /* THE FORMATTING RULE THAT MATTERS. Every number computed on this chip
     * from this chip's own PCM is prefixed DEVICE:. Every number that is a
     * constant compiled into this binary from a host run is prefixed
     * HOST-REF:. They are never printed on the same line, because a reference
     * value in parentheses next to a measured one is exactly how a host
     * constant gets read back as a silicon result. */
    printf("\n---- SUMMARY: measured on this chip ----\n");
    printf("DEVICE: r00 SRAM  corr %s\n", f6(cr0, b1));
    printf("DEVICE: r00 PSRAM corr %s\n", f6(cr0p, b1));
    printf("DEVICE: r06 PSRAM corr %s\n", f6(cr6p, b1));
    printf("DEVICE: r06 SRAM  corr %s\n", f6(cr6, b1));
    printf("\n---- REFERENCE: constants compiled in from the host gate, "
           "NOT measured here ----\n");
    printf("HOST-REF: r00 host fastmath corr = 0.987887\n");
    printf("HOST-REF: r06 host fastmath corr = 0.984762  (minimum over the "
           "8-row host gate)\n");
    double cr = cr0 < cr6 ? cr0 : cr6;
    printf("\ncounts/s-audio at 93.75 fps: exp %s, sincos(2 trig ea) %s, "
           "LayerNorm %s\n",
           f3(513.0 * 93.75 - 2 * 93.75, b1), f3(513.0 * 93.75, b2),
           f3(6.0 * 93.75, (char[24]){0}));
    printf("\nDEVICE RESULT: components %s, correlation %s\n",
           bad_components ? "FAIL" : "OK", cr > GATE ? "PASS" : "FAIL");
    printf("================ END ================\n");

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
