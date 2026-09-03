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
 *
 * PART B (added 2026-09-02) answers two things the whole-utterance median
 * cannot answer:
 *
 *   A. What does an ASSOCIATED, ACTIVELY TRANSMITTING radio cost this
 *      workload? Every earlier number on this board was taken with the radio
 *      down, which is not the state a talking device is in. The WiFi driver is
 *      brought up BEFORE the control run and the PHY is started only after it,
 *      so the control and the loaded run differ in the radio and in nothing
 *      else -- same binary, same boot, same arena, same core count, and the
 *      driver's memory already allocated in both.
 *
 *   B. What does the WORST chunk cost? A mean RTF hides a drained buffer: if
 *      one chunk takes longer to generate than it takes to play, the listener
 *      hears a gap. snt_nano_synthesize emits PCM once per frame; the callback
 *      records elapsed microseconds and the sample count per chunk into RAM
 *      and the distribution is printed after the run. Nothing is printed from
 *      inside the timed path.
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_chip_info.h"
#include "esp_private/esp_clk.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_heap_caps.h"
#include "esp_netif.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs_flash.h"
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

/* Per-chunk sink (measurement B). One record per PCM callback: the wall time
 * that elapsed producing that chunk, and how many samples it carries. Both are
 * plain stores into a preallocated array -- no printf, no allocation, no
 * locking -- so the instrumentation cannot move the number it measures. The
 * mark is re-read AFTER the bookkeeping, so the two timer reads and the two
 * stores fall between chunks rather than inside one. */
#define MAX_CHUNKS 800
static int32_t g_chunk_us[MAX_CHUNKS];
static uint16_t g_chunk_n[MAX_CHUNKS];
static int g_chunk_used;
static int g_chunk_dropped;      /* callbacks past MAX_CHUNKS, if it ever happens */
static int64_t g_chunk_mark;

static int chunk_cb(const float *pcm, int n, void *user) {
    int64_t now = esp_timer_get_time();
    (void)pcm; (void)user;
    if (g_chunk_used < MAX_CHUNKS) {
        g_chunk_us[g_chunk_used] = (int32_t)(now - g_chunk_mark);
        g_chunk_n[g_chunk_used] = (uint16_t)n;
        g_chunk_used++;
    } else {
        g_chunk_dropped++;
    }
    g_count += n;
    g_chunk_mark = esp_timer_get_time();
    return 0;
}

typedef struct {
    int n;                    /* chunks recorded                              */
    double mean, p95, max;    /* per-chunk RTF over all chunks                */
    int over;                 /* chunks with RTF >= 1.0                       */
    double first;             /* RTF of chunk 1 (carries the whole front end) */
    double ss_mean, ss_p95, ss_max;   /* same, chunks 2..N                    */
    int ss_over;
    int worst_idx;
} ChunkStats;

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

/* Per-chunk RTF = elapsed_us / (n_samples / 24000 * 1e6). A chunk at RTF >= 1
 * took longer to make than it takes to play: that is the audible gap. p95 is
 * nearest-rank on the sorted RTFs, computed after the run from the buffered
 * records. Chunk 1 is reported apart because it carries the whole front end
 * (duration + acoustic model) before the first sample exists -- that is
 * start-up latency, not a drained buffer, and averaging it in would hide both. */
static void chunk_report(const char *label, ChunkStats *out) {
    char b1[48], b2[48], b3[48];
    memset(out, 0, sizeof *out);
    int n = g_chunk_used;
    if (n <= 0) {
        printf("DEVICE: [%s] per-chunk: NO CHUNKS RECORDED\n", label);
        return;
    }
    if (g_chunk_dropped)
        printf("DEVICE: [%s] per-chunk: WARNING %d callbacks past the %d-record "
               "buffer were not recorded\n", label, g_chunk_dropped, MAX_CHUNKS);

    int64_t *srt = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!srt) { printf("DEVICE: [%s] per-chunk: out of memory for the report\n", label); return; }

    double sum = 0.0, mx = 0.0;
    int over = 0, worst = 0;
    long tot_samples = 0;
    for (int i = 0; i < n; i++) {
        double rtf = (double)g_chunk_us[i] * SAMPLE_RATE / ((double)g_chunk_n[i] * 1e6);
        srt[i] = (int64_t)(rtf * 1e6 + 0.5);
        sum += rtf;
        if (rtf > mx) { mx = rtf; worst = i; }
        if (rtf >= 1.0) over++;
        tot_samples += g_chunk_n[i];
    }
    qsort(srt, (size_t)n, sizeof(int64_t), cmp_i64);
    int p95i = (95 * n + 99) / 100 - 1;      /* nearest rank */
    if (p95i < 0) p95i = 0;
    if (p95i >= n) p95i = n - 1;
    out->n = n;
    out->mean = sum / (double)n;
    out->p95 = (double)srt[p95i] / 1e6;
    out->max = mx;
    out->over = over;
    out->worst_idx = worst;
    out->first = (double)g_chunk_us[0] * SAMPLE_RATE / ((double)g_chunk_n[0] * 1e6);

    printf("DEVICE: [%s] per-chunk: %d chunks, %ld samples (%s s of audio)\n",
           label, n, tot_samples, f6((double)tot_samples / SAMPLE_RATE, b1));
    printf("DEVICE: [%s] per-chunk ALL     : mean RTF %s  p95 %s  max %s (chunk %d)  "
           "chunks with RTF>=1.0: %d\n", label, f6(out->mean, b1), f6(out->p95, b2),
           f6(out->max, b3), worst + 1, over);
    printf("DEVICE: [%s] per-chunk CHUNK 1 : %ld us over %d samples, RTF %s "
           "(this one chunk carries duration+acoustic for the whole utterance)\n",
           label, (long)g_chunk_us[0], (int)g_chunk_n[0], f6(out->first, b1));

    if (n > 1) {
        double s2 = 0.0, m2 = 0.0;
        int o2 = 0, w2 = 1;
        for (int i = 1; i < n; i++) {
            double rtf = (double)g_chunk_us[i] * SAMPLE_RATE / ((double)g_chunk_n[i] * 1e6);
            srt[i - 1] = (int64_t)(rtf * 1e6 + 0.5);
            s2 += rtf;
            if (rtf > m2) { m2 = rtf; w2 = i; }
            if (rtf >= 1.0) o2++;
        }
        int m = n - 1;
        qsort(srt, (size_t)m, sizeof(int64_t), cmp_i64);
        int q = (95 * m + 99) / 100 - 1;
        if (q < 0) q = 0;
        if (q >= m) q = m - 1;
        out->ss_mean = s2 / (double)m;
        out->ss_p95 = (double)srt[q] / 1e6;
        out->ss_max = m2;
        out->ss_over = o2;
        printf("DEVICE: [%s] per-chunk 2..N  : %d chunks, mean RTF %s  p95 %s  max %s "
               "(chunk %d)  chunks with RTF>=1.0: %d\n", label, m,
               f6(out->ss_mean, b1), f6(out->ss_p95, b2), f6(out->ss_max, b3), w2 + 1, o2);
    }

    /* the eight slowest chunks, so the tail is visible and not just summarised */
    printf("DEVICE: [%s] slowest chunks (idx, us, samples, RTF):", label);
    for (int k = 0; k < 8 && k < n; k++) {
        int bi = -1;
        double bv = -1.0;
        for (int i = 0; i < n; i++) {
            double rtf = (double)g_chunk_us[i] * SAMPLE_RATE / ((double)g_chunk_n[i] * 1e6);
            int seen = 0;
            for (int j = 0; j < k; j++) if (srt[j] == (int64_t)i) { seen = 1; break; }
            if (!seen && rtf > bv) { bv = rtf; bi = i; }
        }
        if (bi < 0) break;
        srt[k] = (int64_t)bi;
        printf("  (%d, %ld, %d, %s)", bi + 1, (long)g_chunk_us[bi],
               (int)g_chunk_n[bi], f6(bv, b1));
    }
    printf("\n");
    free(srt);
}

static int64_t run_timed(snt_nano_config *cfg, const int32_t *ids, int n_ids,
                         const char *label, double audio_sec) {
    char b1[48], b2[48], b3[48];
    snt_nano_stats st;
    int64_t us[N_TIMED];

    g_count = 0;
    memset(&st, 0, sizeof st);
    int rc = snt_nano_synthesize(cfg, ids, n_ids, count_cb, NULL, &st);
    printf("DEVICE: [%s] warm-up: rc=%d frames=%d samples=%d %lld us (DISCARDED)\n",
           label, rc, st.frames, st.samples, (long long)st.elapsed_us);
    if (rc != 0) return -1;

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
    return med;
}


/* ---- WiFi: association and a real transmit load (measurement A) --------- */
/* The AP is the same one the fsd_audio dashboard associates to on this board.
 * Credentials live in the firmware because this is a bench fixture on a lab
 * network; nothing here is shipped. */
#define WIFI_SSID       "testnet"
#define WIFI_PASS       "divideAndConquer"
#define WIFI_CONNECTED_BIT BIT0
#define UDP_PORT        9999
#define UDP_PAYLOAD     512
#define UDP_PERIOD_MS   2     /* 512 B every 2 ms == 256 KB/s == ~2 Mbit/s */

static EventGroupHandle_t g_wifi_ev;
static volatile uint32_t g_udp_pkts, g_udp_fails, g_udp_bytes;
static volatile uint32_t g_wifi_disconnects;
static volatile int g_udp_run;
static int g_udp_sock_err;
static uint32_t g_sta_ip;

/* No printf in here: this handler can fire while a timed run is in flight and
 * serial I/O in another task is exactly the kind of distortion this experiment
 * is trying not to introduce. Counters are printed afterwards instead. */
static void wifi_evt(void *arg, esp_event_base_t base, int32_t id, void *data) {
    (void)arg;
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        g_wifi_disconnects++;
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        g_sta_ip = e->ip_info.ip.addr;
        xEventGroupSetBits(g_wifi_ev, WIFI_CONNECTED_BIT);
    }
}

/* Everything except esp_wifi_start(). This is what allocates the driver's
 * memory and creates its tasks, and it runs BEFORE the control measurement so
 * that the control and the loaded run see an identical heap and an identical
 * task set. esp_wifi_start() is what powers the PHY, and that is the single
 * variable between the two runs. */
static esp_err_t wifi_driver_init(int *reduced_out) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        err = nvs_flash_erase();
        if (err != ESP_OK) return err;
        err = nvs_flash_init();
    }
    if (err != ESP_OK) return err;
    err = esp_netif_init();
    if (err != ESP_OK) return err;
    err = esp_event_loop_create_default();
    if (err != ESP_OK) return err;
    if (esp_netif_create_default_wifi_sta() == NULL) return ESP_ERR_NO_MEM;
    g_wifi_ev = xEventGroupCreate();
    if (g_wifi_ev == NULL) return ESP_ERR_NO_MEM;

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&cfg);
    if (err == ESP_ERR_NO_MEM) {
        /* The arena has already taken the largest internal block. If the
         * default buffer counts do not fit in what is left, retry with fewer
         * -- and SAY SO, because it changes what the radio can push. */
        wifi_init_config_t small = WIFI_INIT_CONFIG_DEFAULT();
        small.static_rx_buf_num = 6;
        small.dynamic_rx_buf_num = 12;
        small.tx_buf_type = 1;
        small.dynamic_tx_buf_num = 12;
        err = esp_wifi_init(&small);
        if (err == ESP_OK) *reduced_out = 1;
    }
    if (err != ESP_OK) return err;

    err = esp_wifi_set_storage(WIFI_STORAGE_RAM);
    if (err != ESP_OK) return err;
    err = esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                              wifi_evt, NULL, NULL);
    if (err != ESP_OK) return err;
    err = esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                              wifi_evt, NULL, NULL);
    if (err != ESP_OK) return err;

    wifi_config_t wc;
    memset(&wc, 0, sizeof wc);
    snprintf((char *)wc.sta.ssid, sizeof wc.sta.ssid, "%s", WIFI_SSID);
    snprintf((char *)wc.sta.password, sizeof wc.sta.password, "%s", WIFI_PASS);
    wc.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) return err;
    return esp_wifi_set_config(WIFI_IF_STA, &wc);
}

static esp_err_t wifi_radio_up(int timeout_ms) {
    esp_err_t err = esp_wifi_start();
    if (err != ESP_OK) return err;
    EventBits_t bits = xEventGroupWaitBits(g_wifi_ev, WIFI_CONNECTED_BIT,
                                           pdFALSE, pdTRUE,
                                           pdMS_TO_TICKS(timeout_ms));
    if (!(bits & WIFI_CONNECTED_BIT)) return ESP_ERR_TIMEOUT;
    /* modem sleep would let the radio idle between beacons, which is exactly
     * the load this measurement is meant to include; keep it awake. */
    return esp_wifi_set_ps(WIFI_PS_NONE);
}

/* Association alone is nearly free. This puts real frames in the air and real
 * work in the lwip/WiFi tasks. Broadcast so no peer has to exist and no ARP
 * has to resolve; the packet count and the failure count are both reported, so
 * the load is a measured fact and not an assumption. */
static void udp_blast_task(void *arg) {
    static uint8_t payload[UDP_PAYLOAD];
    (void)arg;
    for (int i = 0; i < UDP_PAYLOAD; i++) payload[i] = (uint8_t)i;

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sock < 0) {
        g_udp_sock_err = errno ? errno : -1;
        g_udp_run = 0;
        vTaskDelete(NULL);
        return;
    }
    int on = 1;
    if (setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &on, sizeof on) < 0)
        g_udp_sock_err = errno ? errno : -1;

    struct sockaddr_in dst;
    memset(&dst, 0, sizeof dst);
    dst.sin_family = AF_INET;
    dst.sin_port = htons(UDP_PORT);
    dst.sin_addr.s_addr = htonl(INADDR_BROADCAST);

    while (g_udp_run) {
        int r = sendto(sock, payload, UDP_PAYLOAD, 0,
                       (struct sockaddr *)&dst, sizeof dst);
        if (r == UDP_PAYLOAD) {
            g_udp_pkts++;
            g_udp_bytes += (uint32_t)r;
        } else {
            g_udp_fails++;
        }
        vTaskDelay(pdMS_TO_TICKS(UDP_PERIOD_MS));
    }
    close(sock);
    vTaskDelete(NULL);
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

/* ---- one point of the radio experiment ---------------------------------- */
/* Three runs: the five timed runs with the plain counting sink (directly
 * comparable with every earlier number taken on this board), one instrumented
 * run for the per-chunk distribution, and one untimed correlation run so the
 * 0.98 gate is verified in THIS configuration and not inferred from another. */
static int64_t run_point(const char *label, snt_nano_config *cfg,
                         const int32_t *ids, int n_ids, const float *gold,
                         size_t n_gold, double audio_sec, ChunkStats *cs,
                         double *corr_out) {
    char b1[48], b2[48];
    snt_nano_stats st;

    int64_t med = run_timed(cfg, ids, n_ids, label, audio_sec);

    g_count = 0;
    g_chunk_used = 0;
    g_chunk_dropped = 0;
    memset(&st, 0, sizeof st);
    g_chunk_mark = esp_timer_get_time();
    int rc = snt_nano_synthesize(cfg, ids, n_ids, chunk_cb, NULL, &st);
    printf("DEVICE: [%s] instrumented run: rc=%d frames=%d samples=%d %lld us  "
           "RTF %s  (compare the median above: the recording must not move it)\n",
           label, rc, st.frames, st.samples, (long long)st.elapsed_us,
           f6((double)st.elapsed_us / 1e6 / audio_sec, b1));
    memset(cs, 0, sizeof *cs);
    if (rc == 0) chunk_report(label, cs);

    CorrSink sink;
    memset(&sink, 0, sizeof sink);
    sink.gold = gold;
    sink.n_gold = n_gold;
    memset(&st, 0, sizeof st);
    snt_port_res_reset();
    rc = snt_nano_synthesize(cfg, ids, n_ids, corr_cb, &sink, &st);
    *corr_out = -2.0;
    if (rc != 0 || sink.pos == 0) {
        printf("DEVICE: [%s] rc=%d, %u samples produced -- NO CORRELATION COMPUTED\n",
               label, rc, (unsigned)sink.pos);
        return med;
    }
    double n = (double)sink.pos;
    double cov = sink.sab - sink.sa * sink.sb / n;
    double cr = cov / sqrt((sink.saa - sink.sa * sink.sa / n) *
                           (sink.sbb - sink.sb * sink.sb / n) + 1e-30);
    *corr_out = cr;
    printf("DEVICE: [%s] CORRELATION = %s  rms_ratio = %s  ==> GATE %s (0.98)\n",
           label, f6(cr, b1), f6(sqrt(sink.saa / n) / sqrt(sink.sbb / n), b2),
           cr > GATE ? "PASS" : "FAIL");
    int64_t tot = g_mv_macs_simd + g_mv_macs_scalar;
    printf("DEVICE: [%s] int8 matvec residency: SIMD %s%%, SCALAR %s%%\n", label,
           f3(tot ? 100.0 * (double)g_mv_macs_simd / (double)tot : 0.0, b1),
           f3(tot ? 100.0 * (double)g_mv_macs_scalar / (double)tot : 0.0, b2));
    return med;
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

    /* ================= PART B: the radio experiment ======================
     * Everything above ran with the radio down, as every earlier number on
     * this board did. What follows measures the same r00 row twice in this one
     * boot: once with the WiFi driver initialised but the PHY never started,
     * once associated to a real AP and actively transmitting. Same arena, same
     * two cores, same model, same fixture. The radio is the only variable.
     *
     * Arena sizing is NOT cosmetic and is the easiest way to get this wrong.
     * Weight staging inside the runtime only happens when the staged block
     * plus the reserved noise plane still fit under the cap, so a smaller
     * arena silently stages less and runs slower -- measured on this board:
     * 122,880 B of arena costs r00 3.9x against 184,320 B (RTF 0.576 vs
     * 0.146), with the high-water mark falling from 128,944 to 110,864 B. Any
     * cap at or above ~135,600 B (the 128,944 B peak plus the 6,656 B noise
     * reserve held during the last staging attempt) reproduces the full-speed
     * behaviour exactly. Take the largest that still fits, refuse to run below
     * the floor, and print arena_peak so the claim is checkable rather than
     * asserted: it must come back 128,944. */
#define PARTB_ARENA_FLOOR 136192   /* 133 KiB: above the 135,600 B staging cliff */
    printf("\n\n================ PART B: radio off vs radio up ================\n");
    free(arena);
    if (parena) free(parena);
    arena = NULL;
    parena = NULL;
    printf("heap after releasing part-A arenas: internal free %u, largest internal "
           "block %u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    size_t bsz = heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    bsz &= ~(size_t)1023;
    void *barena = NULL;
    while (bsz >= PARTB_ARENA_FLOOR) {
        barena = heap_caps_aligned_alloc(16, bsz, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (barena) break;
        bsz -= 1024;
    }
    if (!barena) {
        printf("PART B ABORTED: no internal-SRAM arena of at least %u B is "
               "available, and a smaller one would measure a different (slower) "
               "staging regime rather than the radio\n", (unsigned)PARTB_ARENA_FLOOR);
        printf("================ END ================\n");
        while (1) vTaskDelay(pdMS_TO_TICKS(10000));
    }
    printf("part-B arena: %u bytes @ %p in INTERNAL SRAM (resident=%d)\n",
           (unsigned)bsz, barena, snt_weights_resident(barena));

    snt_nano_config bcfg;
    bcfg.front_blob = front_start;
    bcfg.dec_blob = model_start;
    bcfg.arena = barena;
    bcfg.arena_size = bsz;
    bcfg.dur_override = durs;
    bcfg.noise_seed = ROW_SEED;
    snt_nano_prof_set(0);

    int reduced_bufs = 0;
    esp_err_t werr = wifi_driver_init(&reduced_bufs);
    printf("wifi driver init: %s%s\n", esp_err_to_name(werr),
           reduced_bufs ? " (RETRIED WITH REDUCED BUFFER COUNTS)" : "");
    printf("heap after wifi driver init (PHY still off): internal free %u, "
           "largest internal block %u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    ChunkStats cs_ctl, cs_wifi;
    double corr_ctl = -2.0, corr_wifi = -2.0;
    int64_t med_ctl = -1, med_wifi = -1;
    memset(&cs_ctl, 0, sizeof cs_ctl);
    memset(&cs_wifi, 0, sizeof cs_wifi);

    printf("\n---- CONTROL: radio DOWN (driver initialised, esp_wifi_start not "
           "called, no PHY, no association) ----\n");
    med_ctl = run_point("r00/SRAM/2core/RADIO-OFF", &bcfg, ids, n_ids, gold,
                        n_gold, audio_sec0, &cs_ctl, &corr_ctl);

    int radio_ok = 0;
    uint32_t pkts_before = 0, fails_before = 0;
    if (werr != ESP_OK) {
        printf("\nWIFI RUN SKIPPED: the driver did not initialise (%s). The "
               "control above is the only measurement in this boot.\n",
               esp_err_to_name(werr));
    } else {
        printf("\n---- bringing the radio up: associating to SSID \"%s\" ----\n",
               WIFI_SSID);
        esp_err_t uerr = wifi_radio_up(30000);
        if (uerr != ESP_OK) {
            printf("ASSOCIATION FAILED: %s. NO WIFI TIMING IS REPORTED -- the "
                   "radio was not up.\n", esp_err_to_name(uerr));
        } else {
            printf("associated: IP %u.%u.%u.%u\n",
                   (unsigned)(g_sta_ip & 0xff), (unsigned)((g_sta_ip >> 8) & 0xff),
                   (unsigned)((g_sta_ip >> 16) & 0xff), (unsigned)((g_sta_ip >> 24) & 0xff));
            wifi_ap_record_t ap;
            if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK)
                printf("link: ssid \"%s\" channel %d rssi %d dBm\n",
                       (const char *)ap.ssid, (int)ap.primary, (int)ap.rssi);

            g_udp_run = 1;
            BaseType_t ok = xTaskCreatePinnedToCore(udp_blast_task, "udpblast",
                                                    4096, NULL, 5, NULL, 0);
            if (ok != pdPASS) {
                printf("UDP LOAD TASK DID NOT START -- the radio is associated "
                       "but idle; NO WIFI TIMING IS REPORTED.\n");
                g_udp_run = 0;
            } else {
                vTaskDelay(pdMS_TO_TICKS(3000));   /* let the stream settle */
                printf("udp load after 3 s: %u packets sent, %u failed, %u bytes "
                       "(sock_err %d)\n", (unsigned)g_udp_pkts,
                       (unsigned)g_udp_fails, (unsigned)g_udp_bytes, g_udp_sock_err);
                if (g_udp_pkts == 0) {
                    printf("NOTHING WAS TRANSMITTED -- NO WIFI TIMING IS REPORTED.\n");
                } else {
                    radio_ok = 1;
                    pkts_before = g_udp_pkts;
                    fails_before = g_udp_fails;
                    printf("\n---- LOADED: radio UP, associated, transmitting %d B "
                           "UDP broadcast every %d ms ----\n", UDP_PAYLOAD, UDP_PERIOD_MS);
                    med_wifi = run_point("r00/SRAM/2core/RADIO-TX", &bcfg, ids,
                                         n_ids, gold, n_gold, audio_sec0,
                                         &cs_wifi, &corr_wifi);
                    printf("DEVICE: udp during the loaded runs: %u packets, %u "
                           "failed sends, %u disconnects\n",
                           (unsigned)(g_udp_pkts - pkts_before),
                           (unsigned)(g_udp_fails - fails_before),
                           (unsigned)g_wifi_disconnects);
                    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK)
                        printf("DEVICE: link still up after the loaded runs, "
                               "rssi %d dBm\n", (int)ap.rssi);
                    else
                        printf("DEVICE: WARNING -- link was NOT up at the end of "
                               "the loaded runs\n");
                }
            }
            g_udp_run = 0;
        }
    }

    printf("\n---- PART B SUMMARY (one boot, one binary, one arena) ----\n");
    printf("DEVICE: control  median %lld us  whole-utterance RTF %s  corr %s\n",
           (long long)med_ctl, f6(med_ctl > 0 ? (double)med_ctl / 1e6 / audio_sec0 : -1.0, b1),
           f6(corr_ctl, b2));
    printf("DEVICE: control  per-chunk  n %d  mean %s  p95 %s  max %s  >=1.0 %d\n",
           cs_ctl.n, f6(cs_ctl.mean, b1), f6(cs_ctl.p95, b2),
           f6(cs_ctl.max, (char[48]){0}), cs_ctl.over);
    printf("DEVICE: control  per-chunk 2..N  mean %s  p95 %s  max %s  >=1.0 %d\n",
           f6(cs_ctl.ss_mean, b1), f6(cs_ctl.ss_p95, b2),
           f6(cs_ctl.ss_max, (char[48]){0}), cs_ctl.ss_over);
    if (radio_ok && med_wifi > 0) {
        printf("DEVICE: radio-tx median %lld us  whole-utterance RTF %s  corr %s\n",
               (long long)med_wifi, f6((double)med_wifi / 1e6 / audio_sec0, b1),
               f6(corr_wifi, b2));
        printf("DEVICE: radio-tx per-chunk  n %d  mean %s  p95 %s  max %s  >=1.0 %d\n",
               cs_wifi.n, f6(cs_wifi.mean, b1), f6(cs_wifi.p95, b2),
               f6(cs_wifi.max, (char[48]){0}), cs_wifi.over);
        printf("DEVICE: radio-tx per-chunk 2..N  mean %s  p95 %s  max %s  >=1.0 %d\n",
               f6(cs_wifi.ss_mean, b1), f6(cs_wifi.ss_p95, b2),
               f6(cs_wifi.ss_max, (char[48]){0}), cs_wifi.ss_over);
        printf("DEVICE: RADIO COST = %lld us on the median utterance, "
               "%s%% of the radio-off time\n",
               (long long)(med_wifi - med_ctl),
               f3(100.0 * (double)(med_wifi - med_ctl) / (double)med_ctl, b1));
    } else {
        printf("DEVICE: radio-tx NOT MEASURED (see the reason printed above)\n");
    }
    printf("================ END ================\n");

    while (1) vTaskDelay(pdMS_TO_TICKS(10000));
}
