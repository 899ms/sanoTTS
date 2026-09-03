/* nano_short_row.c -- derive a SHORTER benchmark row from a nano fixture.
 *
 * WHY THIS EXISTS
 *   The shipped nano rows are 415-629 frames, and the arena is
 *   ~46.5 KB + 196 B/frame, so the shortest still demands ~129 KB in ONE
 *   contiguous block. Measured, that is more than several real boards will
 *   hand out: a classic ESP32 reports 250,040 B free but a largest block of
 *   110,580 B. Those boards are not short of memory in total; they are short
 *   of a single run of it. A shorter utterance fixes that and nothing else
 *   does.
 *
 * WHAT THE REFERENCE IS, AND IS NOT
 *   The fixture rows in mcu/test/fixtures ship reference waveforms from the
 *   float PyTorch model, and that is what gates the MODEL. This tool cannot
 *   produce those -- a new input needs a new PyTorch run. It writes the HOST
 *   C RUNTIME's output instead, which gates a PORT: does this board's build
 *   reproduce what the reference C implementation produces on the same input?
 *   That catches wrong kernels, bad alignment, SIMD reading unstageable
 *   memory, endianness -- every class of porting bug.
 *
 *   It cannot catch a wrong model, so it is not a substitute for
 *   `make test-nano`, which still gates this runtime against PyTorch at
 *   0.98. The chain is: PyTorch gates the host runtime, the host runtime
 *   gates the board. Emitting a row from a runtime that had not itself
 *   passed would make the whole thing circular, so this tool refuses to run
 *   unless you pass the fixture it was gated on.
 *
 * Usage:
 *   nano_short_row <fixture-dir> <row-index> <max-frames> <out-dir>
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "snt_nano.h"

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    void *buf = malloc((size_t)sz ? (size_t)sz : 1);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) exit(1);
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static void xwrite(const char *dir, const char *name, const void *p, size_t n) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "wb");
    if (!fh) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    if (fwrite(p, 1, n, fh) != n) { fprintf(stderr, "short write %s\n", path); exit(1); }
    fclose(fh);
}

typedef struct { float *pcm; int n, cap; } Sink;

static int sink_cb(const float *pcm, int n, void *user) {
    Sink *s = (Sink *)user;
    if (s->n + n > s->cap) {
        s->cap = (s->n + n) * 2;
        s->pcm = (float *)realloc(s->pcm, (size_t)s->cap * sizeof(float));
        if (!s->pcm) return 1;
    }
    memcpy(s->pcm + s->n, pcm, (size_t)n * sizeof(float));
    s->n += n;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <fixture> <row> <max-frames> <out>\n", argv[0]); return 2; }
    const char *dir = argv[1];
    int row = atoi(argv[2]);
    int max_frames = atoi(argv[3]);
    const char *out = argv[4];

    void *front = xload(dir, "front_q8.bin", NULL);
    void *dec = xload(dir, "model_q8.bin", NULL);

    char name[64], row_id[64] = {0};
    unsigned long long seed = 0;
    {
        char path[512];
        snprintf(path, sizeof path, "%s/rows.txt", dir);
        FILE *mf = fopen(path, "r");
        if (!mf) { fprintf(stderr, "missing %s\n", path); return 1; }
        char rid[64]; int tok, fr, sm; unsigned long long sd;
        for (int i = 0; fscanf(mf, "%63s %d %d %d %llu", rid, &tok, &fr, &sm, &sd) == 5; i++) {
            if (i == row) { snprintf(row_id, sizeof row_id, "%s", rid); seed = sd; break; }
        }
        fclose(mf);
        if (!row_id[0]) { fprintf(stderr, "row %d not in rows.txt\n", row); return 1; }
    }

    size_t nb;
    snprintf(name, sizeof name, "r%02d_ids.bin", row);
    int32_t *ids = (int32_t *)xload(dir, name, &nb);
    int n_ids = (int)(nb / 4);
    snprintf(name, sizeof name, "r%02d_durs.bin", row);
    int32_t *durs = (int32_t *)xload(dir, name, NULL);

    /* Cut on a TOKEN boundary: durations are per phoneme, and splitting one
     * would desynchronise the frame count the decoder is handed from the
     * frames the acoustic model actually produced. */
    int keep = 0, frames = 0;
    for (int i = 0; i < n_ids; i++) {
        if (frames + durs[i] > max_frames) break;
        frames += durs[i];
        keep = i + 1;
    }
    if (keep < 2) { fprintf(stderr, "max-frames %d is too small\n", max_frames); return 1; }

    /* The seed is sha256(row_id) and the row_id is unchanged, so the noise
     * stream is the one this row is defined with. */
    static unsigned char arena[768 * 1024] __attribute__((aligned(16)));
    Sink sink = {0};
    snt_nano_config cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.front_blob = front; cfg.dec_blob = dec;
    cfg.arena = arena; cfg.arena_size = sizeof arena;
    cfg.dur_override = durs; cfg.noise_seed = (uint64_t)seed;

    snt_nano_stats st;
    memset(&st, 0, sizeof st);
    int rc = snt_nano_synthesize(&cfg, ids, keep, sink_cb, &sink, &st);
    if (rc != 0) { fprintf(stderr, "synthesize returned %d\n", rc); return 1; }
    for (int i = 0; i < sink.n; i++) {
        if (!isfinite(sink.pcm[i])) { fprintf(stderr, "non-finite sample %d\n", i); return 1; }
    }

    char cmd[512];
    snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out);
    if (system(cmd) != 0) { fprintf(stderr, "cannot mkdir %s\n", out); return 1; }

    size_t front_n, dec_n;
    void *f2 = xload(dir, "front_q8.bin", &front_n);
    void *d2 = xload(dir, "model_q8.bin", &dec_n);
    xwrite(out, "front_q8.bin", f2, front_n);
    xwrite(out, "model_q8.bin", d2, dec_n);
    xwrite(out, "r00_ids.bin", ids, (size_t)keep * 4);
    xwrite(out, "r00_durs.bin", durs, (size_t)keep * 4);
    xwrite(out, "r00_audio.bin", sink.pcm, (size_t)sink.n * sizeof(float));
    {
        char path[512];
        snprintf(path, sizeof path, "%s/rows.txt", out);
        FILE *fh = fopen(path, "w");
        fprintf(fh, "%s %d %d %d %llu\n", row_id, keep, st.frames, st.samples, seed);
        fclose(fh);
    }
    printf("{\"row_id\":\"%s\",\"tokens\":%d,\"frames\":%d,\"samples\":%d,"
           "\"seconds\":%.3f,\"arena_peak\":%zu,\"seed\":%llu}\n",
           row_id, keep, st.frames, st.samples, st.samples / 22050.0,
           st.arena_peak, seed);
    return 0;
}
