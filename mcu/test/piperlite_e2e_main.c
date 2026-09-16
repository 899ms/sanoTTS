/* piperlite_e2e_main.c -- phoneme ids in, PCM out, for a whole piperlite
 * voice, in int8 and in fp32, side by side.
 *
 * This is the composition gate: it is what proves the two int8 halves fit
 * together rather than each being separately plausible. It runs
 *
 *   ids -> snt_front_q8_durations -> snt_front_q8_latent
 *       -> snt_piperlite_q8_synthesize -> PCM        (int8 stack)
 *   ids -> snt_front_durations     -> snt_front_latent
 *       -> snt_piperlite_synthesize    -> PCM        (fp32 stack)
 *
 * and reports, per row, the duration agreement and the waveform correlation
 * between the two. The fp32 stack here is the same C code the existing
 * test-front / test-piperlite gates hold against PyTorch, so a number from
 * this tool is anchored without needing a third golden.
 *
 * By default the int8 stack uses ITS OWN durations: that is the deployed
 * path. Be careful reading the correlation in that mode. A one-frame
 * duration difference shifts everything after it by 256 samples, and a
 * sample-aligned correlation between a waveform and a time-shifted copy of
 * itself falls off a cliff -- 0.1 is routine -- while the audio is perfectly
 * fine, just spoken a hair differently. Those rows are flagged
 * [LENGTHS DIFFER]; read their RMS agreement, and read the ASR gate.
 *
 * --lock-durations drives the int8 acoustic + decoder with the FP32 duration
 * student's frame counts. That is not the deployed path, but it is the
 * apples-to-apples number: everything but the timing is identical, so the
 * correlation measures quantisation and nothing else. It is the number to
 * compare against the nano int8 path's 0.994664 on ESP32-S3.
 *
 *   ./piperlite_e2e [--lock-durations] [--rate hz] <front_dir> <dec_dir> \
 *       <length_scale> <out_dir> ids0.bin [...]
 *
 * Writes <out_dir>/NNN-int8.wav and NNN-f32.wav (16-bit PCM, --rate, default
 * 22050) so the same rows can go straight into an ASR intelligibility gate.
 */
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include "snt_front_f32.h"
#include "snt_front_q8.h"
#include "snt_piperlite.h"
#include "snt_piperlite_q8.h"

static int SAMPLE_RATE = 22050;

static void *xload(const char *path, size_t *bytes, int required) {
    FILE *fh = fopen(path, "rb");
    long sz;
    void *buf;
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

static void *xload_in(const char *dir, const char *name, size_t *bytes,
                      int required) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    return xload(path, bytes, required);
}

static void wr_u32(FILE *fh, unsigned v) {
    fputc(v & 255, fh); fputc((v >> 8) & 255, fh);
    fputc((v >> 16) & 255, fh); fputc((v >> 24) & 255, fh);
}

static void wr_u16(FILE *fh, unsigned v) {
    fputc(v & 255, fh); fputc((v >> 8) & 255, fh);
}

static int write_wav(const char *path, const float *pcm, long n) {
    FILE *fh = fopen(path, "wb");
    long i;
    if (!fh) { fprintf(stderr, "cannot write %s\n", path); return -1; }
    fwrite("RIFF", 1, 4, fh);
    wr_u32(fh, (unsigned)(36 + 2 * n));
    fwrite("WAVEfmt ", 1, 8, fh);
    wr_u32(fh, 16);
    wr_u16(fh, 1);
    wr_u16(fh, 1);
    wr_u32(fh, (unsigned)SAMPLE_RATE);
    wr_u32(fh, (unsigned)(SAMPLE_RATE * 2));
    wr_u16(fh, 2);
    wr_u16(fh, 16);
    fwrite("data", 1, 4, fh);
    wr_u32(fh, (unsigned)(2 * n));
    for (i = 0; i < n; i++) {
        float v = pcm[i];
        long s;
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        s = lrintf(v * 32767.0f);
        if (s > 32767) s = 32767;
        if (s < -32768) s = -32768;
        wr_u16(fh, (unsigned)(s & 0xFFFF));
    }
    fclose(fh);
    return 0;
}

static double corr_of(const float *a, const float *b, long n) {
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    long i;
    for (i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
    }
    return (sab - sa * sb / (double)n) /
           sqrt((saa - sa * sa / (double)n) * (sbb - sb * sb / (double)n) + 1e-30);
}

/* Host seconds. A host number is NOT a device number; the useful quantity
 * here is the int8/fp32 RATIO measured in one process on one machine. */
static double now_s(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static double rms_of(const float *a, long n) {
    double s = 0;
    long i;
    for (i = 0; i < n; i++) s += (double)a[i] * a[i];
    return sqrt(s / (double)(n ? n : 1));
}

int main(int argc, char **argv) {
    const char *front_dir, *dec_dir, *out_dir;
    float length_scale;
    size_t fq_meta_nb, fq_w_nb, ff_meta_nb, ff_w_nb;
    size_t dq_meta_nb, dq_w_nb, df_meta_nb, df_w_nb;
    void *fq_meta, *ff_meta, *dq_meta, *df_meta;
    int8_t *fq_w, *dq_w;
    float *ff_w, *df_w;
    snt_front_q8_model fq;
    snt_front_model ff;
    snt_piperlite_q8_model dq;
    snt_piperlite_model df;
    int i, rc, argstart = 1, failures = 0, lock_durations = 0;

    for (; argstart < argc; ) {
        if (strcmp(argv[argstart], "--lock-durations") == 0) {
            lock_durations = 1;
            argstart += 1;
        } else if (argstart + 1 < argc && strcmp(argv[argstart], "--rate") == 0) {
            SAMPLE_RATE = atoi(argv[argstart + 1]);
            argstart += 2;
        } else break;
    }
    if (argc - argstart < 5) {
        fprintf(stderr,
                "usage: %s [--lock-durations] [--rate hz] <front_dir> <dec_dir> "
                "<length_scale> <out_dir> ids0.bin [ids1.bin ...]\n", argv[0]);
        return 2;
    }
    front_dir = argv[argstart];
    dec_dir = argv[argstart + 1];
    length_scale = (float)atof(argv[argstart + 2]);
    out_dir = argv[argstart + 3];
    if (!(length_scale > 0.0f)) {
        fprintf(stderr, "length_scale must be positive\n");
        return 2;
    }

    fq_meta = xload_in(front_dir, "front_meta_q8.bin", &fq_meta_nb, 1);
    fq_w = (int8_t *)xload_in(front_dir, "front_weights_q8.bin", &fq_w_nb, 1);
    ff_meta = xload_in(front_dir, "meta.bin", &ff_meta_nb, 1);
    ff_w = (float *)xload_in(front_dir, "front_weights_f32.bin", &ff_w_nb, 1);
    dq_meta = xload_in(dec_dir, "meta_q8.bin", &dq_meta_nb, 1);
    dq_w = (int8_t *)xload_in(dec_dir, "weights_q8.bin", &dq_w_nb, 1);
    df_meta = xload_in(dec_dir, "meta.bin", &df_meta_nb, 1);
    df_w = (float *)xload_in(dec_dir, "weights_f32.bin", &df_w_nb, 1);

    if ((rc = snt_front_q8_init(&fq, fq_meta, fq_meta_nb, fq_w, fq_w_nb)) != 0) {
        fprintf(stderr, "snt_front_q8_init: %d\n", rc); return 1;
    }
    if ((rc = snt_front_init(&ff, ff_meta, ff_meta_nb, ff_w, ff_w_nb / 4)) != 0) {
        fprintf(stderr, "snt_front_init: %d\n", rc); return 1;
    }
    if ((rc = snt_piperlite_q8_init(&dq, dq_meta, dq_meta_nb, dq_w, dq_w_nb)) != 0) {
        fprintf(stderr, "snt_piperlite_q8_init: %d\n", rc); return 1;
    }
    if ((rc = snt_piperlite_init(&df, df_meta, df_meta_nb, df_w, df_w_nb / 4)) != 0) {
        fprintf(stderr, "snt_piperlite_init: %d\n", rc); return 1;
    }
    if (fq.a_out != dq.in_ch || ff.a_out != df.in_ch || fq.a_out != ff.a_out) {
        fprintf(stderr, "front latent %d/%d channels does not match decoder "
                        "%d/%d -- these are not the same voice\n",
                fq.a_out, ff.a_out, dq.in_ch, df.in_ch);
        return 1;
    }
    printf("front int8 %zu B (+meta %zu) vs fp32 %zu B (+meta %zu)\n",
           fq_w_nb, fq_meta_nb, ff_w_nb, ff_meta_nb);
    printf("decoder int8 %zu B (+meta %zu) vs fp32 %zu B (+meta %zu)\n",
           dq_w_nb, dq_meta_nb, df_w_nb, df_meta_nb);
    printf("stack int8 %zu B vs fp32 %zu B  = %.2fx smaller\n",
           fq_w_nb + fq_meta_nb + dq_w_nb + dq_meta_nb,
           ff_w_nb + ff_meta_nb + df_w_nb + df_meta_nb,
           (double)(ff_w_nb + ff_meta_nb + df_w_nb + df_meta_nb) /
               (double)(fq_w_nb + fq_meta_nb + dq_w_nb + dq_meta_nb));
    printf("durations: %s\n", lock_durations
           ? "LOCKED to the fp32 student (correlation measures quantisation only)"
           : "int8 student's own (the deployed path)");
    printf("%-20s %6s %6s %8s %12s %9s %9s %8s %8s\n",
           "row", "tok", "dfrm", "durmatch", "audio_corr", "rms_i8", "rms_f32",
           "rtf_i8", "rtf_f32");

    for (i = argstart + 4; i < argc; i++) {
        size_t ids_nb;
        int32_t *ids = (int32_t *)xload(argv[i], &ids_nb, 1);
        int n_tokens = (int)(ids_nb / 4);
        int32_t *dq_dur, *df_dur;
        long frames_q, frames_f, min_frames, j, exact = 0;
        float *lat_q, *lat_f, *aq, *af;
        void *arena_q;
        float *arena_f;
        size_t an;
        char path[1024];
        double t0, sec_q = 0.0, sec_f = 0.0;
        const char *base = strrchr(argv[i], '/');
        char stem[256];
        double corr;

        base = base ? base + 1 : argv[i];
        snprintf(stem, sizeof stem, "%s", base);
        { char *dot = strrchr(stem, '.'); if (dot) *dot = 0; }

        if (n_tokens <= 0) { fprintf(stderr, "%s: empty\n", argv[i]); failures++; continue; }
        dq_dur = (int32_t *)malloc((size_t)n_tokens * 4);
        df_dur = (int32_t *)malloc((size_t)n_tokens * 4);
        if (!dq_dur || !df_dur) { fprintf(stderr, "oom\n"); return 1; }

        an = snt_front_q8_duration_arena_bytes(&fq, n_tokens);
        arena_q = malloc(an);
        frames_q = snt_front_q8_durations(&fq, ids, n_tokens, length_scale,
                                          dq_dur, arena_q, an);
        free(arena_q);
        an = snt_front_duration_arena_floats(&ff, n_tokens);
        arena_f = (float *)malloc(an * sizeof(float));
        frames_f = snt_front_durations(&ff, ids, n_tokens, length_scale,
                                       df_dur, arena_f, an);
        free(arena_f);
        if (frames_q <= 0 || frames_f <= 0) {
            fprintf(stderr, "%s: duration failure (%ld / %ld)\n", argv[i],
                    frames_q, frames_f);
            failures++;
            continue;
        }
        for (j = 0; j < n_tokens; j++) if (dq_dur[j] == df_dur[j]) exact++;
        if (lock_durations) {
            memcpy(dq_dur, df_dur, (size_t)n_tokens * sizeof(int32_t));
            frames_q = frames_f;
        }

        lat_q = (float *)malloc((size_t)fq.a_out * (size_t)frames_q * 4);
        lat_f = (float *)malloc((size_t)ff.a_out * (size_t)frames_f * 4);
        an = snt_front_q8_latent_arena_bytes(&fq, n_tokens, frames_q);
        arena_q = malloc(an);
        t0 = now_s();
        rc = snt_front_q8_latent(&fq, ids, dq_dur, n_tokens, frames_q, lat_q,
                                 arena_q, an);
        sec_q += now_s() - t0;
        free(arena_q);
        if (rc != 0) { fprintf(stderr, "%s: q8 latent %d\n", argv[i], rc); return 1; }
        an = snt_front_latent_arena_floats(&ff, n_tokens, frames_f);
        arena_f = (float *)malloc(an * sizeof(float));
        t0 = now_s();
        rc = snt_front_latent(&ff, ids, df_dur, n_tokens, frames_f, lat_f,
                              arena_f, an);
        sec_f += now_s() - t0;
        free(arena_f);
        if (rc != 0) { fprintf(stderr, "%s: f32 latent %d\n", argv[i], rc); return 1; }

        aq = (float *)malloc((size_t)frames_q * SNT_PIPERLITE_HOP * 4);
        af = (float *)malloc((size_t)frames_f * SNT_PIPERLITE_HOP * 4);
        an = snt_piperlite_q8_arena_bytes(&dq, (int)frames_q);
        arena_q = malloc(an);
        t0 = now_s();
        rc = snt_piperlite_q8_synthesize(&dq, lat_q, (int)frames_q, aq, arena_q, an);
        sec_q += now_s() - t0;
        free(arena_q);
        if (rc != 0) { fprintf(stderr, "%s: q8 decoder %d\n", argv[i], rc); return 1; }
        an = snt_piperlite_arena_floats(&df, (int)frames_f);
        arena_f = (float *)malloc(an * sizeof(float));
        t0 = now_s();
        rc = snt_piperlite_synthesize(&df, lat_f, (int)frames_f, af, arena_f, an);
        sec_f += now_s() - t0;
        free(arena_f);
        if (rc != 0) { fprintf(stderr, "%s: f32 decoder %d\n", argv[i], rc); return 1; }

        min_frames = frames_q < frames_f ? frames_q : frames_f;
        corr = corr_of(aq, af, min_frames * SNT_PIPERLITE_HOP);
        {
            double audio_s = (double)(frames_q * SNT_PIPERLITE_HOP) /
                             (double)SAMPLE_RATE;
            printf("%-20s %6d %6ld %5ld/%-3d %12.6f %9.5f %9.5f %8.4f %8.4f%s\n",
                   stem, n_tokens, frames_q, exact, n_tokens, corr,
                   rms_of(aq, frames_q * SNT_PIPERLITE_HOP),
                   rms_of(af, frames_f * SNT_PIPERLITE_HOP),
                   sec_q / audio_s, sec_f / audio_s,
                   frames_q == frames_f ? "" : "  [LENGTHS DIFFER]");
        }

        if (lock_durations) snprintf(stem + strlen(stem), sizeof stem - strlen(stem),
                                     "-locked");
        snprintf(path, sizeof path, "%s/%s-int8.wav", out_dir, stem);
        if (write_wav(path, aq, frames_q * SNT_PIPERLITE_HOP) != 0) failures++;
        snprintf(path, sizeof path, "%s/%s-f32.wav", out_dir, stem);
        if (write_wav(path, af, frames_f * SNT_PIPERLITE_HOP) != 0) failures++;

        free(ids); free(dq_dur); free(df_dur);
        free(lat_q); free(lat_f); free(aq); free(af);
    }
    return failures ? 1 : 0;
}
