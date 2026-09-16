/* snt_front_q8.c -- int8 piperlite front half (see snt_front_q8.h).
 * Plain C99, caller arena, no malloc.
 *
 * Every float-semantics note in snt_front_f32.c applies here unchanged and
 * for the same reason -- the duration path has to reproduce PyTorch's integer
 * frame counts exactly, and it does that only if linspace, round-half-to-even
 * and the double-precision positional features are reproduced exactly:
 *  - torch.linspace(0,1,n) computes step = 1/(n-1) in fp32, fills i < n/2 as
 *    step*i and i >= n/2 as fma(-step, n-1-i, 1);
 *  - torch.round is round-half-to-even, i.e. rintf under FE_TONEAREST;
 *  - expand_features builds token_pos/duration_pos in double, then casts.
 * Only the arithmetic between those points is quantised.
 */
#include "snt_front_q8.h"

#include <math.h>
#include <string.h>

#define FQ_QMAX SNT_FRONT_Q8_QMAX

#ifdef SNT_FRONT_Q8_ACT_F32
typedef float fq_t;
#else
typedef int16_t fq_t;
#endif

/* ---- meta parsing ------------------------------------------------------ */

static int32_t rd_i32(const unsigned char *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float rd_f32(const unsigned char *p) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return v.f;
}

/* One residual block group: (scale, conv1 w/b, conv2 w/b). */
static int block_slot(int r, int h, int k, int *kind, long *elems, int *rows) {
    int part = r % 5;
    if (part == 0) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = 1; *rows = 0; return 0; }
    if (part == 1 || part == 3) {
        *kind = SNT_FRONT_Q8_KIND_W8; *elems = (long)h * h * k; *rows = h; return 0;
    }
    *kind = SNT_FRONT_Q8_KIND_F32; *elems = h; *rows = 0;
    return 0;
}

static int adapter_slot_count(int mode) {
    switch (mode) {
    case SNT_FRONT_Q8_ADAPTER_NONE: return 0;
    case SNT_FRONT_Q8_ADAPTER_AFFINE: return 2;
    case SNT_FRONT_Q8_ADAPTER_DEPTHWISE: return 3;
    case SNT_FRONT_Q8_ADAPTER_LOWRANK: return 6;
    case SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK: return 7;
    default: return -1;
    }
}

/* Expected (kind, elems, per-row-scale count) of slot `idx`. Slot order is
 * tools/export_front_golden.py's, which tools/export_front_q8.py reuses
 * verbatim; -1 means out of range. */
static int slot_spec(const snt_front_q8_model *m, int idx,
                     int *kind, long *elems, int *rows) {
    int base;
    if (idx < 0) return -1;
    if (idx == 0) {
        *kind = SNT_FRONT_Q8_KIND_EMB8;
        *elems = (long)m->d_vocab * m->d_hidden;
        *rows = m->d_vocab;
        return 0;
    }
    if (idx == 1) {
        *kind = SNT_FRONT_Q8_KIND_W8;
        *elems = (long)m->d_hidden * (m->d_hidden + 3);
        *rows = m->d_hidden;
        return 0;
    }
    if (idx == 2) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->d_hidden; *rows = 0; return 0; }
    base = 3;
    if (idx < base + 5 * m->d_depth)
        return block_slot(idx - base, m->d_hidden, m->d_kernel, kind, elems, rows);
    base += 5 * m->d_depth;
    if (idx == base) {
        *kind = SNT_FRONT_Q8_KIND_W8; *elems = m->d_hidden; *rows = 1; return 0;
    }
    if (idx == base + 1) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = 1; *rows = 0; return 0; }
    base += 2;
    if (idx == base) {
        *kind = SNT_FRONT_Q8_KIND_EMB8;
        *elems = (long)m->a_vocab * m->a_hidden;
        *rows = m->a_vocab;
        return 0;
    }
    if (idx == base + 1) {
        *kind = SNT_FRONT_Q8_KIND_W8;
        *elems = (long)m->a_hidden * (m->a_hidden + 2);
        *rows = m->a_hidden;
        return 0;
    }
    if (idx == base + 2) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->a_hidden; *rows = 0; return 0; }
    base += 3;
    if (idx < base + 5 * m->a_token_depth)
        return block_slot(idx - base, m->a_hidden, m->a_kernel, kind, elems, rows);
    base += 5 * m->a_token_depth;
    if (idx == base) {
        *kind = SNT_FRONT_Q8_KIND_W8;
        *elems = (long)m->a_hidden * (m->a_hidden + 3);
        *rows = m->a_hidden;
        return 0;
    }
    if (idx == base + 1) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->a_hidden; *rows = 0; return 0; }
    base += 2;
    if (idx < base + 5 * m->a_depth)
        return block_slot(idx - base, m->a_hidden, m->a_kernel, kind, elems, rows);
    base += 5 * m->a_depth;
    if (idx == base) {
        *kind = SNT_FRONT_Q8_KIND_W8;
        *elems = (long)m->a_out * m->a_hidden;
        *rows = m->a_out;
        return 0;
    }
    if (idx == base + 1) { *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->a_out; *rows = 0; return 0; }
    base += 2;
    if (m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE ||
        m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) {
        if (idx == base) {
            *kind = SNT_FRONT_Q8_KIND_W8;
            *elems = (long)m->a_out * m->adapter_kernel;
            *rows = m->a_out;
            return 0;
        }
        base += 1;
    }
    if (m->adapter_mode == SNT_FRONT_Q8_ADAPTER_LOWRANK ||
        m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) {
        if (idx == base) {
            *kind = SNT_FRONT_Q8_KIND_W8;
            *elems = (long)m->adapter_rank * m->a_out;
            *rows = m->adapter_rank;
            return 0;
        }
        if (idx == base + 1) {
            *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->adapter_rank; *rows = 0; return 0;
        }
        if (idx == base + 2) {
            *kind = SNT_FRONT_Q8_KIND_W8;
            *elems = (long)m->a_out * m->adapter_rank;
            *rows = m->a_out;
            return 0;
        }
        if (idx == base + 3) {
            *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->a_out; *rows = 0; return 0;
        }
        base += 4;
    }
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE) {
        if (idx == base || idx == base + 1) {
            *kind = SNT_FRONT_Q8_KIND_F32; *elems = m->a_out; *rows = 0; return 0;
        }
    }
    return -1;
}

/* The int32 accumulator must hold QMAX * 127 * fan_in for the widest conv. */
static int accumulator_is_safe(const snt_front_q8_model *m) {
    long fan = (long)m->d_hidden * m->d_kernel;
    long v = (long)m->a_hidden * m->a_kernel;
    if (v > fan) fan = v;
    v = (long)m->a_hidden + 3;
    if (v > fan) fan = v;
    v = m->a_out;                    /* adapter lowrank_down is a 1x1 over C */
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE && v > fan) fan = v;
    /* 2147483647 / (127 * QMAX) is the largest fan-in this lane can take. */
    return fan <= (long)(2147483647L / (127L * (long)FQ_QMAX));
}

int snt_front_q8_init(snt_front_q8_model *m,
                      const void *meta, size_t meta_bytes,
                      const int8_t *weights, size_t weight_bytes) {
    const unsigned char *p = (const unsigned char *)meta;
    size_t cur, pool_hdr;
    const unsigned char *pool_raw;
    long pool_n;
    int i, expected_tensors, expected_acts, adapter_slots;
    if (!m || !p || !weights) return -1;
    if (meta_bytes < 19 * 4) return -2;
    memset(m, 0, sizeof *m);
    if (rd_i32(p) != (int32_t)SNT_FRONT_Q8_MAGIC) return -3;
    if (rd_i32(p + 4) != 1) return -4;
    m->d_vocab = rd_i32(p + 8);
    m->d_hidden = rd_i32(p + 12);
    m->d_depth = rd_i32(p + 16);
    m->d_kernel = rd_i32(p + 20);
    m->d_max_tokens = rd_i32(p + 24);
    m->d_max_duration = rd_i32(p + 28);
    m->a_vocab = rd_i32(p + 32);
    m->a_hidden = rd_i32(p + 36);
    m->a_token_depth = rd_i32(p + 40);
    m->a_depth = rd_i32(p + 44);
    m->a_kernel = rd_i32(p + 48);
    m->a_out = rd_i32(p + 52);
    m->adapter_mode = rd_i32(p + 56);
    m->adapter_kernel = rd_i32(p + 60);
    m->adapter_rank = rd_i32(p + 64);
    m->n_tensors = rd_i32(p + 68);
    m->n_act = rd_i32(p + 72);
    if (m->d_vocab <= 0 || m->d_hidden <= 0 || m->d_depth <= 0 ||
        m->d_kernel <= 0 || m->d_kernel % 2 == 0 ||
        m->d_max_tokens <= 0 || m->d_max_duration <= 0)
        return -5;
    if (m->a_vocab <= 0 || m->a_hidden <= 0 || m->a_token_depth <= 0 ||
        m->a_depth <= 0 || m->a_kernel <= 0 || m->a_kernel % 2 == 0 ||
        m->a_out <= 0)
        return -5;
    adapter_slots = adapter_slot_count(m->adapter_mode);
    if (adapter_slots < 0) return -5;
    if ((m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE ||
         m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) &&
        (m->adapter_kernel <= 0 || m->adapter_kernel % 2 == 0))
        return -5;
    if ((m->adapter_mode == SNT_FRONT_Q8_ADAPTER_LOWRANK ||
         m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) &&
        m->adapter_rank <= 0)
        return -5;
    expected_tensors = (5 + 5 * m->d_depth) +
                       (7 + 5 * m->a_token_depth + 5 * m->a_depth) +
                       adapter_slots;
    if (m->n_tensors != expected_tensors ||
        m->n_tensors > SNT_FRONT_Q8_MAX_TENSORS)
        return -6;
    expected_acts = 6 + 2 * (m->d_depth + m->a_token_depth + m->a_depth);
    if (m->n_act != expected_acts || m->n_act > SNT_FRONT_Q8_MAX_ACTS)
        return -6;
    if (!accumulator_is_safe(m)) return -10;

    cur = 76;
    if (meta_bytes < cur + (size_t)m->n_act * 4) return -2;
    for (i = 0; i < m->n_act; i++) {
        m->act[i] = rd_f32(p + cur);
        if (!(m->act[i] > 0.0f) || !isfinite(m->act[i])) return -7;
        cur += 4;
    }
    if (meta_bytes < cur + (size_t)m->n_tensors * 20 + 4) return -2;
    pool_hdr = cur + (size_t)m->n_tensors * 20;
    pool_n = rd_i32(p + pool_hdr);
    if (pool_n < 0 || meta_bytes < pool_hdr + 4 + (size_t)pool_n * 4) return -2;
    pool_raw = p + pool_hdr + 4;

    for (i = 0; i < m->n_tensors; i++) {
        int kind = rd_i32(p + cur);
        long off = rd_i32(p + cur + 4);
        long size = rd_i32(p + cur + 8);
        long aux_off = rd_i32(p + cur + 12);
        long aux_n = rd_i32(p + cur + 16);
        int want_kind, want_rows;
        long want_elems;
        cur += 20;
        if (slot_spec(m, i, &want_kind, &want_elems, &want_rows) != 0) return -8;
        if (size != want_elems || off < 0) return -8;
        /* An embedding slot may legitimately arrive as fp32 (--embed f32). */
        if (kind != want_kind &&
            !(want_kind == SNT_FRONT_Q8_KIND_EMB8 && kind == SNT_FRONT_Q8_KIND_F32))
            return -8;
        m->kind[i] = (signed char)kind;
        if (kind == SNT_FRONT_Q8_KIND_W8 || kind == SNT_FRONT_Q8_KIND_EMB8) {
            if ((size_t)off + (size_t)size > weight_bytes) return -9;
            if (aux_n != want_rows || aux_off < 0 || aux_off + aux_n > pool_n)
                return -9;
            m->wq[i] = weights + off;
            m->wscale[i] = (const float *)(const void *)(pool_raw + 4 * aux_off);
        } else {
            if (off + size > pool_n) return -9;
            m->f32[i] = (const float *)(const void *)(pool_raw + 4 * off);
        }
    }
    /* NOTE: the fp32 pool points into the caller's meta buffer; it must
     * outlive m. The exporter keeps every section 4-aligned. */
    return 0;
}

/* ---- activation lane ---------------------------------------------------- */

#ifdef SNT_FRONT_Q8_ACT_F32
static fq_t fq_pack(float v, float inv) { (void)inv; return v; }
static float fq_unpack(fq_t q, float s) { (void)s; return q; }
#else
static fq_t fq_pack(float v, float inv) {
    long r = lrintf(v * inv);
    if (r > FQ_QMAX) return (fq_t)FQ_QMAX;
    if (r < -FQ_QMAX) return (fq_t)(-FQ_QMAX);
    return (fq_t)r;
}
static float fq_unpack(fq_t q, float s) { return (float)q * s; }
#endif

/* ---- kernels ------------------------------------------------------------ */

/* One output row of a "same"-padded Conv1d, in real units:
 *   orow[t] = bias + sum_ic sum_k w[oc][ic][k] * x[ic][t + k - K/2]
 * x is a quantised plane with step s_in; w is int8 with per-out-channel
 * scale ws[oc]. acc is int32 scratch of length T (unused in the fp32-act
 * build). */
static void fq_conv_row(float *orow, const fq_t *x, int in_ch, int T,
                        const int8_t *wq, const float *ws, float bias,
                        int oc, int K, float s_in, int32_t *acc) {
    const int8_t *wbase = wq + (long)oc * in_ch * K;
    int pad = K / 2;
    int ic, k, t;
#ifdef SNT_FRONT_Q8_ACT_F32
    float sw = ws[oc];
    (void)acc;
    (void)s_in;
    for (t = 0; t < T; t++) orow[t] = bias;
    for (ic = 0; ic < in_ch; ic++) {
        const fq_t *xrow = x + (long)ic * T;
        const int8_t *wrow = wbase + (long)ic * K;
        for (k = 0; k < K; k++) {
            float wv = (float)wrow[k] * sw;
            int off = k - pad;
            int lo = off < 0 ? -off : 0;
            int hi = off > 0 ? T - off : T;
            if (wv != 0.0f)
                for (t = lo; t < hi; t++) orow[t] += wv * (float)xrow[t + off];
        }
    }
#else
    float mult = s_in * ws[oc];
    memset(acc, 0, (size_t)T * sizeof(int32_t));
    for (ic = 0; ic < in_ch; ic++) {
        const fq_t *xrow = x + (long)ic * T;
        const int8_t *wrow = wbase + (long)ic * K;
        for (k = 0; k < K; k++) {
            int32_t wv = wrow[k];
            int off = k - pad;
            int lo = off < 0 ? -off : 0;
            int hi = off > 0 ? T - off : T;
            if (wv != 0)
                for (t = lo; t < hi; t++) acc[t] += wv * (int32_t)xrow[t + off];
        }
    }
    for (t = 0; t < T; t++) orow[t] = (float)acc[t] * mult + bias;
#endif
}

static float fq_silu(float v) { return v / (1.0f + expf(-v)); }

/* torch.linspace(0, 1, n) with exact CPU-kernel float semantics. */
static void fq_linspace01(float *dst, int n) {
    int half = n / 2, i;
    float step;
    if (n <= 0) return;
    if (n == 1) { dst[0] = 0.0f; return; }
    step = 1.0f / (float)(n - 1);
    for (i = 0; i < half; i++) dst[i] = step * (float)i;
    for (i = half; i < n; i++)
        dst[i] = fmaf(-step, (float)(n - 1 - i), 1.0f);
}

/* depth residual blocks starting at slot base, over plane x (scale s_x),
 * using plane t (scale table entry act_base..) -- 2 clips per block. */
static void fq_run_blocks(const snt_front_q8_model *m, int slot_base,
                          int act_base, int depth, int h, int K, int T,
                          fq_t *x, float *s_x, fq_t *t, int32_t *acc,
                          float *rowf) {
    int b, oc, j;
    for (b = 0; b < depth; b++) {
        int slot = slot_base + 5 * b;
        float scale = m->f32[slot][0];
        const int8_t *w1 = m->wq[slot + 1];
        const float *s1 = m->wscale[slot + 1];
        const float *b1 = m->f32[slot + 2];
        const int8_t *w2 = m->wq[slot + 3];
        const float *s2 = m->wscale[slot + 3];
        const float *b2 = m->f32[slot + 4];
        float clip_s = m->act[act_base + 2 * b];
        float clip_x = m->act[act_base + 2 * b + 1];
        float s_s = clip_s / (float)FQ_QMAX, inv_s = (float)FQ_QMAX / clip_s;
        float s_new = clip_x / (float)FQ_QMAX, inv_new = (float)FQ_QMAX / clip_x;
        for (oc = 0; oc < h; oc++) {
            fq_conv_row(rowf, x, h, T, w1, s1, b1[oc], oc, K, *s_x, acc);
            {
                fq_t *trow = t + (long)oc * T;
                for (j = 0; j < T; j++) trow[j] = fq_pack(fq_silu(rowf[j]), inv_s);
            }
        }
        for (oc = 0; oc < h; oc++) {
            fq_t *xrow = x + (long)oc * T;
            fq_conv_row(rowf, t, h, T, w2, s2, b2[oc], oc, K, s_s, acc);
            for (j = 0; j < T; j++) {
                float v = fq_unpack(xrow[j], *s_x) + scale * rowf[j];
                xrow[j] = fq_pack(v, inv_new);
            }
        }
        *s_x = s_new;
    }
}

static void fq_tap(const snt_front_q8_model *m, const char *name,
                   const fq_t *q, float s, int ch, long len, float *scratch) {
    long n = (long)ch * len, i;
    if (!m->stage_cb) return;
    for (i = 0; i < n; i++) scratch[i] = fq_unpack(q[i], s);
    m->stage_cb(name, scratch, ch, (int)len, m->stage_user);
}

/* Gather one embedding row into `dst` (fp32, `h` values). */
static int fq_embed_row(const snt_front_q8_model *m, int slot, int vocab,
                        int h, long id, float *dst) {
    int c;
    if (id < 0 || id >= vocab) return -1;
    if (m->kind[slot] == SNT_FRONT_Q8_KIND_F32) {
        const float *row = m->f32[slot] + id * h;
        for (c = 0; c < h; c++) dst[c] = row[c];
    } else {
        const int8_t *row = m->wq[slot] + id * h;
        float s = m->wscale[slot][id];
        for (c = 0; c < h; c++) dst[c] = (float)row[c] * s;
    }
    return 0;
}

/* ---- arena ------------------------------------------------------------- */

static unsigned char *align16(unsigned char *p) {
    return p + ((16 - ((uintptr_t)p & 15)) & 15);
}

size_t snt_front_q8_duration_arena_bytes(const snt_front_q8_model *m,
                                         int n_tokens) {
    long h, N;
    if (!m || n_tokens <= 0) return 0;
    h = m->d_hidden;
    N = n_tokens;
    /* feat (h+3)N + x hN + t hN planes, int32 acc N, fp32 row N, emb row h,
     * plus 3N floats for the bring-up tap when one is attached */
    return (size_t)((3 * h + 3) * N * (long)sizeof(fq_t) + 8 * N + 4 * h +
                    (m->stage_cb ? 12 * N : 0) + 128);
}

static long fq_feat_rows(const snt_front_q8_model *m) {
    return m->a_hidden + 3;
}

size_t snt_front_q8_latent_arena_bytes(const snt_front_q8_model *m,
                                       int n_tokens, long frames) {
    long h, C, T, N, bytes, widest;
    if (!m || n_tokens <= 0 || frames < n_tokens) return 0;
    widest = m->a_out > m->a_hidden + 3 ? m->a_out : m->a_hidden + 3;
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE)
        widest += m->adapter_rank + m->a_out;
    if (frames > 0x7FFFFFFFL / (4L * (widest + 2L * m->a_hidden + 4L)))
        return 0;   /* would overflow a 32-bit long; the caller sees 0 */
    h = m->a_hidden;
    C = m->a_out;
    T = frames;
    N = n_tokens;
    bytes = (long)sizeof(fq_t) * (h * N + fq_feat_rows(m) * T + 2 * h * T);
    bytes += 8 * T;                               /* int32 acc + fp32 row */
    bytes += 4 * (h + 3 > C ? h + 3 : C);         /* tap / embed scratch */
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE)
        bytes += 4 * (C * T + (long)m->adapter_rank * T);
    if (m->stage_cb) bytes += 4 * (fq_feat_rows(m) * T);
    return (size_t)(bytes + 256);
}

/* ---- duration student --------------------------------------------------- */

long snt_front_q8_durations(const snt_front_q8_model *m,
                            const int32_t *ids, int n_tokens,
                            float length_scale,
                            int32_t *dur_out,
                            void *arena, size_t arena_bytes) {
    int h, N, i, c, oc;
    fq_t *feat, *x, *t;
    int32_t *acc;
    float *rowf, *erow, *tapbuf;
    unsigned char *base;
    float clip_feat, inv_feat, s_feat, clip_x0, s_x, inv_x0;
    float length_hint, total_f;
    long total;
    int last;

    if (!m || !ids || n_tokens <= 0 || !dur_out || !arena) return -1;
    if (!(length_scale > 0.0f) || !isfinite(length_scale)) return -1;
    if (arena_bytes < snt_front_q8_duration_arena_bytes(m, n_tokens)) return -2;
    h = m->d_hidden;
    N = n_tokens;
    base = align16((unsigned char *)arena);
    feat = (fq_t *)(void *)base;
    x = feat + (long)(h + 3) * N;
    t = x + (long)h * N;
    base = align16((unsigned char *)(t + (long)h * N));
    acc = (int32_t *)(void *)base;
    rowf = (float *)(void *)(base + 4 * (long)N);
    erow = rowf + N;
    tapbuf = erow + h;

    clip_feat = m->act[0];
    s_feat = clip_feat / (float)FQ_QMAX;
    inv_feat = (float)FQ_QMAX / clip_feat;
    clip_x0 = m->act[1];
    s_x = clip_x0 / (float)FQ_QMAX;
    inv_x0 = (float)FQ_QMAX / clip_x0;

    /* input features: embedded ids [h][N] + positions, length_hint, valid */
    for (i = 0; i < N; i++) {
        if (fq_embed_row(m, 0, m->d_vocab, h, ids[i], erow) != 0) return -3;
        for (c = 0; c < h; c++)
            feat[(long)c * N + i] = fq_pack(erow[c], inv_feat);
    }
    fq_linspace01(rowf, N);
    for (i = 0; i < N; i++)
        feat[(long)h * N + i] = fq_pack(rowf[i], inv_feat);
    length_hint = log1pf((float)N) / (float)log1p((double)m->d_max_tokens);
    for (i = 0; i < N; i++) {
        feat[(long)(h + 1) * N + i] = fq_pack(length_hint, inv_feat);
        feat[(long)(h + 2) * N + i] = fq_pack(1.0f, inv_feat);
    }
    if (m->stage_cb)
        fq_tap(m, "dur_feats", feat + (long)h * N, s_feat, 3, N, tapbuf);

    for (oc = 0; oc < h; oc++) {
        fq_t *xrow = x + (long)oc * N;
        fq_conv_row(rowf, feat, h + 3, N, m->wq[1], m->wscale[1],
                    m->f32[2][oc], oc, 1, s_feat, acc);
        for (i = 0; i < N; i++) xrow[i] = fq_pack(rowf[i], inv_x0);
    }
    fq_run_blocks(m, 3, 2, m->d_depth, h, m->d_kernel, N, x, &s_x, t, acc, rowf);

    last = 3 + 5 * m->d_depth;
    fq_conv_row(rowf, x, h, N, m->wq[last], m->wscale[last],
                m->f32[last + 1][0], 0, 1, s_x, acc);
    if (m->stage_cb) m->stage_cb("dur_log", rowf, 1, N, m->stage_user);

    /* predict_durations: exp -> clamp_min(1) -> *scale -> round -> clamp */
    total = 0;
    for (i = 0; i < N; i++) {
        float v = expf(rowf[i]);
        if (v < 1.0f) v = 1.0f;
        v = rintf(v * length_scale);
        if (v < 1.0f) v = 1.0f;
        total_f = (float)m->d_max_duration;
        if (v > total_f) v = total_f;
        dur_out[i] = (int32_t)v;
        total += (long)dur_out[i];
    }
    return total;
}

/* ---- acoustic student ---------------------------------------------------- */

int snt_front_q8_latent(const snt_front_q8_model *m,
                        const int32_t *ids, const int32_t *durations,
                        int n_tokens, long frames,
                        float *latent_out,
                        void *arena, size_t arena_bytes) {
    int h, C, N, i, c, oc;
    long T, sum, ti, j, pos;
    int A0, TB, F0, FB, O0, AD, act_a, act_f;
    fq_t *tok, *feat, *x, *tt;
    int32_t *acc;
    float *rowf, *erow, *tapbuf = NULL, *ad_cur = NULL, *ad_r = NULL;
    unsigned char *base;
    float clip, s_tfeat, inv_tfeat, s_tok, s_ffeat, inv_ffeat, s_x, inv_fx0;
    float maxd, log_maxd;

    if (!m || !ids || !durations || n_tokens <= 0 || frames <= 0 ||
        !latent_out || !arena)
        return -1;
    sum = 0;
    for (i = 0; i < n_tokens; i++) {
        if (durations[i] < 1) return -3;
        sum += durations[i];
    }
    if (sum != frames) return -3;
    /* `long` is 32-bit on the ARMv7 targets this runtime exists for, so the
     * plane sizes have to be checked before they are computed, not after. */
    {
        long widest = m->a_out > m->a_hidden + 3 ? m->a_out : m->a_hidden + 3;
        if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE)
            widest += m->adapter_rank + m->a_out;
        if (frames > 0x7FFFFFFFL / (4L * (widest + 2L * m->a_hidden + 4L)))
            return -1;
    }
    if (arena_bytes < snt_front_q8_latent_arena_bytes(m, n_tokens, frames))
        return -2;
    h = m->a_hidden;
    C = m->a_out;
    N = n_tokens;
    T = frames;
    A0 = 5 + 5 * m->d_depth;                  /* acoustic embedding slot */
    TB = A0 + 3;                              /* token blocks */
    F0 = TB + 5 * m->a_token_depth;           /* frame_input_proj slot */
    FB = F0 + 2;                              /* frame blocks */
    O0 = FB + 5 * m->a_depth;                 /* output 1x1 */
    AD = O0 + 2;                              /* adapter base */
    act_a = 2 + 2 * m->d_depth;               /* a_tfeat */
    act_f = act_a + 2 + 2 * m->a_token_depth; /* a_ffeat */

    base = align16((unsigned char *)arena);
    tok = (fq_t *)(void *)base;
    feat = tok + (long)h * N;
    x = feat + fq_feat_rows(m) * T;
    tt = x + (long)h * T;
    base = align16((unsigned char *)(tt + (long)h * T));
    acc = (int32_t *)(void *)base;
    rowf = (float *)(void *)(base + 4 * T);
    erow = rowf + T;
    base = align16((unsigned char *)(erow + (h + 3 > C ? h + 3 : C)));
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE) {
        ad_cur = (float *)(void *)base;
        ad_r = ad_cur + (long)C * T;
        base = align16((unsigned char *)(ad_r + (long)m->adapter_rank * T));
    }
    if (m->stage_cb) tapbuf = (float *)(void *)base;

    /* -- token stage -- */
    clip = m->act[act_a];
    s_tfeat = clip / (float)FQ_QMAX;
    inv_tfeat = (float)FQ_QMAX / clip;
    for (i = 0; i < N; i++) {
        if (fq_embed_row(m, A0, m->a_vocab, h, ids[i], erow) != 0) return -3;
        for (c = 0; c < h; c++)
            feat[(long)c * N + i] = fq_pack(erow[c], inv_tfeat);
    }
    fq_linspace01(rowf, N);
    for (i = 0; i < N; i++)
        feat[(long)h * N + i] = fq_pack(rowf[i], inv_tfeat);
    maxd = 1.0f;
    for (i = 0; i < N; i++)
        if ((float)durations[i] > maxd) maxd = (float)durations[i];
    log_maxd = log1pf(maxd);
    for (i = 0; i < N; i++)
        feat[(long)(h + 1) * N + i] =
            fq_pack(log1pf((float)durations[i]) / log_maxd, inv_tfeat);

    clip = m->act[act_a + 1];
    s_tok = clip / (float)FQ_QMAX;
    for (oc = 0; oc < h; oc++) {
        fq_t *trow = tok + (long)oc * N;
        fq_conv_row(rowf, feat, h + 2, N, m->wq[A0 + 1], m->wscale[A0 + 1],
                    m->f32[A0 + 2][oc], oc, 1, s_tfeat, acc);
        for (i = 0; i < N; i++) trow[i] = fq_pack(rowf[i], (float)FQ_QMAX / clip);
    }
    fq_run_blocks(m, TB, act_a + 2, m->a_token_depth, h, m->a_kernel, N,
                  tok, &s_tok, feat, acc, rowf);
    if (m->stage_cb) fq_tap(m, "tok_ctx", tok, s_tok, h, N, tapbuf);

    /* -- expand token context + positional features to frames -- */
    clip = m->act[act_f];
    s_ffeat = clip / (float)FQ_QMAX;
    inv_ffeat = (float)FQ_QMAX / clip;
    for (c = 0; c < h; c++) {
        fq_t *orow = feat + (long)c * T;
        const fq_t *trow = tok + (long)c * N;
        pos = 0;
        for (ti = 0; ti < N; ti++) {
            float v = fq_unpack(trow[ti], s_tok);
            fq_t q = fq_pack(v, inv_ffeat);
            for (j = 0; j < durations[ti]; j++) orow[pos++] = q;
        }
    }
    fq_linspace01(rowf, (int)T);
    for (j = 0; j < T; j++)
        feat[(long)h * T + j] = fq_pack(rowf[j], inv_ffeat);
    {
        /* expand_features: python doubles cast to fp32 */
        long tcount = N > 1 ? N - 1 : 1;
        fq_t *tp = feat + (long)(h + 1) * T;
        fq_t *dp = feat + (long)(h + 2) * T;
        pos = 0;
        for (ti = 0; ti < N; ti++) {
            long d = durations[ti];
            float tv = (float)((double)ti / (double)tcount);
            fq_t qtv = fq_pack(tv, inv_ffeat);
            for (j = 0; j < d; j++) {
                tp[pos] = qtv;
                dp[pos] = fq_pack(d == 1 ? 0.0f
                                         : (float)((double)j / (double)(d - 1)),
                                  inv_ffeat);
                pos++;
            }
        }
    }
    if (m->stage_cb)
        fq_tap(m, "frame_feats", feat + (long)h * T, s_ffeat, 3, T, tapbuf);

    /* -- frame stage -- */
    clip = m->act[act_f + 1];
    s_x = clip / (float)FQ_QMAX;
    inv_fx0 = (float)FQ_QMAX / clip;
    for (oc = 0; oc < h; oc++) {
        fq_t *xrow = x + (long)oc * T;
        fq_conv_row(rowf, feat, h + 3, (int)T, m->wq[F0], m->wscale[F0],
                    m->f32[F0 + 1][oc], oc, 1, s_ffeat, acc);
        for (j = 0; j < T; j++) xrow[j] = fq_pack(rowf[j], inv_fx0);
    }
    fq_run_blocks(m, FB, act_f + 2, m->a_depth, h, m->a_kernel, (int)T,
                  x, &s_x, tt, acc, rowf);

    /* -- output 1x1 straight to fp32 -- */
    for (oc = 0; oc < C; oc++)
        fq_conv_row(latent_out + (long)oc * T, x, h, (int)T, m->wq[O0],
                    m->wscale[O0], m->f32[O0 + 1][oc], oc, 1, s_x, acc);
    if (m->stage_cb)
        m->stage_cb("latent_base", latent_out, C, (int)T, m->stage_user);

    /* -- optional output adapter, fp32 with weights dequantized on the fly -- */
    if (m->adapter_mode != SNT_FRONT_Q8_ADAPTER_NONE) {
        int slot = AD;
        const float *ad_scale, *ad_bias;
        if (m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE ||
            m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) {
            int K = m->adapter_kernel, pad = K / 2, k;
            const int8_t *dw = m->wq[slot];
            const float *dws = m->wscale[slot];
            slot++;
            for (c = 0; c < C; c++) {
                float *orow = ad_cur + (long)c * T;
                const float *xrow = latent_out + (long)c * T;
                const int8_t *wrow = dw + (long)c * K;
                float sw = dws[c];
                for (j = 0; j < T; j++) orow[j] = 0.0f;
                for (k = 0; k < K; k++) {
                    float wv = (float)wrow[k] * sw;
                    int off = k - pad;
                    long lo = off < 0 ? -(long)off : 0;
                    long hi = off > 0 ? T - off : T;
                    if (wv != 0.0f)
                        for (j = lo; j < hi; j++) orow[j] += wv * xrow[j + off];
                }
            }
        } else {
            memcpy(ad_cur, latent_out, (size_t)C * (size_t)T * sizeof(float));
        }
        if (m->adapter_mode == SNT_FRONT_Q8_ADAPTER_LOWRANK ||
            m->adapter_mode == SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK) {
            int rank = m->adapter_rank, r;
            const int8_t *dwn = m->wq[slot];
            const float *dwns = m->wscale[slot];
            const float *dnb = m->f32[slot + 1];
            const int8_t *dup = m->wq[slot + 2];
            const float *dups = m->wscale[slot + 2];
            const float *dub = m->f32[slot + 3];
            for (r = 0; r < rank; r++) {
                float *orow = ad_r + (long)r * T;
                float sw = dwns[r], b = dnb[r];
                for (j = 0; j < T; j++) orow[j] = b;
                for (c = 0; c < C; c++) {
                    float wv = (float)dwn[(long)r * C + c] * sw;
                    const float *xrow = ad_cur + (long)c * T;
                    if (wv != 0.0f)
                        for (j = 0; j < T; j++) orow[j] += wv * xrow[j];
                }
                for (j = 0; j < T; j++) orow[j] = tanhf(orow[j]);
            }
            for (c = 0; c < C; c++) {
                float *orow = ad_cur + (long)c * T;
                float sw = dups[c], b = dub[c];
                for (j = 0; j < T; j++) orow[j] += b;
                for (r = 0; r < rank; r++) {
                    float wv = (float)dup[(long)c * rank + r] * sw;
                    const float *rrow = ad_r + (long)r * T;
                    if (wv != 0.0f)
                        for (j = 0; j < T; j++) orow[j] += wv * rrow[j];
                }
            }
            slot += 4;
        }
        ad_scale = m->f32[slot];
        ad_bias = m->f32[slot + 1];
        for (c = 0; c < C; c++) {
            float s = ad_scale[c], b = ad_bias[c];
            float *orow = latent_out + (long)c * T;
            const float *irow = ad_cur + (long)c * T;
            for (j = 0; j < T; j++) orow[j] = irow[j] * s + b;
        }
    }
    if (m->stage_cb) m->stage_cb("latent", latent_out, C, (int)T, m->stage_user);
    return 0;
}
