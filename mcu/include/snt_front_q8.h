/* snt_front_q8.h -- int8 inference of the sanoTTS piperlite "front half":
 * phoneme ids -> per-token frame counts (DurationStudent) -> expanded
 * features -> latent [a_out, frames] (ContextualLatentStudent "token_context"
 * plus the optional CalibratedLatentStudent output adapter). The latent it
 * produces is the exact layout snt_piperlite_q8_synthesize consumes as z, so
 * the two halves compose into a whole int8 voice.
 *
 * This is the int8 twin of snt_front_f32.c and shares its graph, its float
 * semantics and its slot order; the difference is where the numbers live.
 *
 * Quantization scheme (blobs from tools/export_front_q8.py):
 *   - conv weights: symmetric per-output-channel int8 (max|row|/127); biases
 *     and the blocks' learned residual scalars stay fp32. The int8 payload is
 *     the distribution blob.
 *   - embedding tables: symmetric int8 with one scale per VOCABULARY ROW.
 *     The runtime gathers one row per token anyway, so per-row granularity is
 *     free, and it is what keeps a 7-bit table from costing anything: each
 *     phoneme spends its own full int8 range. (The R7 exporter left these
 *     fp32. At piperlite's vocab sizes they are only 3-4% of the parameters
 *     but 11-13% of an otherwise-int8 blob, which is worth reclaiming if it
 *     is free -- and it measures free. Pass --embed f32 to check.)
 *   - activations: int16 lane holding SNT_FRONT_Q8_QMAX-bounded values with
 *     symmetric per-tensor STATIC clips, calibrated offline over real phoneme
 *     ids. Clips are stored in REAL UNITS in the blob, not as quantisation
 *     steps, so the blob does not encode the runtime's lane width; this file
 *     divides by its own SNT_FRONT_Q8_QMAX.
 *   - convolutions accumulate int16 x int8 -> int32 and dequantise with an
 *     fp32 multiplier (s_in * s_w[oc]); the bias, the SiLU and the residual
 *     add all happen in fp32 at that point, and the result is requantised
 *     into the next plane. snt_front_q8_init() refuses a model whose fan-in
 *     could overflow the int32 accumulator rather than trusting the dims.
 *   - the duration logits and the acoustic output 1x1 dequantise straight to
 *     fp32: exp/round and the decoder input are never quantised.
 *   - the optional output adapter runs fp32 with int8 weights dequantised on
 *     the fly, the same call snt_piperlite_q8.c makes for the waveform post
 *     filter. It is a few hundred multiply-adds per frame and it sits at the
 *     point where error is least recoverable.
 *
 * Build options:
 *   -DSNT_FRONT_Q8_QMAX=n      activation lane maximum (default 2047, 12-bit)
 *   -DSNT_FRONT_Q8_ACT_F32     fp32 activations over the same int8 weights
 *                              (the WebAssembly math path, and the number
 *                              that separates weight cost from lane cost)
 *
 * front_meta_q8.bin carries dims, adapter shape, the clip table and per-tensor
 * kind/offset/size, so this code hardcodes no shapes. Hardcoding them into a
 * generated header is what left esp32c3/fsd/export_front_q8.py usable for
 * exactly one model.
 */
#ifndef SNT_FRONT_Q8_H
#define SNT_FRONT_Q8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNT_FRONT_Q8_MAGIC 0x534E4651L /* 'SNFQ' */
#define SNT_FRONT_Q8_MAX_TENSORS 96
#define SNT_FRONT_Q8_MAX_ACTS 64

#ifndef SNT_FRONT_Q8_QMAX
#define SNT_FRONT_Q8_QMAX 2047
#endif

/* adapter_mode values (match tools/export_front_golden.py and snt_front_f32.h) */
#define SNT_FRONT_Q8_ADAPTER_NONE 0
#define SNT_FRONT_Q8_ADAPTER_AFFINE 1
#define SNT_FRONT_Q8_ADAPTER_DEPTHWISE 2
#define SNT_FRONT_Q8_ADAPTER_LOWRANK 3
#define SNT_FRONT_Q8_ADAPTER_DEPTHWISE_LOWRANK 4

/* per-tensor record kinds in front_meta_q8.bin */
#define SNT_FRONT_Q8_KIND_W8 0
#define SNT_FRONT_Q8_KIND_F32 1
#define SNT_FRONT_Q8_KIND_EMB8 2

typedef struct {
    /* duration student dims */
    int d_vocab, d_hidden, d_depth, d_kernel;
    int d_max_tokens, d_max_duration;
    /* acoustic student dims */
    int a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out;
    /* optional output adapter */
    int adapter_mode, adapter_kernel, adapter_rank;
    int n_tensors, n_act;
    /* per slot exactly one of wq (int8 weights or embedding, with wscale =
     * per-row scales) or f32 (bias / learned scalar / fp32 embedding) is
     * non-NULL; kind[] says which. Slot order is the fp32 exporter's. */
    const int8_t *wq[SNT_FRONT_Q8_MAX_TENSORS];
    const float *wscale[SNT_FRONT_Q8_MAX_TENSORS];
    const float *f32[SNT_FRONT_Q8_MAX_TENSORS];
    signed char kind[SNT_FRONT_Q8_MAX_TENSORS];
    /* activation clips in real units, exporter order (see export_front_q8.py):
     * d_feat, d_x0, then 2 per duration block; a_tfeat, a_tx0, then 2 per
     * token block; a_ffeat, a_fx0, then 2 per frame block. */
    float act[SNT_FRONT_Q8_MAX_ACTS];
    /* bring-up tap: dequantized fp32 view, same names as the fp32 runtime
     * ("dur_feats", "dur_log", "tok_ctx", "frame_feats", "latent_base",
     * "latent"). NULL in production. */
    void (*stage_cb)(const char *name, const float *data, int ch, int len,
                     void *user);
    void *stage_user;
} snt_front_q8_model;

/* Parse front_meta_q8.bin + bind pointers into the int8 blob (flash ok).
 * `meta` must outlive the model: the fp32 pool is read in place.
 * Returns 0 on success, negative on a malformed/out-of-range blob or on a
 * fan-in that would overflow this build's int32 accumulators. */
int snt_front_q8_init(snt_front_q8_model *m,
                      const void *meta, size_t meta_bytes,
                      const int8_t *weights, size_t weight_bytes);

/* Working-memory bytes needed by snt_front_q8_durations (16-aligned base). */
size_t snt_front_q8_duration_arena_bytes(const snt_front_q8_model *m,
                                         int n_tokens);

/* ids[n_tokens] -> dur_out[n_tokens] integer frame counts, each in
 * [1, d_max_duration]. Returns the total frame count, negative on error. */
long snt_front_q8_durations(const snt_front_q8_model *m,
                            const int32_t *ids, int n_tokens,
                            float length_scale,
                            int32_t *dur_out,
                            void *arena, size_t arena_bytes);

/* Working-memory bytes needed by snt_front_q8_latent. */
size_t snt_front_q8_latent_arena_bytes(const snt_front_q8_model *m,
                                       int n_tokens, long frames);

/* ids + durations (every entry >= 1, summing to `frames`) ->
 * latent_out[a_out * frames], channel-major [a_out][frames]. */
int snt_front_q8_latent(const snt_front_q8_model *m,
                        const int32_t *ids, const int32_t *durations,
                        int n_tokens, long frames,
                        float *latent_out,
                        void *arena, size_t arena_bytes);

#ifdef __cplusplus
}
#endif
#endif
