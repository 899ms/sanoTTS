"""Numpy runtime for the nano voices (heart, heart-nano).

The piperlite voices in `models.py` are a different graph: `duration_conv` /
`token_context` / `piperlite`. The nano lineage is duration student ->
contextual acoustic student -> mel-100 -> a ConvNeXt1D decoder -> iSTFT, with
a noise-fed decoder. `models.py` raises NotImplementedError on it, which is
why heart could not be played from Python.

WEIGHTS
    Read straight from the shipped `front_q8.bin` / `model_q8.bin` using the
    byte offsets in `nano_q8_meta.h`, so a voice package needs no re-export.
    Rows are dequantised to float32 (`w = q * scale`) and the arithmetic runs
    in float.

    That is deliberately NOT what the C runtime does. It also quantises
    activations, one dynamic scale per output column, because an MCU has int8
    kernels and no spare cycles. Reproducing that here would buy nothing: a
    Python caller has floats, and dequantised-weight float math lands closer
    to the PyTorch reference than the device does, not further away. So this
    is gated against the same PyTorch reference the C is gated against, at the
    same 0.98 threshold, rather than against the C's output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from .nano_rng import seeded_noise

LENGTH_SCALE = 1.0
DC_BLOCK_R = 0.9973
DC_BLOCK_TAPS = 4096


def parse_meta_header(path: Path) -> dict[str, int]:
    """`#define NAME value` -> dict. The header is generated, never hand-edited."""
    out: dict[str, int] = {}
    pattern = re.compile(r"^#define\s+([A-Z0-9_]+)\s+(-?\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    if "NANO_FRONT_BYTES" not in out:
        raise ValueError(f"{path} does not look like a nano meta header")
    return out


@dataclass
class Q8Layer:
    """One quantised layer: int8 rows, a per-row scale, and a bias.

    `weight` is already dequantised and trimmed to `in_flat`; the exporter pads
    each row out to a 16-byte multiple so the device SIMD kernels can read
    whole groups, and those trailing zeros are shipped bytes, not parameters.
    """
    weight: np.ndarray   # [out, in_flat] float32
    bias: np.ndarray     # [out] float32


class NanoWeights:
    def __init__(self, front: bytes, dec: bytes, meta: dict[str, int]):
        self._front, self._dec, self.m = front, dec, meta

    def _q8(self, blob: bytes, off_w: int, off_s: int, off_b: int,
            out_ch: int, n16: int, in_flat: int) -> Q8Layer:
        """Read one weight region.

        NANO_WEIGHT_FORMAT selects the row dtype: 0 is int8 with a
        per-output-channel scale, 1 is float32 rows with unit scales (the
        browser build, and the 2.27M `heart` voice, which does not survive
        int8). The region layout is otherwise identical, so both go through
        the same offsets; only the element type and stride differ.
        """
        if self.m.get("NANO_WEIGHT_FORMAT", 0) == 1:
            rows = np.frombuffer(blob, dtype=np.float32, count=out_ch * n16,
                                 offset=off_w).reshape(out_ch, n16)
            w = rows[:, :in_flat].astype(np.float32)
        else:
            q = np.frombuffer(blob, dtype=np.int8, count=out_ch * n16,
                              offset=off_w).reshape(out_ch, n16)
            scale = np.frombuffer(blob, dtype=np.float32, count=out_ch, offset=off_s)
            w = q[:, :in_flat].astype(np.float32) * scale[:, None]
        bias = np.frombuffer(blob, dtype=np.float32, count=out_ch, offset=off_b)
        return Q8Layer(np.ascontiguousarray(w), bias.astype(np.float32).copy())

    def _f32(self, blob: bytes, off: int, count: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32, count=count, offset=off).copy()

    def front_q8(self, name: str, out_ch: int, in_flat: int) -> Q8Layer:
        m = self.m
        return self._q8(self._front, m[f"NOFF_{name}_W8"], m[f"NOFF_{name}_SCALE"],
                        m[f"NOFF_{name}_BIAS"], out_ch, m[f"NANO_{name}_N16"], in_flat)

    def front_f32(self, name: str, count: int) -> np.ndarray:
        return self._f32(self._front, self.m[f"NOFF_{name}_F32"], count)

    def dec_q8(self, name: str, out_ch: int, in_flat: int) -> Q8Layer:
        m = self.m
        return self._q8(self._dec, m[f"DOFF_{name}_W8"], m[f"DOFF_{name}_SCALE"],
                        m[f"DOFF_{name}_BIAS"], out_ch, m[f"NANO_{name}_N16"], in_flat)

    def dec_f32(self, name: str, count: int) -> np.ndarray:
        return self._f32(self._dec, self.m[f"DOFF_{name}_F32"], count)


# ---- primitives ---------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), computed without overflowing.

    The naive 1/(1+exp(-x)) overflows for large negative x. The result is
    still right -- x/inf is 0 -- but it raises a RuntimeWarning on every
    utterance, and a library that cries wolf teaches its callers to ignore
    warnings. Branching on the sign keeps both exponentials bounded.
    """
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = x[~pos] * e / (1.0 + e)
    return out


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf GELU -- what nn.GELU() computes by default, NOT the tanh
    approximation. numpy has no erf, and pulling in scipy for one function
    would add a heavy dependency to a package that advertises numpy only, so
    the identity erf(z) = 2*Phi(z*sqrt(2)) - 1 is not available either. A
    vectorised math.erf is fast enough: the decoder calls it 4 times over
    [144, T], which is milliseconds."""
    return (0.5 * x * (1.0 + _ERF(x.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32)


_ERF = np.vectorize(math.erf, otypes=[np.float64])


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, kernel: int) -> np.ndarray:
    """[C_in, T] -> [C_out, T], 'same' padding, weight rows [C_out, C_in*K].

    Rows are laid out exactly as the device kernel consumes them, which is
    torch's [C_out, C_in, K] flattened, so the gather below must walk the
    kernel axis fastest.
    """
    c_in, t = x.shape
    pad = kernel // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    gathered = np.empty((c_in * kernel, t), dtype=np.float32)
    for k in range(kernel):
        gathered[k::kernel] = xp[:, k : k + t]
    return (w @ gathered + b[:, None]).astype(np.float32)


def residual_block(x: np.ndarray, c0: Q8Layer, c1: Q8Layer, scale: float,
                   kernel: int) -> np.ndarray:
    """x + scale * conv1(silu(conv0(x))) -- the front end's ResidualConvBlock.

    The training code multiplies by a padding mask on both sides; inference is
    a single unpadded sequence, so the mask is all ones and drops out.
    """
    h = conv1d(x, c0.weight, c0.bias, kernel)
    h = silu(h)
    h = conv1d(h, c1.weight, c1.bias, kernel)
    return (x + scale * h).astype(np.float32)


# The decoder's norms are built by make_norm(), which asks for eps=1e-6 --
# NOT torch's 1e-5 default. The difference looks negligible and is not: it
# compounds through four blocks and is then amplified by the exp() in the
# magnitude head, which cost 0.06 of correlation and a third of the output
# amplitude before it was found.
LAYER_NORM_EPS = 1e-6


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
               eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """Over the channel axis of [C, T], as torch does on channel-last."""
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * w[:, None] + b[:, None]).astype(np.float32)


# ---- stages -------------------------------------------------------------

def duration_forward(w: NanoWeights, ids: np.ndarray,
                     length_scale: float = 1.0) -> np.ndarray:
    """Phoneme ids -> integer frame counts, one per token.

    `length_scale` > 1 slows speech down, < 1 speeds it up, applied before
    rounding and the per-token clamp exactly as the C runtime does.
    """
    m = w.m
    H = m["NANO_DUR_HIDDEN"]
    K = m["NANO_DUR_KERNEL"]
    n = int(ids.size)

    emb = w.front_f32("DUR_EMB", m["NANO_VOCAB"] * H).reshape(m["NANO_VOCAB"], H)
    x = emb[ids].T.astype(np.float32)                       # [H, n]

    # positions, length hint and valid hint, exactly as DurationStudent builds
    # them. Inference is one unpadded sequence, so valid_hint is all ones and
    # lengths == n.
    positions = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
    length_hint = np.float32(math.log1p(max(n, 1)) / math.log1p(float(m["NANO_DUR_MAX_TOKENS"])))
    feats = np.stack([positions,
                      np.full(n, length_hint, dtype=np.float32),
                      np.ones(n, dtype=np.float32)])
    x = np.concatenate([x, feats], axis=0)                  # [H+3, n]

    proj = w.front_q8("DUR_PROJ", H, H + 3)
    x = conv1d(x, proj.weight, proj.bias, 1)
    for b in range(m["NANO_DUR_DEPTH"]):
        c0 = w.front_q8(f"DUR_B{b}_C0", H, H * K)
        c1 = w.front_q8(f"DUR_B{b}_C1", H, H * K)
        scale = float(w.front_f32(f"DUR_B{b}_SCALE", 1)[0])
        x = residual_block(x, c0, c1, scale, K)

    out = w.front_q8("DUR_OUT", 1, H)
    log_d = conv1d(x, out.weight, out.bias, 1)[0]
    d = np.round(np.maximum(np.exp(log_d), 1.0) * float(length_scale))
    return np.clip(d, 1, m["NANO_DUR_MAX_DURATION"]).astype(np.int32)


def acoustic_forward(w: NanoWeights, ids: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """ids + durations -> mel-100 [100, T]. ContextualLatentStudent."""
    m = w.m
    H = m["NANO_AC_HIDDEN"]
    K = m["NANO_AC_KERNEL"]
    n = int(ids.size)

    emb = w.front_f32("AC_EMB", m["NANO_VOCAB"] * H).reshape(m["NANO_VOCAB"], H)
    token_x = emb[ids].T.astype(np.float32)                 # [H, n]

    token_count = max(n - 1, 1)
    token_pos_tok = (np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1
                     else np.zeros(1, np.float32))
    dur_f = durations.astype(np.float32)
    max_dur = max(float(dur_f.max()), 1.0)
    duration_hint = (np.log1p(dur_f) / np.log1p(max_dur)).astype(np.float32)

    token_x = np.concatenate([token_x, np.stack([token_pos_tok, duration_hint])], axis=0)
    tproj = w.front_q8("AC_TPROJ", H, H + 2)
    token_x = conv1d(token_x, tproj.weight, tproj.bias, 1)
    for b in range(m["NANO_AC_TOKEN_DEPTH"]):
        c0 = w.front_q8(f"AC_TB{b}_C0", H, H * K)
        c1 = w.front_q8(f"AC_TB{b}_C1", H, H * K)
        scale = float(w.front_f32(f"AC_TB{b}_SCALE", 1)[0])
        token_x = residual_block(token_x, c0, c1, scale, K)

    # Repeat each contextual token state across its own frames, and build the
    # three frame features the same way expand_features does -- note token_pos
    # divides by (n_tokens - 1) while duration_pos divides by (duration - 1),
    # and a duration of 1 contributes a single 0.0 rather than a division.
    x = np.repeat(token_x, durations, axis=1)               # [H, T]
    frames = int(x.shape[1])
    frame_pos = (np.linspace(0.0, 1.0, frames, dtype=np.float32) if frames > 1
                 else np.zeros(1, np.float32))
    token_pos = np.empty(frames, dtype=np.float32)
    duration_pos = np.empty(frames, dtype=np.float32)
    at = 0
    for ti, d in enumerate(durations.tolist()):
        d = int(d)
        if d <= 0:
            continue
        token_pos[at : at + d] = np.float32(ti) / np.float32(token_count)
        duration_pos[at : at + d] = (0.0 if d == 1
                                     else np.arange(d, dtype=np.float32) / np.float32(d - 1))
        at += d

    x = np.concatenate([x, np.stack([frame_pos, token_pos, duration_pos])], axis=0)
    fproj = w.front_q8("AC_FPROJ", H, H + 3)
    x = conv1d(x, fproj.weight, fproj.bias, 1)
    for b in range(m["NANO_AC_DEPTH"]):
        c0 = w.front_q8(f"AC_FB{b}_C0", H, H * K)
        c1 = w.front_q8(f"AC_FB{b}_C1", H, H * K)
        scale = float(w.front_f32(f"AC_FB{b}_SCALE", 1)[0])
        x = residual_block(x, c0, c1, scale, K)

    out = w.front_q8("AC_OUT", m["NANO_MELS"], H)
    return conv1d(x, out.weight, out.bias, 1)


def decoder_forward(w: NanoWeights, mel: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """mel-100 [100, T] + noise [4, T] -> complex spectrum [513, T]."""
    m = w.m
    dim, ek, dk = m["NANO_DIM"], m["NANO_EMBED_KERNEL"], m["NANO_DW_KERNEL"]

    embed = w.dec_q8("EMBED", dim, m["NANO_MELS"] * ek)
    x = conv1d(mel, embed.weight, embed.bias, ek)
    adapter = w.dec_q8("NOISE", dim, m["NANO_NOISE_CH"] * ek)
    x = x + conv1d(noise, adapter.weight, adapter.bias, ek)

    x = layer_norm(x, w.dec_f32("NORM_W", dim), w.dec_f32("NORM_B", dim))

    for b in range(m["NANO_BLOCKS"]):
        residual = x
        dw_w = w.dec_f32(f"B{b}_DW_W", dim * dk).reshape(dim, dk)
        dw_b = w.dec_f32(f"B{b}_DW_B", dim)
        # depthwise: one kernel per channel, groups == dim
        pad = dk // 2
        xp = np.pad(x, ((0, 0), (pad, pad)))
        h = np.zeros_like(x)
        for k in range(dk):
            h += dw_w[:, k : k + 1] * xp[:, k : k + x.shape[1]]
        h = h + dw_b[:, None]
        h = layer_norm(h, w.dec_f32(f"B{b}_NORM_W", dim), w.dec_f32(f"B{b}_NORM_B", dim))
        pw0 = w.dec_q8(f"B{b}_PW0", m["NANO_PW_HIDDEN"], dim)
        h = pw0.weight @ h + pw0.bias[:, None]
        h = gelu(h)
        pw1 = w.dec_q8(f"B{b}_PW1", dim, m["NANO_PW_HIDDEN"])
        h = pw1.weight @ h + pw1.bias[:, None]
        h = h * w.dec_f32(f"B{b}_GAMMA", dim)[:, None]
        x = (residual + h).astype(np.float32)

    x = layer_norm(x, w.dec_f32("FNORM_W", dim), w.dec_f32("FNORM_B", dim))
    head = w.dec_q8("HEAD", m["NANO_HEAD_OUT"], dim)
    out = head.weight @ x + head.bias[:, None]              # [1026, T]

    bins = m["NANO_BINS"]
    mag, phase = out[:bins], out[bins:]
    mag = np.exp(mag)
    # Bin 0 and Nyquist stay zeroed: the mag*(cos,sin) parametrisation
    # phase-collapses at bin 0, which is where the frame-DC artefact came from.
    mag[0] = 0.0
    mag[-1] = 0.0
    mag = np.clip(mag, None, 1e2)
    return mag * (np.cos(phase) + 1j * np.sin(phase))


def istft(spec: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Inverse STFT with a Hann window and centre padding, matching torch.

    torch.istft divides by the summed squared window rather than assuming the
    COLA condition, so a frame the window sums to zero over is not silently
    scaled wrong; the same normalisation is applied here.
    """
    bins, frames = spec.shape
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)  # periodic, as torch
    length = (frames - 1) * hop + n_fft
    out = np.zeros(length, dtype=np.float64)
    norm = np.zeros(length, dtype=np.float64)
    frames_td = np.fft.irfft(spec, n=n_fft, axis=0)            # [n_fft, frames]
    wsq = window * window
    for i in range(frames):
        start = i * hop
        out[start : start + n_fft] += frames_td[:, i] * window
        norm[start : start + n_fft] += wsq
    pad = n_fft // 2
    out = out[pad : length - pad]
    norm = norm[pad : length - pad]
    nonzero = norm > 1e-11
    out[nonzero] /= norm[nonzero]
    return out.astype(np.float32)


def dc_block(wav: np.ndarray) -> np.ndarray:
    """H(z) = (1 - z^-1)/(1 - R z^-1) as a truncated FIR, via FFT convolution."""
    impulse = np.zeros(DC_BLOCK_TAPS, dtype=np.float64)
    impulse[0] = 1.0
    n = np.arange(1, DC_BLOCK_TAPS, dtype=np.float64)
    impulse[1:] = (DC_BLOCK_R - 1.0) * DC_BLOCK_R ** (n - 1.0)
    samples = wav.size
    n_lin = samples + DC_BLOCK_TAPS - 1
    n_fft = 1 << (n_lin - 1).bit_length()
    filtered = np.fft.irfft(np.fft.rfft(wav.astype(np.float64), n_fft)
                            * np.fft.rfft(impulse, n_fft), n_fft)
    return filtered[:samples].astype(np.float32)


def synthesize_ids(front: bytes, dec: bytes, meta: dict[str, int],
                   ids: np.ndarray, *, seed: int,
                   durations: np.ndarray | None = None,
                   length_scale: float = 1.0) -> np.ndarray:
    """Phoneme ids -> waveform. `durations` overrides the duration model."""
    w = NanoWeights(front, dec, meta)
    ids = np.asarray(ids, dtype=np.int64)
    if durations is None:
        durations = duration_forward(w, ids, length_scale=length_scale)
    durations = np.asarray(durations, dtype=np.int64)
    mel = acoustic_forward(w, ids, durations)
    noise = seeded_noise(seed, meta["NANO_NOISE_CH"], int(mel.shape[1]))
    spec = decoder_forward(w, mel, noise)
    return dc_block(istft(spec, meta["NANO_N_FFT"], meta["NANO_HOP"]))
