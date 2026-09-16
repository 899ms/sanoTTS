#!/usr/bin/env python3
"""C-faithful simulation of the saanoTTS MCU int8 runtime.

Why this file exists
--------------------
Every embedded decision on this project is taken from a PyTorch *simulation*
of a quantisation scheme that is really implemented in C (``mcu/src/snt_tts.c``)
and really runs on silicon.  If the simulation is optimistic we ship a model
that fails the on-device gate; if it is pessimistic we throw away architectures
that would have worked.  So this module models the C arithmetic explicitly
rather than "roughly int8", and ``tools/validate_quant_sim_vs_r7.py`` checks it
against the shipped R7 blobs and the real C runtime.

What the C actually does (``mcu/src/snt_tts.c``)
-----------------------------------------------
* weights: per-output-channel symmetric int8, ``scale = max|row| / 127``,
  ``round-half-to-even`` (numpy/torch ``round`` in ``export_front_q8.py``),
  clipped to [-127, 127].  Bias and the residual ``scale`` parameter stay f32.
* activations: **per output column, over the whole gathered receptive field**.
  ``qkconv_col`` gathers ``in_ch * K`` floats for one output time step and calls
  ``quant_gather``, which takes ONE ``max|.| / 127`` scale across all of them.
  This is the single most important difference from a naive PyTorch fake-quant,
  which quantises the input tensor per *frame* (channels only) and then lets
  ``conv1d`` mix frames that were scaled differently.  Per-frame scaling is
  strictly finer than per-window scaling, so the naive simulation is
  *optimistic* for every k>1 convolution.
* rounding of activations: ``fast_round`` = round-half-**away-from-zero**
  (``(int)(y + copysign(0.5, y))``), not round-half-to-even.
* accumulation: plain int32, no saturation (``snt_kernels_ref.c``).
* dequantisation: ``out[o] = s_act * w_scale[o] * acc32[o] + bias[o]``, float.
* edges: ``qkconv_col`` gathers hard zeros outside [0, T), i.e. ordinary
  zero padding, and the zeros are inside the window the scale is taken over.
* residual: ``x[o][t] += block_scale * (dequantised conv1 output)``.

The activation scale is *dynamic* (recomputed per column from the live data);
no calibration table ships for the front half.  ``act_scales.h`` / ``frozen_norm.h``
belong to the fsd decoder, not to the acoustic front.

Modes
-----
``QuantConfig.act_scope``
    ``"window"``  - what the C does: one scale per output column over in_ch*K.
    ``"frame"``   - the naive PyTorch fake-quant: one scale per input frame.
    ``"none"``    - float activations (weight-only quantisation).

Everything else (per-tensor vs per-channel weights, asymmetric weights,
percentile clipping, int16 activations, per-layer bit overrides) is exposed so
remedies can be simulated without retraining.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------
# rounding primitives
# --------------------------------------------------------------------------

def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """C ``fast_round``: (int)(y + (y >= 0 ? 0.5 : -0.5))."""
    return torch.trunc(x + torch.where(x >= 0, 0.5, -0.5))


def round_half_even(x: torch.Tensor) -> torch.Tensor:
    """numpy/torch ``round``, used by ``export_front_q8.py`` for weights."""
    return torch.round(x)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantConfig:
    """One layer's quantisation policy."""

    # weights
    w_bits: int | None = 8          # None -> float weights
    w_per_channel: bool = True      # False -> one scale for the whole tensor
    w_symmetric: bool = True        # False -> affine (scale + zero point)
    w_clip_percentile: float | None = None  # e.g. 99.9; clip |w| before scaling
    # activations
    act_bits: int | None = 8        # None -> float activations
    act_scope: str = "window"       # "window" | "frame" | "none"
    act_round: str = "away"         # "away" (C fast_round) | "even" (torch.round)

    def describe(self) -> dict:
        return {
            "w_bits": self.w_bits,
            "w_per_channel": self.w_per_channel,
            "w_symmetric": self.w_symmetric,
            "w_clip_percentile": self.w_clip_percentile,
            "act_bits": self.act_bits,
            "act_scope": self.act_scope,
            "act_round": self.act_round,
        }


FLOAT = QuantConfig(w_bits=None, act_bits=None, act_scope="none")


# --------------------------------------------------------------------------
# weight quantisation
# --------------------------------------------------------------------------

def quantise_weight(w: torch.Tensor, cfg: QuantConfig) -> tuple[torch.Tensor, dict]:
    """Fake-quantise a Conv1d/Linear weight, returning the dequantised tensor.

    Rows are ``w.reshape(out_ch, -1)``: exactly the layout
    ``export_front_q8.py`` writes and ``snt_matvec_s8`` consumes.
    """
    if cfg.w_bits is None:
        return w.clone(), {"scheme": "float"}
    qmax = float((1 << (cfg.w_bits - 1)) - 1)          # 127 for int8
    flat = w.reshape(w.shape[0], -1).to(torch.float32)

    src = flat
    if cfg.w_clip_percentile is not None:
        pct = float(cfg.w_clip_percentile)
        if not 0.0 < pct <= 100.0:
            raise ValueError(f"w_clip_percentile must be in (0, 100], got {pct}")
        if cfg.w_per_channel:
            lim = torch.quantile(flat.abs(), pct / 100.0, dim=1, keepdim=True)
        else:
            lim = torch.quantile(flat.abs().reshape(-1), pct / 100.0).reshape(1, 1)
        lim = torch.clamp(lim, min=1e-12)
        src = torch.clamp(flat, -lim, lim)

    if cfg.w_symmetric:
        if cfg.w_per_channel:
            scale = src.abs().amax(dim=1, keepdim=True) / qmax
        else:
            scale = src.abs().amax().reshape(1, 1) / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = torch.clamp(round_half_even(src / scale), -qmax, qmax)
        deq = q * scale
        info = {"scheme": "symmetric", "scale": scale.detach().reshape(-1).clone()}
    else:
        # affine: q in [-2^(b-1), 2^(b-1)-1]
        qlo, qhi = -float(1 << (cfg.w_bits - 1)), qmax
        dim = 1 if cfg.w_per_channel else None
        if cfg.w_per_channel:
            wmin = src.amin(dim=1, keepdim=True)
            wmax = src.amax(dim=1, keepdim=True)
        else:
            wmin = src.amin().reshape(1, 1)
            wmax = src.amax().reshape(1, 1)
        span = torch.clamp(wmax - wmin, min=1e-12)
        scale = span / (qhi - qlo)
        zp = round_half_even(qlo - wmin / scale)
        q = torch.clamp(round_half_even(src / scale) + zp, qlo, qhi)
        deq = (q - zp) * scale
        info = {"scheme": "affine", "scale": scale.detach().reshape(-1).clone(),
                "zero_point": zp.detach().reshape(-1).clone()}
        del dim
    return deq.reshape(w.shape).to(w.dtype), info


# --------------------------------------------------------------------------
# activation quantisation
# --------------------------------------------------------------------------

def _act_qmax(bits: int) -> float:
    return float((1 << (bits - 1)) - 1)


def _rounder(mode: str):
    if mode == "away":
        return round_half_away
    if mode == "even":
        return round_half_even
    raise ValueError(f"unknown act_round {mode!r}")


def quantise_act_frame(x: torch.Tensor, bits: int, rounding: str = "away") -> torch.Tensor:
    """One symmetric scale per time frame, over channels. NOT what the C does.

    x: [B, C, T].  Kept so the naive simulation can be reproduced and compared.
    """
    qmax = _act_qmax(bits)
    scale = x.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.clamp(_rounder(rounding)(x / scale), -qmax, qmax) * scale


def conv1d_depthwise_window_quant(x: torch.Tensor, weight: torch.Tensor,
                                  bias: torch.Tensor | None, kernel: int, bits: int,
                                  rounding: str) -> torch.Tensor:
    """Depthwise conv with one activation scale per (output column, channel).

    A depthwise output channel only ever sees its own input channel, so the
    per-output-column analogue of ``quant_gather`` covers just that channel's
    K taps. The shipped R7 runtime keeps depthwise weights in f32 entirely
    (``Q8OFF_B*_DW_W_F32``, and the int16 dw chain is disabled), so this path
    exists only to price what quantising them *would* cost.
    """
    qmax = _act_qmax(bits)
    half = kernel // 2
    pad = torch.nn.functional.pad(x, (half, half))
    win = pad.unfold(dimension=2, size=kernel, step=1)          # [B, C, T, K]
    scale = win.abs().amax(dim=3, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    deq = torch.clamp(_rounder(rounding)(win / scale), -qmax, qmax) * scale
    out = (deq.to(torch.float64) * weight.reshape(1, -1, 1, kernel).to(torch.float64)).sum(dim=3)
    if bias is not None:
        out = out + bias.reshape(1, -1, 1).to(torch.float64)
    return out.to(x.dtype)


def conv1d_window_quant(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                        kernel: int, bits: int, rounding: str = "away") -> torch.Tensor:
    """conv1d with the C runtime's per-output-column window activation scale.

    Reproduces ``qkconv_col`` / ``q1x1_col``: for each output column the
    ``in_ch * K`` receptive field is gathered (zeros outside the sequence),
    a single ``max|.| / 127`` scale is taken over the whole gathered vector,
    every element is rounded half-away-from-zero, the int32 dot products are
    formed, and the result is dequantised as ``s_act * w_scale[o] * acc + b``.

    Because the weight tensor handed in is already fake-quantised (dequantised
    int8), multiplying dequantised activations by dequantised weights in float64
    is algebraically identical to ``s_act * w_scale[o] * acc32[o]`` up to float
    rounding, which is far below the int8 step.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [B, C, T] activations, got {tuple(x.shape)}")
    batch, in_ch, length = x.shape
    if batch != 1:
        raise ValueError("window-scope simulation runs one utterance at a time")
    if weight.shape[1] != in_ch:
        raise ValueError(f"weight in_ch {weight.shape[1]} != activation {in_ch}")
    if weight.shape[2] != kernel:
        raise ValueError(f"weight kernel {weight.shape[2]} != {kernel}")
    qmax = _act_qmax(bits)
    half = kernel // 2

    # gather every window at once: [T, in_ch, K] -> [T, in_ch*K] (in-major,
    # k-minor, matching S->gather[i * K + k])
    pad = torch.nn.functional.pad(x, (half, half))
    win = pad.unfold(dimension=2, size=kernel, step=1)      # [1, in_ch, T, K]
    win = win.squeeze(0).permute(1, 0, 2).reshape(length, in_ch * kernel)

    scale = win.abs().amax(dim=1, keepdim=True) / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    deq = torch.clamp(_rounder(rounding)(win / scale), -qmax, qmax) * scale

    wflat = weight.reshape(weight.shape[0], -1)             # [out_ch, in_ch*K]
    out = deq.to(torch.float64) @ wflat.to(torch.float64).t()   # [T, out_ch]
    if bias is not None:
        out = out + bias.to(torch.float64)
    return out.t().unsqueeze(0).to(x.dtype)


# --------------------------------------------------------------------------
# quantised module wrappers
# --------------------------------------------------------------------------

class QConv1d(nn.Module):
    """Drop-in for nn.Conv1d under a QuantConfig."""

    def __init__(self, base: nn.Conv1d, cfg: QuantConfig) -> None:
        super().__init__()
        if base.stride != (1,):
            raise NotImplementedError(f"stride {base.stride} not modelled")
        if base.dilation != (1,):
            raise NotImplementedError(f"dilation {base.dilation} not modelled")
        self.groups = int(base.groups)
        in_ch, out_ch = int(base.weight.shape[1]), int(base.weight.shape[0])
        if self.groups != 1 and not (self.groups == out_ch and in_ch == 1):
            raise NotImplementedError(
                f"only dense (groups=1) and depthwise (groups=out_ch) conv1d are "
                f"modelled; got groups={self.groups}, weight {tuple(base.weight.shape)}"
            )
        self.depthwise = self.groups != 1
        self.cfg = cfg
        self.kernel = int(base.weight.shape[2])
        expected_pad = self.kernel // 2
        if int(base.padding[0]) != expected_pad:
            raise NotImplementedError(
                f"padding {base.padding[0]} != same-padding {expected_pad}; "
                "the MCU gather assumes same padding"
            )
        wq, self.winfo = quantise_weight(base.weight.data.clone(), cfg)
        self.register_buffer("wq", wq)
        self.register_buffer("bias_f", None if base.bias is None else base.bias.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        pad = self.kernel // 2
        if cfg.act_bits is None or cfg.act_scope == "none":
            return nn.functional.conv1d(x, self.wq, self.bias_f, 1, pad, 1, self.groups)
        if cfg.act_scope == "window":
            if self.depthwise:
                return conv1d_depthwise_window_quant(x, self.wq, self.bias_f, self.kernel,
                                                     cfg.act_bits, cfg.act_round)
            return conv1d_window_quant(x, self.wq, self.bias_f, self.kernel,
                                       cfg.act_bits, cfg.act_round)
        if cfg.act_scope == "frame":
            xq = quantise_act_frame(x, cfg.act_bits, cfg.act_round)
            return nn.functional.conv1d(xq, self.wq, self.bias_f, 1, pad, 1, self.groups)
        raise ValueError(f"unknown act_scope {cfg.act_scope!r}")


class QLinear(nn.Module):
    """Drop-in for nn.Linear under a QuantConfig (window == frame for 1x1)."""

    def __init__(self, base: nn.Linear, cfg: QuantConfig) -> None:
        super().__init__()
        self.cfg = cfg
        wq, self.winfo = quantise_weight(base.weight.data.clone(), cfg)
        self.register_buffer("wq", wq)
        self.register_buffer("bias_f", None if base.bias is None else base.bias.data.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        if cfg.act_bits is None or cfg.act_scope == "none":
            return nn.functional.linear(x, self.wq, self.bias_f)
        qmax = _act_qmax(cfg.act_bits)
        scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        xq = torch.clamp(_rounder(cfg.act_round)(x / scale), -qmax, qmax) * scale
        return nn.functional.linear(xq, self.wq, self.bias_f)


# --------------------------------------------------------------------------
# model surgery
# --------------------------------------------------------------------------

def named_quantisable(model: nn.Module, prefix: str = "") -> list[tuple[str, nn.Module]]:
    """Every Conv1d / Linear in the model, with a dotted name."""
    found: list[tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        path = f"{prefix}{name}"
        if isinstance(child, (nn.Conv1d, nn.Linear)):
            found.append((path, child))
        else:
            found.extend(named_quantisable(child, prefix=f"{path}."))
    return found


PolicyFn = Callable[[str, nn.Module], QuantConfig]


def apply_quantisation(model: nn.Module, policy: PolicyFn, prefix: str = "") -> list[str]:
    """Replace every Conv1d/Linear with its quantised twin. Mutates ``model``.

    ``policy(name, module)`` returns the QuantConfig for that layer; returning
    ``FLOAT`` leaves it in float (the module is still wrapped, so the layer list
    stays stable, but no rounding happens).
    """
    touched: list[str] = []
    for name, child in list(model.named_children()):
        path = f"{prefix}{name}"
        if isinstance(child, nn.Conv1d):
            cfg = policy(path, child)
            setattr(model, name, QConv1d(child, cfg))
            touched.append(path)
        elif isinstance(child, nn.Linear):
            cfg = policy(path, child)
            setattr(model, name, QLinear(child, cfg))
            touched.append(path)
        else:
            touched.extend(apply_quantisation(child, policy, prefix=f"{path}."))
    return touched


def uniform_policy(cfg: QuantConfig) -> PolicyFn:
    def _policy(name: str, module: nn.Module) -> QuantConfig:
        del name, module
        return cfg
    return _policy


def override_policy(default: QuantConfig, overrides: dict[str, QuantConfig]) -> PolicyFn:
    def _policy(name: str, module: nn.Module) -> QuantConfig:
        del module
        return overrides.get(name, default)
    return _policy


def only_policy(names: Iterable[str], cfg: QuantConfig) -> PolicyFn:
    wanted = set(names)

    def _policy(name: str, module: nn.Module) -> QuantConfig:
        del module
        return cfg if name in wanted else FLOAT
    return _policy


# --------------------------------------------------------------------------
# statistics + metrics
# --------------------------------------------------------------------------

def weight_stats(w: torch.Tensor) -> dict:
    """Outlier / dynamic-range diagnostics on the export row layout."""
    flat = w.reshape(w.shape[0], -1).to(torch.float64)
    a = flat.abs()
    row_max = a.amax(dim=1)
    row_rms = torch.sqrt((flat * flat).mean(dim=1))
    crest = row_max / torch.clamp(row_rms, min=1e-30)
    v = flat.reshape(-1)
    mu, sd = v.mean(), v.std()
    kurt = float((((v - mu) / torch.clamp(sd, min=1e-30)) ** 4).mean())
    # how much of a row's int8 range the *second* largest magnitude reaches:
    # 127 * (2nd max / max). A low number means one weight owns the scale.
    srt = a.sort(dim=1, descending=True).values
    second = srt[:, 1] if srt.shape[1] > 1 else srt[:, 0]
    occupancy = 127.0 * (second / torch.clamp(row_max, min=1e-30))
    # effective int8 levels used per row: 127 * rms / max
    levels = 127.0 * (row_rms / torch.clamp(row_max, min=1e-30))
    return {
        "shape": list(w.shape),
        "numel": int(w.numel()),
        "abs_max": float(a.max()),
        "std": float(sd),
        "kurtosis": kurt,
        "crest_max": float(crest.max()),
        "crest_mean": float(crest.mean()),
        "row_levels_min": float(levels.min()),
        "row_levels_mean": float(levels.mean()),
        "second_max_int8_min": float(occupancy.min()),
        "worst_row": int(torch.argmax(crest)),
        "per_tensor_vs_per_channel_range": float(a.max() / torch.clamp(row_max.min(), min=1e-30)),
    }


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, same estimator as ``mcu/test/golden_main.c``."""
    n = min(a.size, b.size)
    x = a.reshape(-1)[:n].astype(np.float64)
    y = b.reshape(-1)[:n].astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / den) if den > 0 else 0.0


def rel_rms(ref: np.ndarray, got: np.ndarray) -> float:
    r = ref.reshape(-1).astype(np.float64)
    g = got.reshape(-1).astype(np.float64)
    n = min(r.size, g.size)
    r, g = r[:n], g[:n]
    denom = math.sqrt(float((r * r).mean()))
    if denom <= 0:
        return float("inf")
    return math.sqrt(float(((r - g) ** 2).mean())) / denom


# --------------------------------------------------------------------------
# SmoothQuant-style residual-stream rescaling (no retraining)
# --------------------------------------------------------------------------

def smooth_residual_stream(model: nn.Module, scale: torch.Tensor) -> None:
    """Rescale the acoustic student's 44/48-channel residual stream in place.

    The stream is exactly linear everywhere it is touched: ``frame_input_proj``
    writes it, each ``ResidualConvBlock`` reads it with ``net[0]`` and adds back
    through ``net[2]``, and ``output`` reads it.  SiLU sits *inside* a block,
    between net[0] and net[2], in a different channel space, so it is never
    crossed.  Multiplying stream channel c by ``1/s[c]`` and compensating the
    reading/writing weights is therefore an exact identity on the float model,
    while changing what the per-channel int8 grids have to cover.

    Applied to both ``token_*`` and ``frame_*`` halves; ``scale`` is [hidden].
    """
    hidden = int(scale.numel())
    inv = 1.0 / scale

    def scale_out(conv: nn.Conv1d) -> None:
        if conv.weight.shape[0] != hidden:
            raise ValueError(f"out_ch {conv.weight.shape[0]} != hidden {hidden}")
        conv.weight.data.mul_(inv.reshape(-1, 1, 1))
        if conv.bias is not None:
            conv.bias.data.mul_(inv)

    def scale_in(conv: nn.Conv1d) -> None:
        if conv.weight.shape[1] != hidden:
            raise ValueError(f"in_ch {conv.weight.shape[1]} != hidden {hidden}")
        conv.weight.data.mul_(scale.reshape(1, -1, 1))

    def scale_in_prefix(conv: nn.Conv1d) -> None:
        """Scale only the first `hidden` input channels (stream + side features)."""
        if conv.weight.shape[1] < hidden:
            raise ValueError(f"in_ch {conv.weight.shape[1]} < hidden {hidden}")
        conv.weight.data[:, :hidden, :].mul_(scale.reshape(1, -1, 1))

    for proj_name, blocks_name in (("token_input_proj", "token_blocks"),
                                   ("frame_input_proj", "frame_blocks")):
        proj = getattr(model, proj_name, None)
        blocks = getattr(model, blocks_name, None)
        if proj is None or blocks is None:
            raise AttributeError(f"model has no {proj_name}/{blocks_name}")
        scale_out(proj)
        for block in blocks:
            scale_in(block.net[0])    # reads the stream
            scale_out(block.net[2])   # writes back into the stream

    # The token stream leaves token_blocks, is repeat_interleaved to frames and
    # becomes the first `hidden` input channels of frame_input_proj; the three
    # positional features occupy the rest and must NOT be rescaled.
    scale_in_prefix(model.frame_input_proj)
    out = getattr(model, "output", None)
    if out is None:
        raise AttributeError("model has no output conv")
    # the frame stream's only other reader
    scale_in(out)


def smoothquant_scale(act_absmax: torch.Tensor, weight_in_absmax: torch.Tensor,
                      alpha: float) -> torch.Tensor:
    """s_c = act_absmax[c]^alpha / weight_absmax[c]^(1-alpha), SmoothQuant eq. 4."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    a = torch.clamp(act_absmax.to(torch.float64), min=1e-8)
    w = torch.clamp(weight_in_absmax.to(torch.float64), min=1e-8)
    s = (a ** alpha) / (w ** (1.0 - alpha))
    s = s / s.mean()          # keep the overall magnitude put
    return s.to(torch.float32)


__all__ = [
    "QuantConfig", "FLOAT", "QConv1d", "QLinear",
    "quantise_weight", "quantise_act_frame", "conv1d_window_quant",
    "apply_quantisation", "named_quantisable",
    "uniform_policy", "override_policy", "only_policy",
    "weight_stats", "correlation", "rel_rms",
    "smooth_residual_stream", "smoothquant_scale",
    "round_half_away", "round_half_even", "replace", "field",
]
