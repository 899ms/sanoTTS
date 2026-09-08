#!/usr/bin/env python3
"""Straight-through-estimator QAT primitives + Dynamic Tanh, C-runtime-faithful.

Purpose (E13 substitution/reallocation experiment, registered in
experiments/substitution-reallocation-sub300k-20260823.json): train the tiny
decoder and the acoustic student with fake int8 quantisation IN the forward
pass, matching the arithmetic the shipped C runtime really performs, so the
post-hoc int8 gate (min waveform correlation >= 0.98 under
tools/mcu_quant_sim.py act_scope="window") is trained for rather than hoped
for. E12's post-hoc-quantised acoustic failed that gate at 0.9758
(experiments/evidence/e12-nano-int8-quantisation-20260822.json).

Scheme, mirrored from tools/mcu_quant_sim.py (the C-faithful simulator for
mcu/src/snt_tts.c) -- NOT "roughly int8":

* weights: per-output-channel symmetric int8 over the export row layout
  w.reshape(out_ch, -1); scale = max|row| / 127; round-half-to-even
  (numpy/torch round, exactly what export_front_q8.py does); clip [-127, 127].
* activations: int8 with ONE scale per output column taken over the WHOLE
  gathered receptive field (in_ch * K floats, hard zeros gathered outside
  [0, T) included in the window the scale is taken over), round-half-AWAY-
  from-zero (C fast_round). This is the C `qkconv_col`/`quant_gather` scheme;
  per-frame scaling would be strictly finer and therefore optimistic.
* bias stays f32; accumulation is exact (float here, int32-no-saturation in C;
  the difference is far below the int8 step).
* depthwise convs are NOT quantised: the shipped R7 C runtime keeps depthwise
  weights f32 (Q8OFF_B*_DW_W_F32; the int16 dw chain is disabled), so a
  faithful QAT leaves them float too.

The straight-through estimator (y = x + (fq(x) - x).detach()) is the only
addition over the simulator; tools/mcu_quant_sim.py remains the measurement
instrument (no STE, batch=1, float64 accumulation) and is deliberately not
imported here so measurement and training code cannot drift into each other.

DyT (Dynamic Tanh) is from Zhu, Chen, LeCun, Liu et al. 2025, "Transformers
without Normalization" (arXiv:2503.10622): y = gamma * tanh(alpha * x) + beta.
This project's registered variant makes alpha per-channel (the paper's default
is a scalar alpha; the deviation is declared in the registration).
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

QMAX = 127.0


def round_half_away(x: torch.Tensor) -> torch.Tensor:
    """C ``fast_round``: (int)(y + (y >= 0 ? 0.5 : -0.5))."""
    return torch.trunc(x + torch.where(x >= 0, 0.5, -0.5))


def _ste(x: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: forward xq, gradient of identity."""
    return x + (xq - x).detach()


def fake_quant_weight_per_out_channel(w: torch.Tensor) -> torch.Tensor:
    """Per-output-channel symmetric int8 weight fake-quant with STE.

    Row layout w.reshape(out_ch, -1) = exactly what export_front_q8.py writes
    and snt_matvec_s8 consumes. Rounding is half-to-even (torch.round), the
    export script's rounding.
    """
    if w.ndim < 2:
        raise ValueError(f"expected weight with >= 2 dims, got {tuple(w.shape)}")
    flat = w.reshape(w.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(flat / scale), -QMAX, QMAX)
    return _ste(w, (q * scale).reshape(w.shape))


def fake_quant_act_last_dim(x: torch.Tensor) -> torch.Tensor:
    """One symmetric int8 scale over the LAST dim, round-half-away, STE.

    For a gathered window [.., in_ch*K] this is the C per-output-column
    ``quant_gather`` scale; for a Linear input [.., features] window == frame
    (K=1), matching mcu_quant_sim.QLinear.
    """
    scale = x.abs().amax(dim=-1, keepdim=True) / QMAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(round_half_away(x / scale), -QMAX, QMAX)
    return _ste(x, q * scale)


def qat_linear(x: torch.Tensor, weight: torch.Tensor,
               bias: torch.Tensor | None) -> torch.Tensor:
    """nn.functional.linear under the C int8 scheme with STE."""
    return F.linear(fake_quant_act_last_dim(x),
                    fake_quant_weight_per_out_channel(weight), bias)


def qat_conv1d_dense_window(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor | None) -> torch.Tensor:
    """Dense (groups=1) same-padding stride-1 Conv1d under the C scheme, batched.

    x [B, C, T], weight [O, C, K] -> [B, O, T]. The window is gathered
    in-channel-major / kernel-minor, exactly the C ``gather[i * K + k]``
    layout and exactly ``weight.reshape(O, -1)``; zero padding lands INSIDE
    the window the activation scale is taken over, as in ``qkconv_col``.
    """
    if x.ndim != 3:
        raise ValueError(f"expected [B, C, T] activations, got {tuple(x.shape)}")
    if weight.ndim != 3:
        raise ValueError(f"expected [O, C, K] weight, got {tuple(weight.shape)}")
    if weight.shape[1] != x.shape[1]:
        raise ValueError(f"weight in_ch {weight.shape[1]} != activation {x.shape[1]}")
    out_ch, in_ch, kernel = weight.shape
    half = kernel // 2
    pad = F.pad(x, (half, half))
    win = pad.unfold(dimension=2, size=kernel, step=1)          # [B, C, T, K]
    win = win.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], in_ch * kernel)
    win = fake_quant_act_last_dim(win)                          # scale per (b, t)
    wq = fake_quant_weight_per_out_channel(weight).reshape(out_ch, -1)
    out = torch.matmul(win, wq.t())                             # [B, T, O]
    if bias is not None:
        out = out + bias
    return out.transpose(1, 2)


def enable_dense_conv1d_qat(model: nn.Module) -> list[str]:
    """Monkeypatch every dense Conv1d forward in ``model`` to the QAT path.

    state_dict keys are untouched, so checkpoints stay loadable by every
    existing renderer/export tool. Refuses (loudly) any conv the C runtime
    scheme does not cover instead of silently skipping it.
    """
    touched: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv1d):
            continue
        if module.groups != 1:
            raise NotImplementedError(
                f"{name}: groups={module.groups}; only dense conv1d is quantised "
                f"(depthwise stays f32 in the shipped runtime and is handled by "
                f"the caller, not this helper)"
            )
        if module.stride != (1,) or module.dilation != (1,):
            raise NotImplementedError(
                f"{name}: stride={module.stride} dilation={module.dilation} not modelled"
            )
        if int(module.padding[0]) != int(module.kernel_size[0]) // 2:
            raise NotImplementedError(
                f"{name}: padding {module.padding[0]} != same-padding "
                f"{int(module.kernel_size[0]) // 2}; the MCU gather assumes same padding"
            )

        def qat_forward(self: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
            return qat_conv1d_dense_window(x, self.weight, self.bias)

        module.forward = types.MethodType(qat_forward, module)
        touched.append(name)
    if not touched:
        raise RuntimeError("enable_dense_conv1d_qat found no Conv1d to quantise")
    return touched


class DyT(nn.Module):
    """Dynamic Tanh (Zhu et al. 2025, arXiv:2503.10622): y = g * tanh(a*x) + b.

    Drop-in for a channel-last nn.LayerNorm(dim). Parameters are named
    weight/bias like LayerNorm plus the extra per-channel alpha, so the
    checkpoint-walk audit counts 3*dim per site (LayerNorm: 2*dim). alpha is
    PER-CHANNEL here (registered deviation from the paper's scalar default);
    alpha_init follows the paper's non-attention default 0.5.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"DyT dim must be > 0, got {dim}")
        self.alpha = nn.Parameter(torch.full((dim,), float(alpha_init)))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.numel():
            raise RuntimeError(
                f"DyT expects channel-last dim {self.weight.numel()}, got {tuple(x.shape)}"
            )
        return self.weight * torch.tanh(self.alpha * x) + self.bias


__all__ = [
    "QMAX", "round_half_away",
    "fake_quant_weight_per_out_channel", "fake_quant_act_last_dim",
    "qat_linear", "qat_conv1d_dense_window", "enable_dense_conv1d_qat",
    "DyT",
]
