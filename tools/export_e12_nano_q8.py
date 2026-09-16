#!/usr/bin/env python3
"""Export the E12-nano stack as int8 blobs + a generated C offset header.

This is the nano lineage's counterpart to ``esp32c3/fsd/export_front_q8.py`` /
``export_fsd_q8.py`` (the R7 exporters) and ``tools/export_piperlite_q8.py``.
It is a genuine rewrite rather than a parametrisation of those: R7's decoder is
FiLM-conditioned with a low-rank head and frozen GroupNorm statistics, the nano
decoder is a ConvNeXt1D trunk with dynamic per-frame LayerNorm, LayerScale, a
dense k7 mel embedding, a 4-channel noise adapter and a 1026-wide dense head.

WHAT IS WRITTEN
---------------
``front_q8.bin``   duration student (22,858 params) + acoustic student (128,102)
``model_q8.bin``   TinyVocos decoder (143,682)
                   (``--weights f32`` writes ``front_f32.bin`` / ``model_f32.bin``:
                   the same regions with float32 rows and unit scales, for the
                   browser build; see NANO_WEIGHT_FORMAT in the header)
``nano_q8_meta.h`` every shape constant and every byte offset, so the runtime
                   parses nothing and copies nothing out of flash.
``golden/``        one utterance's phoneme ids, frozen durations, the seeded
                   decoder noise and the float reference waveform.
``export-report.json``

REGION LAYOUT (identical discipline to R7)
-----------------------------------------
Every quantised layer is three consecutive 16-byte-aligned regions::

    <NAME>_W8      int8  [out_ch, n16]   n16 = ceil16(in_flat)
    <NAME>_SCALE   f32   [out_ch]
    <NAME>_BIAS    f32   [out_ch]

and every unquantised tensor is one region ``<NAME>_F32``.  The ``n16-in_flat``
trailing columns of each ``_W8`` region are zeros so the SIMD kernels can read
16-byte groups; they are shipped bytes, not parameters (see
``tools/audit_e12_nano_parameters.py``).

QUANTISATION
------------
Weights: per-output-channel symmetric int8, ``scale = max|row|/127``, rounded
half-to-even (``numpy.round``), clipped to [-127, 127].  Rows are
``w.reshape(out_ch, -1)`` -- exactly the layout ``snt_matvec_s8`` consumes.

Activations are NOT calibrated here.  The runtime quantises them dynamically,
one scale per output column over the whole gathered ``in_ch * K`` receptive
field, rounding half-away-from-zero.  ``tools/mcu_quant_sim.py`` with
``act_scope="window"`` is the validated simulation of that scheme.

Depthwise kernels stay f32 (as in R7: ``Q8OFF_B*_DW_W_F32``).  LayerNorm
weights, LayerScale gammas, embeddings and residual scales stay f32.

E13 (experiments/e13-substitution-reallocation-sub300k-20260823.json)
---------------------------------------------------------------------
The decoder checkpoint's config may carry ``norm_type`` (``layernorm`` |
``dyt``) and ``act_type`` (``gelu`` | ``relu``); absent keys mean the E12
operators.  An E12 checkpoint exports the same blob bytes as before; only the
generated header gains ``NANO_NORM_TYPE 0`` / ``NANO_ACT_TYPE 0``, which the
runtime already assumed.

DyT (y = gamma * tanh(alpha * x) + beta, per-channel alpha) adds ONE f32
region per norm site, appended directly after that site's existing affine
pair -- the same R7 f32-region convention the LayerNorm affines already use::

    <SITE>_W_F32   gamma   (same region name as the LayerNorm weight)
    <SITE>_B_F32   beta    (same region name as the LayerNorm bias)
    <SITE>_A_F32   alpha   (DyT only; absent from LayerNorm exports)

so a DyT decoder blob is the E12 layout with three extra 192/248-byte f32
regions (stem, per-block, final) and shifted offsets -- which is free, because
the runtime reads every offset from the generated header.  The header also
gains ``NANO_NORM_TYPE`` / ``NANO_ACT_TYPE`` (0 = E12 operator, 1 =
substituted); pre-E13 headers simply lack them and the runtime defaults to 0.
``act_type`` changes no bytes in the blob (ReLU has no parameters).

SMOOTHQUANT
-----------
``--smoothquant-alpha`` folds a per-channel rescaling of the acoustic student's
44-channel residual stream into the float weights before quantisation.  The
transform is an exact identity on the float model (the stream is linear
everywhere it is read or written; SiLU sits inside a block in a different
channel space) and it is free at inference: nothing in the C runtime knows it
happened.  ``tools/analyse_e12_nano_int8.py`` measured it as the only post-hoc
remedy that helps materially -- but under a different decoder noise draw than
the production renderer uses.  Under the shipped seeding it costs 0.005 of
minimum correlation, so the default is 0 (off).  See
``tools/validate_nano_c_vs_sim.py`` and
``docs/e12-nano-embedded-pipeline.md``.

Run on k2 (the only machine that may load models):

    nice -n 15 venv/bin/python tools/export_e12_nano_q8.py \\
        --base artifacts/kokoro-corpus-af_heart-20260821 \\
        --decoder-checkpoint .../checkpoint_step225000.pt \\
        --out /tmp/nano-export
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]

MAGIC_FRONT = 0x4E414E46  # 'NANF'
MAGIC_DEC = 0x4E414E44    # 'NAND'
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# module loading (the training scripts are the single source of architecture)
# ---------------------------------------------------------------------------

def _find(*candidates: pathlib.Path) -> pathlib.Path:
    """First existing candidate. The campaign-e8 trainers live under
    ``tools/campaign-e8/`` in the repo and at ``tools/`` on the training host;
    guessing wrong silently imports a different architecture, so resolve it."""
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit("none of these exist: " + ", ".join(str(c) for c in candidates))


def _import(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# blob writer
# ---------------------------------------------------------------------------

def n16(x: int) -> int:
    return (x + 15) // 16 * 16


def quantise_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric int8, bit-identical to mcu_quant_sim.

    The arithmetic is deliberately done in torch float32 with ``torch.round``
    (half-to-even), because ``tools/mcu_quant_sim.py`` -- the simulator every
    quantisation decision on this project is taken from -- does exactly that.
    Doing the same division in float64 or with ``numpy.round`` would agree on
    almost every weight and disagree on the ties, and "almost" is not a
    reproducible export.
    """
    t = torch.as_tensor(np.asarray(w), dtype=torch.float32)
    flat = t.reshape(t.shape[0], -1)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.clamp(torch.round(flat / scale), -127.0, 127.0)
    return (q.numpy().astype(np.int8),
            scale.reshape(-1).numpy().astype(np.float32))


class Blob:
    """16-byte-aligned region writer that emits its own offset defines."""

    def __init__(self, prefix: str, weights: str = "int8") -> None:
        if weights not in ("int8", "f32"):
            raise ValueError(f"unsupported weight format {weights!r}")
        self.prefix = prefix
        self.weights = weights
        self.buf = bytearray()
        self.defines: list[tuple[str, int]] = []
        self.regions: list[dict] = []
        self.source: dict[str, np.ndarray] = {}

    def _align(self) -> None:
        pad = (-len(self.buf)) % 16
        if pad:
            self.buf.extend(b"\0" * pad)

    def _emit(self, name: str, payload: bytes, kind: str, meta: dict) -> int:
        self._align()
        off = len(self.buf)
        self.defines.append((f"{self.prefix}{name}", off))
        self.buf.extend(payload)
        rec = {"region": name, "offset": off, "bytes": len(payload), "kind": kind}
        rec.update(meta)
        self.regions.append(rec)
        return off

    def add_q8(self, name: str, w: np.ndarray, bias: np.ndarray) -> None:
        """Per-output-channel symmetric int8 weights + f32 scales + f32 bias."""
        if w.ndim < 2:
            raise ValueError(f"{name}: weight must be at least 2-D, got {w.shape}")
        out_ch = int(w.shape[0])
        in_flat = int(w.reshape(out_ch, -1).shape[1])
        if bias.size != out_ch:
            raise ValueError(f"{name}: bias {bias.size} != out_ch {out_ch}")
        zero_rows = int((np.abs(w.reshape(out_ch, -1)).max(axis=1) == 0.0).sum())
        if self.weights == "f32":
            # Float weights in the SAME [out_ch, n16] row layout the int8 path
            # uses, with unit scales, so the generated header and every
            # runtime offset/shape macro are unchanged; only the element type
            # (and therefore the byte offsets, which the header carries) differ.
            # The region keeps the _W8 name so NOFF_/DOFF_ macros stay stable.
            padded = np.zeros((out_ch, n16(in_flat)), dtype="<f4")
            padded[:, :in_flat] = np.asarray(w, dtype=np.float32).reshape(out_ch, -1)
            scales = np.ones(out_ch, dtype=np.float32)
            self._emit(f"{name}_W8", padded.tobytes(), "f32_weight",
                       {"in_flat": in_flat, "n16": n16(in_flat), "out_ch": out_ch,
                        "zero_rows": zero_rows})
        else:
            q, scales = quantise_rows(w)
            padded = np.zeros((out_ch, n16(in_flat)), dtype=np.int8)
            padded[:, :in_flat] = q
            self._emit(f"{name}_W8", padded.tobytes(), "q8_weight",
                       {"in_flat": in_flat, "n16": n16(in_flat), "out_ch": out_ch,
                        "zero_rows": zero_rows})
        self.source[f"{name}_W8"] = np.asarray(w)
        self._emit(f"{name}_SCALE", scales.astype("<f4").tobytes(), "scales",
                   {"count": out_ch})
        self._emit(f"{name}_BIAS", bias.reshape(-1).astype("<f4").tobytes(),
                   "bias", {"count": out_ch})
        self.source[f"{name}_BIAS"] = np.asarray(bias).reshape(-1)

    def add_f32(self, name: str, t: np.ndarray) -> None:
        self._emit(f"{name}_F32", t.reshape(-1).astype("<f4").tobytes(), "f32",
                   {"count": int(t.size), "shape": list(t.shape)})
        self.source[f"{name}_F32"] = np.asarray(t)

    def finish(self, total_name: str) -> None:
        self._align()
        self.defines.append((total_name, len(self.buf)))


# ---------------------------------------------------------------------------
# SmoothQuant on the acoustic residual stream
# ---------------------------------------------------------------------------

def collect_stream_absmax(acoustic, latent_mod, rows, device) -> torch.Tensor:
    """max |activation| per residual-stream channel, over the calibration rows.

    The stream is what ``frame_input_proj`` writes and every ``frame_blocks[i]
    .net[0]`` plus ``output`` reads; the token half writes/reads the same
    channel space.  One vector covers both halves because
    ``smooth_residual_stream`` rescales both.
    """
    hidden = int(acoustic.frame_input_proj.weight.shape[0])
    absmax = torch.zeros(hidden, dtype=torch.float64)
    captured: list[torch.Tensor] = []

    def hook(_mod, inp):
        captured.append(inp[0].detach())

    # every point that READS the stream: each residual block's first conv and
    # the output projection. That is exactly the set SmoothQuant rebalances.
    handles = [acoustic.output.register_forward_pre_hook(hook)]
    for block in list(acoustic.token_blocks) + list(acoustic.frame_blocks):
        handles.append(block.net[0].register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for ids, durs in rows:
                captured.clear()
                _acoustic_mel(latent_mod, acoustic, ids, durs, device)
                for t in captured:
                    if t.ndim != 3 or int(t.shape[0]) != 1 or int(t.shape[1]) != hidden:
                        raise RuntimeError(
                            f"unexpected stream activation shape {tuple(t.shape)}, "
                            f"expected [1, {hidden}, T]")
                    v = t.abs().squeeze(0).amax(dim=1).to(torch.float64)
                    absmax = torch.maximum(absmax, v)
    finally:
        for h in handles:
            h.remove()
    if float(absmax.max()) <= 0.0:
        raise RuntimeError("calibration produced an all-zero activation range")
    return absmax.to(torch.float32)


def stream_weight_absmax(acoustic) -> torch.Tensor:
    """max |w| per stream channel over every weight that READS the stream."""
    hidden = int(acoustic.frame_input_proj.weight.shape[0])
    acc = torch.zeros(hidden, dtype=torch.float64)
    readers = [b.net[0] for b in acoustic.token_blocks]
    readers += [b.net[0] for b in acoustic.frame_blocks]
    readers.append(acoustic.output)
    for conv in readers:
        w = conv.weight.detach().to(torch.float64)      # [out, in, K]
        acc = torch.maximum(acc, w.abs().amax(dim=(0, 2)))
    # frame_input_proj reads the stream in its first `hidden` input channels
    w = acoustic.frame_input_proj.weight.detach().to(torch.float64)
    acc = torch.maximum(acc, w[:, :hidden, :].abs().amax(dim=(0, 2)))
    return acc.to(torch.float32)


# ---------------------------------------------------------------------------
# golden vectors
# ---------------------------------------------------------------------------

def seeded_noise(row_id: str, channels: int, frames: int) -> torch.Tensor:
    """Exactly ``render_fullstack_tiny.seeded_noise``."""
    seed = int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest()[:8], "big")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(1, channels, frames, generator=generator)


def noise_seed_u64(row_id: str) -> int:
    return int.from_bytes(hashlib.sha256(row_id.encode("utf-8")).digest()[:8], "big")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=pathlib.Path, required=True,
                    help="artifacts/kokoro-corpus-af_heart-20260821")
    ap.add_argument("--decoder-checkpoint", type=pathlib.Path, required=True)
    ap.add_argument("--duration-checkpoint", type=pathlib.Path, default=None)
    ap.add_argument("--acoustic-checkpoint", type=pathlib.Path, default=None)
    ap.add_argument("--pack", type=pathlib.Path, default=None,
                    help="pack dir with rows.json (default: <base>/packs-regen/eval8)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--golden-rows", type=int, default=8,
                    help="pack rows written to the golden fixture; the gate is "
                         "the MINIMUM correlation over them")
    ap.add_argument("--calib-rows", type=int, default=8,
                    help="rows used for the SmoothQuant activation range")
    ap.add_argument("--smoothquant-alpha", type=float, default=0.0,
                    help="per-channel rescaling of the acoustic residual stream, "
                         "exact on the float model. DEFAULT 0 (off): measured "
                         "under the production sha256(row_id) noise seeding it "
                         "makes the shipped stack WORSE (min corr 0.9797 vs "
                         "0.9843 without). It was recommended on the strength of "
                         "a measurement taken under a different noise draw; see "
                         "docs/e12-nano-embedded-pipeline.md.")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--weights", choices=("int8", "f32"), default="int8",
                    help="int8: the device format (per-output-channel symmetric "
                         "int8, the MCU gates' subject). f32: the same regions "
                         "and header with float32 rows and unit scales, for the "
                         "browser build (SNT_NANO_W_F32), where the 2.27M "
                         "release stack does not survive int8 (min corr 0.951).")
    args = ap.parse_args()

    tools = REPO / "tools"
    latent_mod = _import("nano_latent", tools / "train_roota_piper_latent_student.py")
    dur_mod = _import("nano_duration", tools / "train_roota_piper_duration_student.py")
    tiny_mod = _import("nano_tiny", _find(tools / "campaign-e8" / "train_tiny_vocos_student.py",
                                          tools / "train_tiny_vocos_student.py"))
    sys.path.insert(0, str(tools))
    import mcu_quant_sim as Q  # noqa: E402

    device = torch.device("cpu")
    base = args.base
    dur_ckpt = args.duration_checkpoint or base / "runs/e12-nano-duration/duration-student.pt"
    ac_ckpt = args.acoustic_checkpoint or base / "runs/e12-nano-acoustic/latent-student.pt"
    pack = args.pack or base / "packs-regen/eval8"

    acoustic, ac_cfg = latent_mod.load_model_from_checkpoint(ac_ckpt, device)
    acoustic.eval()
    dur, dur_cfg = dur_mod.load_model_from_checkpoint(dur_ckpt, device)
    dur.eval()

    payload = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=True)
    dec_cfg = payload["config"]
    # E13 operator substitutions: the trainers write these config keys; their
    # absence means the E12 operators. Passed only when non-default, so an E12
    # export against a pre-E13 trainer module still constructs.
    norm_type = str(dec_cfg.get("norm_type", "layernorm"))
    act_type = str(dec_cfg.get("act_type", "gelu"))
    if norm_type not in ("layernorm", "dyt"):
        raise SystemExit(f"unsupported norm_type {norm_type!r} in decoder config")
    if act_type not in ("gelu", "relu"):
        raise SystemExit(f"unsupported act_type {act_type!r} in decoder config")
    op_kwargs = {}
    if norm_type != "layernorm":
        op_kwargs["norm_type"] = norm_type
    if act_type != "gelu":
        op_kwargs["act_type"] = act_type
    decoder = tiny_mod.TinyVocosStudent(
        n_mels=dec_cfg["n_mels"], dim=dec_cfg["width"],
        num_layers=dec_cfg["num_layers"], expand=dec_cfg["expand"],
        n_fft=dec_cfg["n_fft"], noise_channels=dec_cfg["noise_channels"],
        **op_kwargs)
    decoder.load_state_dict(payload["model"])
    decoder.eval()
    is_dyt = norm_type == "dyt"

    # ---- shapes, asserted rather than assumed --------------------------
    DH = int(dur_cfg["hidden"])
    DDEPTH = int(dur_cfg["depth"])
    DK = int(dur_cfg["kernel_size"])
    VOCAB = int(dur_cfg["vocab_size"])
    MAX_TOKENS = int(dur_cfg["max_tokens"])
    MAX_DURATION = int(dur_cfg["max_duration"])

    AH = int(ac_cfg["hidden"])
    ATD = int(ac_cfg["token_depth"])
    AD = int(ac_cfg["depth"])
    AK = int(ac_cfg["kernel_size"])
    AOUT = int(ac_cfg["out_channels"])
    if int(ac_cfg["vocab_size"]) != VOCAB:
        raise SystemExit(f"vocab mismatch: duration {VOCAB} vs acoustic {ac_cfg['vocab_size']}")

    DIM = int(dec_cfg["width"])
    LAYERS = int(dec_cfg["num_layers"])
    HID = DIM * int(dec_cfg["expand"])
    NMELS = int(dec_cfg["n_mels"])
    NFFT = int(dec_cfg["n_fft"])
    NCH = int(dec_cfg["noise_channels"])
    if NMELS != AOUT:
        raise SystemExit(f"interface mismatch: acoustic emits {AOUT}, decoder wants {NMELS}")
    BINS = NFFT // 2 + 1
    if int(decoder.head.weight.shape[0]) != 2 * BINS:
        raise SystemExit(f"head {decoder.head.weight.shape} != [{2 * BINS}, {DIM}]")

    # ---- rows ----------------------------------------------------------
    rows_meta = json.loads((pack / "rows.json").read_text())
    frozen: list[tuple[np.ndarray, np.ndarray]] = []
    for row in rows_meta[:max(args.calib_rows, args.golden_rows)]:
        ids = np.load(row["chunks"][0]["tensor_npz"])["phoneme_ids"]
        ids = ids.astype(np.int64).reshape(-1)
        tokens = torch.as_tensor(ids).unsqueeze(0)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        with torch.inference_mode():
            durs = dur_mod.predict_durations(
                dur, tokens, mask, max_duration=MAX_DURATION,
                length_scale=args.length_scale).squeeze(0).numpy().astype(np.int64)
        frozen.append((ids, durs))

    # ---- SmoothQuant (exact float identity) ----------------------------
    sq_scale = None
    if args.smoothquant_alpha and args.smoothquant_alpha > 0.0:
        act_absmax = collect_stream_absmax(acoustic, latent_mod,
                                           frozen[:args.calib_rows], device)
        w_absmax = stream_weight_absmax(acoustic)
        sq_scale = Q.smoothquant_scale(act_absmax, w_absmax, float(args.smoothquant_alpha))
        # verify the identity BEFORE trusting it
        ref = _acoustic_mel(latent_mod, acoustic, *frozen[0], device)
        Q.smooth_residual_stream(acoustic, sq_scale)
        got = _acoustic_mel(latent_mod, acoustic, *frozen[0], device)
        ident = Q.correlation(ref, got)
        rel = Q.rel_rms(ref, got)
        if ident < 0.999999 or rel > 1e-4:
            raise SystemExit(f"SmoothQuant is not an identity: corr {ident:.8f} rel {rel:.3e}")
    else:
        ident, rel = 1.0, 0.0

    # ---- front blob ----------------------------------------------------
    front = Blob("NOFF_", args.weights)
    dsd = {k: v.detach().cpu().numpy() for k, v in dur.state_dict().items()}
    asd = {k: v.detach().cpu().numpy() for k, v in acoustic.state_dict().items()}

    front.add_f32("DUR_EMB", dsd["embedding.weight"])
    front.add_q8("DUR_PROJ", dsd["input_proj.weight"], dsd["input_proj.bias"])
    for b in range(DDEPTH):
        front.add_q8(f"DUR_B{b}_C0", dsd[f"blocks.{b}.net.0.weight"], dsd[f"blocks.{b}.net.0.bias"])
        front.add_q8(f"DUR_B{b}_C1", dsd[f"blocks.{b}.net.2.weight"], dsd[f"blocks.{b}.net.2.bias"])
        front.add_f32(f"DUR_B{b}_SCALE", dsd[f"blocks.{b}.scale"])
    front.add_q8("DUR_OUT", dsd["output.weight"], dsd["output.bias"])

    front.add_f32("AC_EMB", asd["embedding.weight"])
    front.add_q8("AC_TPROJ", asd["token_input_proj.weight"], asd["token_input_proj.bias"])
    for b in range(ATD):
        front.add_q8(f"AC_TB{b}_C0", asd[f"token_blocks.{b}.net.0.weight"], asd[f"token_blocks.{b}.net.0.bias"])
        front.add_q8(f"AC_TB{b}_C1", asd[f"token_blocks.{b}.net.2.weight"], asd[f"token_blocks.{b}.net.2.bias"])
        front.add_f32(f"AC_TB{b}_SCALE", asd[f"token_blocks.{b}.scale"])
    front.add_q8("AC_FPROJ", asd["frame_input_proj.weight"], asd["frame_input_proj.bias"])
    for b in range(AD):
        front.add_q8(f"AC_FB{b}_C0", asd[f"frame_blocks.{b}.net.0.weight"], asd[f"frame_blocks.{b}.net.0.bias"])
        front.add_q8(f"AC_FB{b}_C1", asd[f"frame_blocks.{b}.net.2.weight"], asd[f"frame_blocks.{b}.net.2.bias"])
        front.add_f32(f"AC_FB{b}_SCALE", asd[f"frame_blocks.{b}.scale"])
    front.add_q8("AC_OUT", asd["output.weight"], asd["output.bias"])
    front.finish("NANO_FRONT_BYTES")

    # ---- decoder blob --------------------------------------------------
    dec = Blob("DOFF_", args.weights)
    ksd = {k: v.detach().cpu().numpy() for k, v in decoder.state_dict().items()}

    dec.add_q8("EMBED", ksd["embed.weight"], ksd["embed.bias"])
    dec.add_q8("NOISE", ksd["noise_adapter.weight"], ksd["noise_adapter.bias"])
    dec.add_f32("NORM_W", ksd["norm.weight"])
    dec.add_f32("NORM_B", ksd["norm.bias"])
    if is_dyt:
        dec.add_f32("NORM_A", ksd["norm.alpha"])
    for b in range(LAYERS):
        # depthwise weights stay f32 (R7 does the same for its dw kernels)
        dec.add_f32(f"B{b}_DW_W", ksd[f"blocks.{b}.dwconv.weight"])
        dec.add_f32(f"B{b}_DW_B", ksd[f"blocks.{b}.dwconv.bias"])
        dec.add_f32(f"B{b}_NORM_W", ksd[f"blocks.{b}.norm.weight"])
        dec.add_f32(f"B{b}_NORM_B", ksd[f"blocks.{b}.norm.bias"])
        if is_dyt:
            dec.add_f32(f"B{b}_NORM_A", ksd[f"blocks.{b}.norm.alpha"])
        dec.add_q8(f"B{b}_PW0", ksd[f"blocks.{b}.pwconv1.weight"], ksd[f"blocks.{b}.pwconv1.bias"])
        dec.add_q8(f"B{b}_PW1", ksd[f"blocks.{b}.pwconv2.weight"], ksd[f"blocks.{b}.pwconv2.bias"])
        dec.add_f32(f"B{b}_GAMMA", ksd[f"blocks.{b}.gamma"])
    dec.add_f32("FNORM_W", ksd["final_norm.weight"])
    dec.add_f32("FNORM_B", ksd["final_norm.bias"])
    if is_dyt:
        dec.add_f32("FNORM_A", ksd["final_norm.alpha"])
    dec.add_q8("HEAD", ksd["head.weight"], ksd["head.bias"])
    dec.finish("NANO_DEC_BYTES")

    _assert_covered(dsd, asd, ksd, is_dyt)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    blob_tag = "q8" if args.weights == "int8" else "f32"
    (out / f"front_{blob_tag}.bin").write_bytes(bytes(front.buf))
    (out / f"model_{blob_tag}.bin").write_bytes(bytes(dec.buf))

    # ---- generated header ----------------------------------------------
    shape_defines = [
        ("NANO_FORMAT_VERSION", FORMAT_VERSION),
        ("NANO_MAGIC_FRONT", MAGIC_FRONT),
        ("NANO_MAGIC_DEC", MAGIC_DEC),
        ("NANO_VOCAB", VOCAB),
        ("NANO_DUR_HIDDEN", DH),
        ("NANO_DUR_DEPTH", DDEPTH),
        ("NANO_DUR_KERNEL", DK),
        ("NANO_DUR_MAX_TOKENS", MAX_TOKENS),
        ("NANO_DUR_MAX_DURATION", MAX_DURATION),
        ("NANO_AC_HIDDEN", AH),
        ("NANO_AC_TOKEN_DEPTH", ATD),
        ("NANO_AC_DEPTH", AD),
        ("NANO_AC_KERNEL", AK),
        ("NANO_MELS", AOUT),
        ("NANO_DIM", DIM),
        ("NANO_BLOCKS", LAYERS),
        ("NANO_PW_HIDDEN", HID),
        ("NANO_DW_KERNEL", int(decoder.blocks[0].dwconv.weight.shape[2])),
        ("NANO_EMBED_KERNEL", int(decoder.embed.weight.shape[2])),
        ("NANO_NOISE_CH", NCH),
        ("NANO_N_FFT", NFFT),
        ("NANO_HOP", 256),
        ("NANO_BINS", BINS),
        ("NANO_HEAD_OUT", 2 * BINS),
        # E13 operator substitutions; the runtime defaults both to 0 when a
        # pre-E13 header omits them, so E12 headers remain reproducible.
        ("NANO_NORM_TYPE", 1 if norm_type == "dyt" else 0),
        ("NANO_ACT_TYPE", 1 if act_type == "relu" else 0),
        # 0 = int8 rows (device); 1 = float32 rows, unit scales (browser).
        # snt_nano.c refuses to compile a blob of the other element type.
        ("NANO_WEIGHT_FORMAT", 0 if args.weights == "int8" else 1),
    ]
    n16_defines = []
    for blob in (front, dec):
        for r in blob.regions:
            if r["kind"] in ("q8_weight", "f32_weight"):
                n16_defines.append((f"NANO_{r['region'][:-3]}_N16", r["n16"]))

    lines = ["/* generated by tools/export_e12_nano_q8.py -- do not edit */",
             "#pragma once", ""]
    for k, v in shape_defines:
        lines.append(f"#define {k} {v}")
    lines.append("")
    for k, v in n16_defines:
        lines.append(f"#define {k} {v}")
    lines.append("")
    lines.append(f"/* byte offsets into front_{blob_tag}.bin */")
    for k, v in front.defines[:-1]:
        lines.append(f"#define {k} {v}")
    lines.append(f"#define {front.defines[-1][0]} {front.defines[-1][1]}")
    lines.append("")
    lines.append(f"/* byte offsets into model_{blob_tag}.bin */")
    for k, v in dec.defines[:-1]:
        lines.append(f"#define {k} {v}")
    lines.append(f"#define {dec.defines[-1][0]} {dec.defines[-1][1]}")
    lines.append("")
    (out / "nano_q8_meta.h").write_text("\n".join(lines))

    # ---- golden ---------------------------------------------------------
    # The gate is the MINIMUM correlation over rows, not one row's:
    # docs/e12-nano-int8-quantisation.md section 1 measured the min moving by
    # 0.07 across 75k decoder steps, so a single-row fixture would be noise.
    gold = out / "golden"
    gold.mkdir(exist_ok=True)
    manifest = []
    golden_rows = []
    for gi in range(min(args.golden_rows, len(frozen))):
        ids, durs = frozen[gi]
        row_id = str(rows_meta[gi]["row_id"])
        frames = int(durs.sum())
        mel = _acoustic_mel(latent_mod, acoustic, ids, durs, device)
        noise = seeded_noise(row_id, NCH, frames)
        with torch.inference_mode():
            mel_t = torch.as_tensor(mel).transpose(0, 1).unsqueeze(0).contiguous()
            spec = decoder(mel_t, noise=noise)
            wav = decoder.synthesize(spec, torch.hann_window(NFFT)).squeeze(0).numpy()
            phi_max = float(np.abs(_head_phase(decoder, mel_t, noise)).max())
        seed = noise_seed_u64(row_id)
        (gold / f"r{gi:02d}_ids.bin").write_bytes(ids.astype("<i4").tobytes())
        (gold / f"r{gi:02d}_durs.bin").write_bytes(durs.astype("<i4").tobytes())
        (gold / f"r{gi:02d}_audio.bin").write_bytes(wav.astype("<f4").tobytes())
        manifest.append(f"{row_id} {int(ids.size)} {frames} {int(wav.size)} {seed}")
        golden_rows.append({"index": gi, "row_id": row_id, "tokens": int(ids.size),
                            "frames": frames, "samples": int(wav.size),
                            "seed_u64": seed, "phi_abs_max": phi_max})
        if gi == 0:
            # component references for the two new numeric primitives
            (gold / "e2e_noise.bin").write_bytes(
                noise.squeeze(0).numpy().astype("<f4").tobytes())
            g = torch.Generator(device="cpu")
            g.manual_seed(seed)
            (gold / "e2e_uniform.bin").write_bytes(
                torch.rand(64, generator=g).numpy().astype("<f4").tobytes())
    (gold / "rows.txt").write_text("\n".join(manifest) + "\n")

    # ---- Python reference reload: does the blob reproduce the simulator?
    src_map = dict(front.source)
    src_map.update(dec.source)
    reload_report = _verify_reload(Q, front, dec, src_map)

    report = {
        "schema": "saanotts.e12-nano-export/1",
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "duration_checkpoint": str(dur_ckpt),
        "acoustic_checkpoint": str(ac_ckpt),
        "pack": str(pack),
        "norm_type": norm_type,
        "act_type": act_type,
        "golden_rows": golden_rows,
        "phi_abs_max_over_rows": max(r["phi_abs_max"] for r in golden_rows),
        "shapes": dict(shape_defines),
        "smoothquant": {
            "alpha": float(args.smoothquant_alpha),
            "applied": sq_scale is not None,
            "float_identity_corr": float(ident),
            "float_identity_rel_rms": float(rel),
            "scale": sq_scale.tolist() if sq_scale is not None else None,
        },
        "weights": args.weights,
        "bytes": {f"front_{blob_tag}.bin": len(front.buf), f"model_{blob_tag}.bin": len(dec.buf)},
        "front_regions": front.regions,
        "dec_regions": dec.regions,
        "reload_check": reload_report,
    }
    (out / "export-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"out": str(out), "front_bytes": len(front.buf),
                      "dec_bytes": len(dec.buf), "golden_rows": len(golden_rows),
                      "phi_abs_max": report["phi_abs_max_over_rows"],
                      "reload": reload_report["summary"]}, indent=1))


def _head_phase(decoder, mel_t, noise) -> np.ndarray:
    """The raw phase argument the head emits, for the sincos range budget."""
    with torch.inference_mode():
        x = decoder.embed(mel_t) + decoder.noise_adapter(noise)
        x = decoder.norm(x.transpose(1, 2)).transpose(1, 2)
        for block in decoder.blocks:
            x = block(x)
        x = decoder.final_norm(x.transpose(1, 2))
        out = decoder.head(x).transpose(1, 2)
        _, phase = out.chunk(2, dim=1)
    return phase.numpy()


def _acoustic_mel(latent_mod, acoustic, ids, durs, device) -> np.ndarray:
    frames = int(durs.sum())
    sample = SimpleNamespace(row_id="g", chunk_index=0, phoneme_ids=ids,
                             durations=durs,
                             target=np.zeros((frames, 100), dtype=np.float32))
    feats = latent_mod.expand_features(sample, device)
    with torch.inference_mode():
        return latent_mod.predict_latent_tensor(acoustic, feats).numpy()


def _assert_covered(dsd, asd, ksd, dec_is_dyt: bool = False) -> None:
    """Every state-dict tensor must land in exactly one region."""
    want = ({f"dur:{k}" for k in dsd} | {f"ac:{k}" for k in asd}
            | {f"dec:{k}" for k in ksd})
    got = set()

    def mark(prefix, keys):
        for k in keys:
            got.add(f"{prefix}:{k}")

    mark("dur", ["embedding.weight", "input_proj.weight", "input_proj.bias", "output.weight", "output.bias"])
    for b in range(sum(1 for k in dsd if k.endswith(".scale") and k.startswith("blocks."))):
        mark("dur", [f"blocks.{b}.net.0.weight", f"blocks.{b}.net.0.bias",
                     f"blocks.{b}.net.2.weight", f"blocks.{b}.net.2.bias",
                     f"blocks.{b}.scale"])
    mark("ac", ["embedding.weight", "token_input_proj.weight", "token_input_proj.bias",
                "frame_input_proj.weight", "frame_input_proj.bias",
                "output.weight", "output.bias"])
    for group in ("token_blocks", "frame_blocks"):
        n = sum(1 for k in asd if k.startswith(f"{group}.") and k.endswith(".scale"))
        for b in range(n):
            mark("ac", [f"{group}.{b}.net.0.weight", f"{group}.{b}.net.0.bias",
                        f"{group}.{b}.net.2.weight", f"{group}.{b}.net.2.bias",
                        f"{group}.{b}.scale"])
    mark("dec", ["embed.weight", "embed.bias", "noise_adapter.weight", "noise_adapter.bias",
                 "norm.weight", "norm.bias", "final_norm.weight", "final_norm.bias",
                 "head.weight", "head.bias"])
    if dec_is_dyt:
        mark("dec", ["norm.alpha", "final_norm.alpha"])
    n = sum(1 for k in ksd if k.endswith(".gamma"))
    for b in range(n):
        mark("dec", [f"blocks.{b}.dwconv.weight", f"blocks.{b}.dwconv.bias",
                     f"blocks.{b}.norm.weight", f"blocks.{b}.norm.bias",
                     f"blocks.{b}.pwconv1.weight", f"blocks.{b}.pwconv1.bias",
                     f"blocks.{b}.pwconv2.weight", f"blocks.{b}.pwconv2.bias",
                     f"blocks.{b}.gamma"])
        if dec_is_dyt:
            mark("dec", [f"blocks.{b}.norm.alpha"])
    missing = sorted(want - got)
    extra = sorted(got - want)
    if missing or extra:
        raise SystemExit(f"state dict coverage mismatch: missing={missing} extra={extra}")


def _verify_reload(Q, front: Blob, dec: Blob, region_source: dict) -> dict:
    """Phase-1 gate: reload every shipped region and prove it is the simulator.

    For each ``_W8`` region the int8 codes and f32 scales are read back out of
    the blob bytes and compared, element by element, against what
    ``mcu_quant_sim.quantise_weight`` produces from the same float tensor under
    the shipped policy (per-output-channel symmetric int8, round-half-to-even).
    A single differing code fails the export.  For each ``_F32`` region the
    bytes must reproduce the float tensor exactly.

    This is what lets every downstream simulation be trusted: the blob and
    ``tools/mcu_quant_sim.py`` are the same quantiser, not two similar ones.
    """
    cfg = Q.QuantConfig(w_bits=8, w_per_channel=True, w_symmetric=True)
    q8_checked = f32_checked = 0
    for blob in (front, dec):
        buf = bytes(blob.buf)
        regions = {r["region"]: r for r in blob.regions}
        for name, r in regions.items():
            if r["kind"] == "q8_weight":
                src = region_source.get(name)
                if src is None:
                    raise SystemExit(f"{name}: no source tensor recorded")
                w = torch.as_tensor(src, dtype=torch.float32)
                deq_ref, info = Q.quantise_weight(w, cfg)
                scale_ref = info["scale"].reshape(-1).numpy().astype(np.float32)
                # deq == q * scale exactly in float32, so dividing back and
                # rounding recovers the simulator's own int8 codes.
                q_ref = np.rint(
                    deq_ref.reshape(w.shape[0], -1).numpy().astype(np.float64)
                    / scale_ref[:, None].astype(np.float64)).astype(np.int8)

                q = np.frombuffer(buf, dtype=np.int8, count=r["out_ch"] * r["n16"],
                                  offset=r["offset"]).reshape(r["out_ch"], r["n16"])
                sc = regions[f"{name[:-3]}_SCALE"]
                s = np.frombuffer(buf, dtype="<f4", count=sc["count"], offset=sc["offset"])
                if np.abs(q[:, r["in_flat"]:]).sum() != 0:
                    raise SystemExit(f"{name}: padding columns are not zero")
                if not np.array_equal(q[:, :r["in_flat"]], q_ref):
                    bad = int((q[:, :r["in_flat"]] != q_ref).sum())
                    raise SystemExit(f"{name}: {bad} int8 codes differ from mcu_quant_sim")
                if not np.array_equal(s, scale_ref):
                    raise SystemExit(f"{name}: f32 scales differ from mcu_quant_sim")
                del deq_ref
                q8_checked += 1
            elif r["kind"] == "f32_weight":
                src = region_source.get(name)
                if src is None:
                    raise SystemExit(f"{name}: no source tensor recorded")
                w = np.asarray(src, dtype=np.float32).reshape(r["out_ch"], -1)
                got = np.frombuffer(buf, dtype="<f4", count=r["out_ch"] * r["n16"],
                                    offset=r["offset"]).reshape(r["out_ch"], r["n16"])
                if np.abs(got[:, r["in_flat"]:]).sum() != 0:
                    raise SystemExit(f"{name}: padding columns are not zero")
                if not np.array_equal(got[:, :r["in_flat"]], w):
                    raise SystemExit(f"{name}: f32 weight rows differ from the checkpoint")
                sc = regions[f"{name[:-3]}_SCALE"]
                s = np.frombuffer(buf, dtype="<f4", count=sc["count"], offset=sc["offset"])
                if not np.all(s == 1.0):
                    raise SystemExit(f"{name}: f32 rows must carry unit scales")
                f32_checked += 1
            elif r["kind"] == "f32":
                src = region_source.get(name)
                if src is None:
                    raise SystemExit(f"{name}: no source tensor recorded")
                got = np.frombuffer(buf, dtype="<f4", count=r["count"], offset=r["offset"])
                if not np.array_equal(got, np.asarray(src, dtype=np.float32).reshape(-1)):
                    raise SystemExit(f"{name}: f32 payload differs from the checkpoint")
                f32_checked += 1
    return {"q8_regions": q8_checked, "f32_regions": f32_checked,
            "summary": f"{q8_checked} int8 regions bit-identical to mcu_quant_sim, "
                       f"{f32_checked} f32 regions bit-identical to the checkpoints"}


if __name__ == "__main__":
    main()
