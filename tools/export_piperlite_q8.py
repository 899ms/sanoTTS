"""Export a piperlite DecoderStudent as per-output-channel symmetric int8.

Companion to tools/export_piperlite_golden.py (run that first for the same
--out dir: it provides z.bin / audio.bin / per-stage goldens). This script
adds:
  weights_q8.bin -- pure int8 weight payload, same slot order as the fp32
                    exporter (this is the distribution blob)
  meta_q8.bin    -- dims + STATIC activation scales (calibrated over real
                    eval128 latents) + per-slot records:
                    kind 0 = int8 weights (offset/size into weights_q8.bin,
                             per-out-channel fp32 scales in the pool)
                    kind 1 = fp32 data (bias / pf unit scalar, in the pool)
  calib_q8.json  -- calibrated ranges, for the record

meta_q8.bin layout (little-endian), parsed by mcu/src/snt_piperlite_q8.c:
  i32 magic 0x534E5051 'SNPQ', i32 version=1,
  i32 in_ch, c0, c1, c2, c3, pf_channels, pf_layers, pf_kernel,
  f32 pf_scale, i32 n_tensors, i32 n_act(=39), f32 act_scales[39],
  n_tensors x (i32 kind, i32 offset, i32 size, i32 aux_off, i32 aux_n),
  i32 pool_n, f32 pool[pool_n].

Activation-scale order (39 entries, scale = max|tensor|/127 over the
calibration set, optional --headroom multiplier):
  [0] z   [1] pre
  stage s in 0..2 at base 2+12*s:
    +0 act-in (leaky 0.1 of previous mix/pre, i.e. up input)
    +1 up output
    +2+3b, +3+3b, +4+3b  branch b: t1 (leaky x), y1 (conv1+x), t2 (leaky y1)
    +11 mix (bank output)
  [38] ap (post-conv input, leaky 0.01 of stage2 mix)

Run from repo root, e.g.:
  .venv/bin/python tools/export_piperlite_q8.py \
      artifacts/.../decoder-student.pt --pack artifacts/.../eval128-piper-native \
      --out mcu/test/golden_piperlite/amy
"""

import argparse
import importlib.util
import json
import pathlib
import struct
import sys

pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dm = _load_module("dm", REPO / "tools/train_roota_piper_decoder_student.py")
golden = _load_module("piperlite_golden", REPO / "tools/export_piperlite_golden.py")

MAGIC = 0x534E5051  # 'SNPQ'
VERSION = 1
N_ACT = 39
BRANCH_KERNELS = (3, 5, 7)


def quant_per_oc(w: np.ndarray, oc_axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8. Returns (q_int8, scales[out_ch])."""
    reduce_axes = tuple(a for a in range(w.ndim) if a != oc_axis)
    amax = np.abs(w).max(axis=reduce_axes)
    scales = amax / 127.0
    scales[scales == 0.0] = 1.0
    shape = [1] * w.ndim
    shape[oc_axis] = -1
    q = np.clip(np.round(w / scales.reshape(shape)), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


SUBSAMPLE_STRIDE = 97  # prime, decorrelates from channel/time layout


def calibrate(dec, latents: list[torch.Tensor]) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Returns (per-tensor abs max, per-tensor subsampled |values|)."""
    amax: dict[str, float] = {}
    samples: dict[str, list[np.ndarray]] = {}

    def note(name: str, t: torch.Tensor) -> None:
        flat = t.detach().reshape(-1)
        v = float(flat.abs().max().item())
        if v > amax.get(name, 0.0):
            amax[name] = v
        samples.setdefault(name, []).append(
            flat[::SUBSAMPLE_STRIDE].abs().numpy().astype(np.float32))

    def out_hook(name):
        return lambda mod, inp, out: note(name, out)

    def in_hook(name):
        return lambda mod, inp: note(name, inp[0])

    handles = []
    handles.append(dec.pre.register_forward_hook(out_hook("pre")))
    for s in range(3):
        up = getattr(dec, f"up{s}")
        handles.append(up.register_forward_pre_hook(in_hook(f"a{s}")))
        handles.append(up.register_forward_hook(out_hook(f"up{s}")))
        res = getattr(dec, f"res{s}")
        handles.append(res.register_forward_hook(out_hook(f"mix{s}")))
        bank = res[0]
        for br, blk in enumerate(bank.blocks):
            handles.append(blk.act0.register_forward_hook(out_hook(f"t1_{s}_{br}")))
            handles.append(blk.act1.register_forward_pre_hook(in_hook(f"y1_{s}_{br}")))
            handles.append(blk.act1.register_forward_hook(out_hook(f"t2_{s}_{br}")))
    handles.append(dec.post.register_forward_pre_hook(in_hook("ap")))
    try:
        with torch.no_grad():
            for z in latents:
                note("z", z)
                dec(z.unsqueeze(0))
    finally:
        for h in handles:
            h.remove()
    expected = 2 + 3 * 12 + 1  # z, pre, 12/stage, ap == 39
    if len(amax) != expected:
        raise SystemExit(f"calibration captured {len(amax)} tensors, expected {expected}: "
                         f"{sorted(amax)}")
    return amax, {k: np.concatenate(v) for k, v in samples.items()}


RUNTIME_ACT_QMAX = 2047  # the C runtime's activation lane is 12-bit int16


def mse_optimal_amax(absvals: np.ndarray, amax: float,
                     qmax: int = RUNTIME_ACT_QMAX) -> float:
    """Pick the clip value minimizing quantization MSE on the runtime's
    activation grid (SQNR search). These activations are heavy-tailed
    (amax/rms is 7..21 here); on an int8 grid clipping buys a lot, on the
    12-bit lane the search usually keeps the full range (no saturation).
    """
    if absvals.size == 0 or amax <= 0.0:
        return amax
    candidates = [np.percentile(absvals, p) for p in (99.0, 99.5, 99.9, 99.99)]
    candidates.append(amax)
    best_c, best_mse = amax, np.inf
    for c in candidates:
        if c <= 0.0:
            continue
        s = c / float(qmax)
        q = np.clip(np.round(absvals / s), 0, qmax) * s
        mse = float(np.mean((q - absvals) ** 2))
        if mse < best_mse:
            best_c, best_mse = float(c), mse
    return best_c


def act_scale_vector(amax: dict[str, float], headroom: float) -> list[float]:
    names = ["z", "pre"]
    for s in range(3):
        names.append(f"a{s}")
        names.append(f"up{s}")
        for br in range(3):
            names.extend([f"t1_{s}_{br}", f"y1_{s}_{br}", f"t2_{s}_{br}"])
        names.append(f"mix{s}")
    names.append("ap")
    assert len(names) == N_ACT
    scales = []
    for n in names:
        v = amax[n] * headroom / 127.0
        scales.append(v if v > 0.0 else 1.0)
    return scales


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--pack", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--calib-n", type=int, default=16,
                    help="calibration latents (rows.json rows, first chunk each)")
    ap.add_argument("--calib-frames", type=int, default=256,
                    help="truncate each calibration latent (0 = full)")
    ap.add_argument("--headroom", type=float, default=1.0,
                    help="multiplier on calibrated activation clip")
    ap.add_argument("--calib-mode", choices=("mse", "max"), default="mse",
                    help="mse: SQNR-optimal clip search (default); max: abs max")
    args = ap.parse_args()

    torch.manual_seed(0)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full_cfg = dict(ck["config"])
    golden.check_supported(full_cfg)
    cfg = {k: v for k, v in full_cfg.items() if k in golden.CONFIG_KEYS}
    dec = dm.DecoderStudent(**cfg)
    dec.load_state_dict(ck["model_state_dict"], strict=True)
    dec.eval()

    in_ch = int(cfg["in_channels"])
    c0, c1, c2, c3 = (int(c) for c in cfg["channels"][:4])
    pf_ch = int(cfg.get("post_filter_channels", 0))
    pf_layers = int(cfg.get("post_filter_layers", 0))
    pf_kernel = int(cfg.get("post_filter_kernel", 9))
    pf_scale = float(cfg.get("post_filter_scale", 0.25))

    # --- calibration over real latents -------------------------------------
    rows = json.loads((args.pack / "rows.json").read_text())
    n_rows = min(args.calib_n, len(rows))
    if n_rows < args.calib_n:
        print(f"pack has only {len(rows)} rows; calibrating on {n_rows}",
              file=sys.stderr)
    if n_rows < 4:
        raise SystemExit(f"need at least 4 calibration rows, pack has {len(rows)}")
    latents = []
    for row in range(n_rows):
        z = golden.load_latent(args.pack, in_ch, row, args.calib_frames)
        latents.append(torch.from_numpy(z))
    raw_amax, samples = calibrate(dec, latents)
    if args.calib_mode == "mse":
        amax = {k: mse_optimal_amax(samples[k], v) for k, v in raw_amax.items()}
    else:
        amax = dict(raw_amax)
    act_scales = act_scale_vector(amax, args.headroom)

    # --- weight quantization, same slot order as the fp32 exporter ---------
    sd = ck["model_state_dict"]
    records = []          # (kind, offset, size, aux_off, aux_n)
    blob = bytearray()    # int8 payload
    pool: list[float] = []  # f32 payload (biases, weight scales, pf scalars)
    slot_names: list[str] = []

    def add_w(name: str, oc_axis: int) -> None:
        w = sd[name].detach().float().numpy()
        q, scales = quant_per_oc(w, oc_axis)
        aux_off = len(pool)
        pool.extend(scales.tolist())
        records.append((0, len(blob), q.size, aux_off, scales.size))
        blob.extend(np.ascontiguousarray(q).tobytes())
        slot_names.append(name)

    def add_f(name: str) -> None:
        t = sd[name].detach().float().reshape(-1).numpy()
        records.append((1, len(pool), t.size, -1, 0))
        pool.extend(t.tolist())
        slot_names.append(name)

    def add_bank(prefix: str, ch: int) -> None:
        for b, k in enumerate(BRANCH_KERNELS):
            for conv in ("conv1", "conv2"):
                w = sd[f"{prefix}.blocks.{b}.{conv}.weight"]
                assert tuple(w.shape) == (ch, ch, k)
                add_w(f"{prefix}.blocks.{b}.{conv}.weight", 0)
                add_f(f"{prefix}.blocks.{b}.{conv}.bias")

    add_w("pre.weight", 0)          # Conv1d [out,in,K] -> per axis 0
    add_f("pre.bias")
    add_w("up0.weight", 1)          # ConvTranspose1d [in,out,K] -> per axis 1
    add_f("up0.bias")
    add_bank("res0.0", c1)
    add_w("up1.weight", 1)
    add_f("up1.bias")
    add_bank("res1.0", c2)
    add_w("up2.weight", 1)
    add_f("up2.bias")
    add_bank("res2.0", c3)
    add_w("post.weight", 0)
    add_f("post.bias")
    if pf_ch > 0:
        add_w("post_filter.in_conv.weight", 0)
        add_f("post_filter.in_conv.bias")
        for layer in range(pf_layers):
            add_f(f"post_filter.units.{layer}.scale")
            add_w(f"post_filter.units.{layer}.conv1.weight", 0)
            add_f(f"post_filter.units.{layer}.conv1.bias")
            add_w(f"post_filter.units.{layer}.conv2.weight", 0)
            add_f(f"post_filter.units.{layer}.conv2.bias")
        add_w("post_filter.out_conv.weight", 0)
        add_f("post_filter.out_conv.bias")

    leftover = sorted(set(sd.keys()) - set(slot_names))
    if leftover:
        raise SystemExit(f"state dict tensors not covered: {leftover}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "weights_q8.bin").write_bytes(bytes(blob))

    with (args.out / "meta_q8.bin").open("wb") as fh:
        fh.write(struct.pack("<10i", MAGIC, VERSION, in_ch, c0, c1, c2, c3,
                             pf_ch, pf_layers, pf_kernel))
        fh.write(struct.pack("<f", pf_scale))
        fh.write(struct.pack("<2i", len(records), N_ACT))
        fh.write(struct.pack(f"<{N_ACT}f", *act_scales))
        for rec in records:
            fh.write(struct.pack("<5i", *rec))
        fh.write(struct.pack("<i", len(pool)))
        fh.write(np.asarray(pool, dtype="<f4").tobytes())

    (args.out / "calib_q8.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "calib_n": args.calib_n,
         "calib_frames": args.calib_frames, "headroom": args.headroom,
         "calib_mode": args.calib_mode,
         "amax": {k: amax[k] for k in sorted(amax)},
         "raw_amax": {k: raw_amax[k] for k in sorted(raw_amax)},
         "weights_q8_bytes": len(blob),
         "meta_q8_bytes": (args.out / "meta_q8.bin").stat().st_size},
        indent=1))
    print(json.dumps({"out": str(args.out), "weights_q8_bytes": len(blob),
                      "meta_q8_bytes": (args.out / "meta_q8.bin").stat().st_size,
                      "n_tensors": len(records), "calib_n": len(latents)}))


if __name__ == "__main__":
    main()
