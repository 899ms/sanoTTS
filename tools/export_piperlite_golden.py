"""Export a piperlite DecoderStudent checkpoint for the portable C99 runtime.

Writes to --out (default mcu/test/golden_piperlite/<name>/):
  weights_f32.bin   -- all weights, float32 LE, fixed documented order (below)
  meta.bin          -- binary meta (dims + per-tensor offset table) read at runtime
  piperlite_meta.h  -- same info as macros, for MCU builds that bake in one model
  z.bin             -- golden latent [in_ch, frames] float32 (real eval128 latent)
  audio.bin         -- PyTorch forward output on z [frames*256] float32
  <stage>.bin       -- per-stage goldens (pre, up0, stage0_mix, up1, stage1_mix,
                       up2, stage2_mix, pre_tanh, audio_pre_filter) for bring-up diffs
  golden.json       -- dims + shapes

WEIGHT ORDER (float32, contiguous, PyTorch layouts kept as-is):
  slot  0 pre.weight        [c0, in_ch, 7]      Conv1d [out, in, K]
  slot  1 pre.bias          [c0]
  slot  2 up0.weight        [c0, c1, 16]        ConvTranspose1d [in, out, K]
  slot  3 up0.bias          [c1]
  slots 4..15  res0 bank: for branch b in 0,1,2 (kernels 3,5,7; dilations
               (1,2),(2,6),(3,12)): conv1.weight [c1,c1,k], conv1.bias [c1],
               conv2.weight [c1,c1,k], conv2.bias [c1]
  slot 16 up1.weight        [c1, c2, 16]
  slot 17 up1.bias          [c2]
  slots 18..29 res1 bank (same structure at c2)
  slot 30 up2.weight        [c2, c3, 8]
  slot 31 up2.bias          [c3]
  slots 32..43 res2 bank (same structure at c3)
  slot 44 post.weight       [1, c3, 7]
  slot 45 post.bias         [1]
  optional post filter (pf_channels > 0):
  slot 46 post_filter.in_conv.weight  [pf_ch, 1, pf_k]
  slot 47 post_filter.in_conv.bias    [pf_ch]
  per layer l in 0..pf_layers-1:
    slot 48+5l   units.l.scale        [1]  (learned scalar)
    slot 48+5l+1 units.l.conv1.weight [pf_ch, pf_ch, 3]  dilation 1+l
    slot 48+5l+2 units.l.conv1.bias   [pf_ch]
    slot 48+5l+3 units.l.conv2.weight [pf_ch, pf_ch, 3]  dilation 1
    slot 48+5l+4 units.l.conv2.bias   [pf_ch]
  last two slots: post_filter.out_conv.weight [1, pf_ch, pf_k], .bias [1]

META.BIN (little-endian):
  int32 magic 0x534E504C ('SNPL'), int32 version=1,
  int32 in_ch, c0, c1, c2, c3,
  int32 pf_channels, pf_layers, pf_kernel, float32 pf_scale,
  int32 n_tensors, then n_tensors x (int32 offset_floats, int32 size_floats).

Run from repo root, e.g.:
  .venv/bin/python tools/export_piperlite_golden.py \
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

spec = importlib.util.spec_from_file_location(
    "dm", REPO / "tools/train_roota_piper_decoder_student.py"
)
dm = importlib.util.module_from_spec(spec)
sys.modules["dm"] = dm
spec.loader.exec_module(dm)

MAGIC = 0x534E504C  # 'SNPL'
VERSION = 1
HOP = 256

CONFIG_KEYS = {
    "in_channels", "channels", "res_layers", "variant", "rank_ratio", "activation",
    "stage_affine", "factorized_pre_rank", "piper_res_factor_rank_ratio",
    "res_bank_scale_mode", "stage0_branches", "stage1_branches", "stage2_branches",
    "stage3_branches", "post_filter_channels", "post_filter_layers",
    "post_filter_kernel", "post_filter_scale", "stage_projection_bottlenecks",
}

BRANCH_KERNELS = (3, 5, 7)


def check_supported(cfg: dict) -> None:
    """Fail loudly on anything the C port does not implement."""
    problems = []
    if cfg.get("variant") != "piperlite":
        problems.append(f"variant={cfg.get('variant')!r} (only piperlite)")
    if cfg.get("activation") != "leaky_relu":
        problems.append(f"activation={cfg.get('activation')!r} (only leaky_relu)")
    if cfg.get("stage_affine"):
        problems.append("stage_affine=True unsupported")
    if int(cfg.get("factorized_pre_rank", 0)) != 0:
        problems.append("factorized_pre_rank != 0 unsupported")
    if float(cfg.get("piper_res_factor_rank_ratio", 0.0)) != 0.0:
        problems.append("piper_res_factor_rank_ratio != 0 unsupported")
    if cfg.get("res_bank_scale_mode", "kept") != "kept":
        problems.append("res_bank_scale_mode != 'kept' unsupported")
    for stage in ("stage0_branches", "stage1_branches", "stage2_branches"):
        if tuple(cfg.get(stage, (0, 1, 2))) != (0, 1, 2):
            problems.append(f"{stage}={cfg.get(stage)} (only full (0,1,2))")
    if int(cfg.get("res_layers", 1)) != 1:
        problems.append(f"res_layers={cfg.get('res_layers')} (only 1)")
    if list(cfg.get("stage_projection_bottlenecks", []) or []):
        problems.append("stage_projection_bottlenecks unsupported")
    if int(cfg.get("pre_tanh_repair_channels", 0)) != 0:
        problems.append("pre_tanh_repair unsupported")
    if int(cfg.get("post_filter_kernel", 9)) % 2 == 0:
        problems.append("post_filter_kernel must be odd")
    if problems:
        raise SystemExit("checkpoint not supported by C port:\n  " + "\n  ".join(problems))


def load_latent(pack: pathlib.Path, in_ch: int, chunk_row: int, max_frames: int) -> np.ndarray:
    rows = json.loads((pack / "rows.json").read_text())
    chunk = rows[chunk_row]["chunks"][0]
    npz = np.load(REPO / chunk["tensor_npz"])
    if "generator_input" not in npz:
        raise SystemExit(f"{chunk['tensor_npz']} has no 'generator_input'")
    z = np.asarray(npz["generator_input"], dtype=np.float32)
    while z.ndim > 2:
        z = z[0]
    if z.shape[0] != in_ch:
        if z.shape[1] == in_ch:
            z = z.T
        else:
            raise SystemExit(f"latent shape {z.shape} does not match in_ch={in_ch}")
    if max_frames > 0 and z.shape[1] > max_frames:
        z = z[:, :max_frames]
    return np.ascontiguousarray(z)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=pathlib.Path)
    ap.add_argument("--pack", type=pathlib.Path, required=True,
                    help="eval pack dir with rows.json + tensors/*.npz (generator_input)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--chunk-row", type=int, default=0, help="row index into rows.json")
    ap.add_argument("--max-frames", type=int, default=256,
                    help="truncate the golden latent to this many frames (0 = full)")
    args = ap.parse_args()

    torch.manual_seed(0)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full_cfg = dict(ck["config"])
    check_supported(full_cfg)
    cfg = {k: v for k, v in full_cfg.items() if k in CONFIG_KEYS}
    dec = dm.DecoderStudent(**cfg)
    dec.load_state_dict(ck["model_state_dict"], strict=True)
    dec.eval()

    in_ch = int(cfg["in_channels"])
    c0, c1, c2, c3 = (int(c) for c in cfg["channels"][:4])
    pf_ch = int(cfg.get("post_filter_channels", 0))
    pf_layers = int(cfg.get("post_filter_layers", 0))
    pf_kernel = int(cfg.get("post_filter_kernel", 9))
    pf_scale = float(cfg.get("post_filter_scale", 0.25))

    sd = ck["model_state_dict"]
    order: list[tuple[str, torch.Tensor]] = []

    def add(name: str) -> None:
        order.append((name, sd[name].detach().float().contiguous()))

    def add_bank(prefix: str, ch: int) -> None:
        for b, k in enumerate(BRANCH_KERNELS):
            for conv in ("conv1", "conv2"):
                w = sd[f"{prefix}.blocks.{b}.{conv}.weight"]
                assert tuple(w.shape) == (ch, ch, k), (prefix, b, conv, w.shape)
                add(f"{prefix}.blocks.{b}.{conv}.weight")
                add(f"{prefix}.blocks.{b}.{conv}.bias")

    add("pre.weight")
    add("pre.bias")
    add("up0.weight")
    add("up0.bias")
    add_bank("res0.0", c1)
    add("up1.weight")
    add("up1.bias")
    add_bank("res1.0", c2)
    add("up2.weight")
    add("up2.bias")
    add_bank("res2.0", c3)
    add("post.weight")
    add("post.bias")
    if pf_ch > 0:
        add("post_filter.in_conv.weight")
        add("post_filter.in_conv.bias")
        for layer in range(pf_layers):
            order.append((f"post_filter.units.{layer}.scale",
                          sd[f"post_filter.units.{layer}.scale"].detach().float().reshape(1)))
            add(f"post_filter.units.{layer}.conv1.weight")
            add(f"post_filter.units.{layer}.conv1.bias")
            add(f"post_filter.units.{layer}.conv2.weight")
            add(f"post_filter.units.{layer}.conv2.bias")
        add("post_filter.out_conv.weight")
        add("post_filter.out_conv.bias")

    consumed = {name for name, _ in order}
    leftover = sorted(set(sd.keys()) - consumed)
    if leftover:
        raise SystemExit(f"state dict tensors not covered by export order: {leftover}")

    args.out.mkdir(parents=True, exist_ok=True)
    offsets: list[tuple[str, int, int]] = []
    cursor = 0
    with (args.out / "weights_f32.bin").open("wb") as fh:
        for name, t in order:
            arr = t.numpy().astype("<f4")
            offsets.append((name, cursor, arr.size))
            fh.write(arr.tobytes())
            cursor += arr.size

    # meta.bin -- runtime-readable dims + offset table
    with (args.out / "meta.bin").open("wb") as fh:
        fh.write(struct.pack("<10i", MAGIC, VERSION, in_ch, c0, c1, c2, c3,
                             pf_ch, pf_layers, pf_kernel))
        fh.write(struct.pack("<f", pf_scale))
        fh.write(struct.pack("<i", len(offsets)))
        for _, off, size in offsets:
            fh.write(struct.pack("<2i", off, size))

    # piperlite_meta.h -- same info as macros (for MCU builds baking in one model)
    def cname(name: str) -> str:
        return name.upper().replace(".", "_")

    lines = ["/* generated by tools/export_piperlite_golden.py -- do not edit */",
             "#pragma once", "",
             f"#define PIPERLITE_IN_CH {in_ch}",
             f"#define PIPERLITE_C0 {c0}",
             f"#define PIPERLITE_C1 {c1}",
             f"#define PIPERLITE_C2 {c2}",
             f"#define PIPERLITE_C3 {c3}",
             f"#define PIPERLITE_PF_CHANNELS {pf_ch}",
             f"#define PIPERLITE_PF_LAYERS {pf_layers}",
             f"#define PIPERLITE_PF_KERNEL {pf_kernel}",
             f"#define PIPERLITE_PF_SCALE {pf_scale}f",
             f"#define PIPERLITE_HOP {HOP}", ""]
    for name, off, size in offsets:
        lines.append(f"#define PIPERLITE_OFF_{cname(name)} {off} /* {size} floats */")
    lines.append(f"\n#define PIPERLITE_WEIGHT_FLOATS {cursor}")
    (args.out / "piperlite_meta.h").write_text("\n".join(lines) + "\n")

    # golden forward on a real latent
    z = load_latent(args.pack, in_ch, args.chunk_row, args.max_frames)
    zt = torch.from_numpy(z).unsqueeze(0)
    with torch.no_grad():
        audio, feats = dec(zt, return_features=True)

    def dump(name: str, t: torch.Tensor) -> list[int]:
        arr = t.detach().squeeze(0).numpy().astype("<f4")
        (args.out / f"{name}.bin").write_bytes(arr.tobytes())
        return list(arr.shape)

    shapes = {"z": dump("z", zt), "audio": dump("audio", audio)}
    for stage in ("pre", "up0", "stage0_mix", "up1", "stage1_mix", "up2",
                  "stage2_mix", "pre_tanh", "audio_pre_filter"):
        shapes[stage] = dump(stage, feats[stage])

    dims = {"in_ch": in_ch, "c0": c0, "c1": c1, "c2": c2, "c3": c3,
            "pf_channels": pf_ch, "pf_layers": pf_layers, "pf_kernel": pf_kernel,
            "pf_scale": pf_scale, "hop": HOP, "frames": int(z.shape[1]),
            "weight_floats": cursor}
    (args.out / "golden.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "dims": dims, "shapes": shapes,
         "tensors": [{"name": n, "offset": o, "floats": s} for n, o, s in offsets]},
        indent=1))
    print(json.dumps({"out": str(args.out), "weight_floats": cursor,
                      "frames": int(z.shape[1]),
                      "audio_samples": int(audio.shape[-1]), "dims": dims}))


if __name__ == "__main__":
    main()
