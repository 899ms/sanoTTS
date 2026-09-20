"""Export a saanoTTS front-half (duration + acoustic student) pair for the C99 runtime.

The "front half" is the text side of the pipeline: phoneme IDs -> per-token
frame counts (DurationStudent, tools/train_roota_piper_duration_student.py)
-> expanded features -> latent [192, T] (ContextualLatentStudent
"token_context", tools/train_roota_piper_latent_student.py, optionally wrapped
in a CalibratedLatentStudent output adapter). The golden latent is exactly what
latent_mod.predict_chunk returns (the tensor the piperlite decoder consumes);
golden durations come from duration_mod.predict_durations, i.e. the
DashboardState duration_source="student" inference path (exp -> clamp_min(1)
-> * length_scale -> torch.round -> clamp(1, max_duration)).

Writes to --out:
  front_weights_f32.bin -- all weights, float32 LE, fixed documented order (below)
  meta.bin              -- binary meta (dims + per-tensor offset table)
  front_meta.h          -- same info as macros, for MCU builds baking in one model
  ids.bin               -- golden phoneme ids, int32 LE [N]
  durations.bin         -- golden per-token frame counts at length_scale 1.0, int32 LE [N]
  durations_ls125.bin   -- same at length_scale 1.25 (exercises the runtime param)
  dur_log.bin           -- duration log-duration output pre exp/round, f32 [N]
  dur_feats.bin         -- duration input features (positions, length_hint, valid), f32 [3, N]
  tok_ctx.bin           -- acoustic token context after token blocks, f32 [hidden, N]
  frame_feats.bin       -- expanded (frame_pos, token_pos, duration_pos), f32 [3, T]
  latent_base.bin       -- pre-adapter latent, f32 [C, T] (calibrated checkpoints only)
  latent.bin            -- final latent, f32 [C, T] (channel-major, decoder layout)
  golden.json           -- dims + shapes + provenance

Golden input ids come from either a real eval pack row (--pack) or a piper
teacher voice phonemizing --text (--piper-model/--piper-config), matching how
the dashboard produces ids for arbitrary text.

WEIGHT ORDER (float32, contiguous, PyTorch layouts kept as-is):
  duration model (DurationStudent):
    slot 0 dur.embedding.weight      [d_vocab, d_hidden]
    slot 1 dur.input_proj.weight     [d_hidden, d_hidden+3, 1]
    slot 2 dur.input_proj.bias       [d_hidden]
    per block b in 0..d_depth-1 (5 slots each):
      dur.blocks.b.scale             [1]  (learned scalar)
      dur.blocks.b.net.0.weight      [h, h, k]   conv1
      dur.blocks.b.net.0.bias        [h]
      dur.blocks.b.net.2.weight      [h, h, k]   conv2
      dur.blocks.b.net.2.bias        [h]
    dur.output.weight                [1, d_hidden, 1]
    dur.output.bias                  [1]
  acoustic model (ContextualLatentStudent, state-dict prefix "base." when
  wrapped in a CalibratedLatentStudent):
    ac.embedding.weight              [a_vocab, a_hidden]
    ac.token_input_proj.weight       [h, h+2, 1]
    ac.token_input_proj.bias         [h]
    token blocks x token_depth       (scale, conv1 w/b, conv2 w/b -- as above)
    ac.frame_input_proj.weight       [h, h+3, 1]
    ac.frame_input_proj.bias         [h]
    frame blocks x depth             (5 slots each, as above)
    ac.output.weight                 [C, h, 1]
    ac.output.bias                   [C]
  output adapter (adapter_mode > 0 only; full-channel scope required):
    adapter.depthwise.weight         [C, 1, K]     (modes 2, 4; no bias)
    adapter.lowrank_down.weight      [rank, C, 1]  (modes 3, 4)
    adapter.lowrank_down.bias        [rank]
    adapter.lowrank_up.weight        [C, rank, 1]
    adapter.lowrank_up.bias          [C]
    adapter.scale                    [C]           (all adapter modes)
    adapter.bias                     [C]

META.BIN (little-endian int32 unless noted):
  magic 0x534E4652 ('SNFR'), version=1,
  d_vocab, d_hidden, d_depth, d_kernel, d_max_tokens, d_max_duration,
  a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out_channels,
  adapter_mode (0 none, 1 affine, 2 depthwise, 3 lowrank, 4 depthwise_lowrank),
  adapter_kernel, adapter_rank,
  n_tensors, then n_tensors x (offset_floats, size_floats).

Run from repo root, e.g.:
  .venv/bin/python tools/export_front_golden.py \
      .../a5-duration/duration-student.pt .../e2-wide192-ac64-joint/latent-student.pt \
      --pack .../eval128-piper-native --out mcu/test/golden_front/amy
"""

import argparse
import importlib.util
import json
import math
import pathlib
import struct
import sys

pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[1]


def load_local_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


duration_mod = load_local_module(
    "front_duration_student", REPO / "tools" / "train_roota_piper_duration_student.py"
)
latent_mod = load_local_module(
    "front_latent_student", REPO / "tools" / "train_roota_piper_latent_student.py"
)

MAGIC = 0x534E4652  # 'SNFR'
VERSION = 1
ADAPTER_MODES = {"affine": 1, "depthwise": 2, "lowrank": 3, "depthwise_lowrank": 4}


def check_acoustic_supported(model, config: dict) -> tuple:
    """Return (base_model, state_prefix, adapter, adapter_mode). Fail loudly otherwise."""
    architecture = str(config.get("architecture") or "")
    if architecture == "token_context":
        if not isinstance(model, latent_mod.ContextualLatentStudent):
            raise SystemExit(f"expected ContextualLatentStudent, got {type(model).__name__}")
        return model, "", None, 0
    if architecture != "calibrated":
        raise SystemExit(
            f"acoustic architecture {architecture!r} not supported by the C port "
            "(only token_context, optionally calibrated)"
        )
    if not isinstance(model, latent_mod.CalibratedLatentStudent):
        raise SystemExit(f"expected CalibratedLatentStudent, got {type(model).__name__}")
    base = model.base
    if not isinstance(base, latent_mod.ContextualLatentStudent):
        raise SystemExit(
            f"calibrated base must be token_context, got {type(base).__name__}"
        )
    adapter = model.adapter
    if adapter.start_channel != 0 or adapter.end_channel != adapter.channels:
        raise SystemExit(
            "only full-channel adapters are supported, got slice "
            f"{adapter.start_channel}:{adapter.end_channel} of {adapter.channels}"
        )
    mode = ADAPTER_MODES.get(adapter.mode)
    if mode is None:
        raise SystemExit(f"unsupported adapter mode {adapter.mode!r}")
    return base, "base.", adapter, mode


def golden_ids_from_pack(pack: pathlib.Path, chunk_row: int) -> tuple[np.ndarray, str]:
    rows = json.loads((pack / "rows.json").read_text())
    row = rows[chunk_row]
    chunk = row["chunks"][0]
    npz_path = pathlib.Path(str(chunk["tensor_npz"]))
    if not npz_path.is_absolute():
        npz_path = REPO / npz_path
    with np.load(npz_path) as npz:
        if "phoneme_ids" not in npz:
            raise SystemExit(f"{npz_path} has no 'phoneme_ids'")
        ids = np.asarray(npz["phoneme_ids"], dtype=np.int64).reshape(-1)
    if ids.size <= 0:
        raise SystemExit(f"{npz_path}: empty phoneme_ids")
    return ids, str(row.get("text") or "")


def golden_ids_from_piper(model_path: pathlib.Path, config_path: pathlib.Path,
                          text: str) -> np.ndarray:
    pack_mod = load_local_module(
        "front_pack_builder", REPO / "tools" / "build_piper_vits_roota_probe_pack.py"
    )
    voice = pack_mod.PiperVoice.load(str(model_path), str(config_path))
    sentence_phonemes = voice.phonemize(text)
    if not sentence_phonemes:
        raise SystemExit(f"piper produced no sentence phonemes for {text!r}")
    ids: list[int] = []
    for phonemes in sentence_phonemes:
        chunk_ids = voice.phonemes_to_ids(phonemes)
        if not chunk_ids:
            raise SystemExit(f"piper produced empty phoneme ids for {phonemes!r}")
        ids.extend(int(v) for v in chunk_ids)
    return np.asarray(ids, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("duration_checkpoint", type=pathlib.Path)
    ap.add_argument("acoustic_checkpoint", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--pack", type=pathlib.Path, default=None,
                    help="eval pack dir with rows.json (real phoneme_ids input)")
    ap.add_argument("--chunk-row", type=int, default=0)
    ap.add_argument("--piper-model", type=pathlib.Path, default=None,
                    help="piper teacher .onnx (id generation via phonemes_to_ids)")
    ap.add_argument("--piper-config", type=pathlib.Path, default=None)
    ap.add_argument("--text", type=str, default=None,
                    help="text to phonemize with --piper-model")
    args = ap.parse_args()

    if (args.pack is None) == (args.piper_model is None):
        raise SystemExit("provide exactly one of --pack or --piper-model/--piper-config/--text")
    if args.piper_model is not None and (args.piper_config is None or not args.text):
        raise SystemExit("--piper-model requires --piper-config and --text")

    torch.manual_seed(0)
    device = torch.device("cpu")

    dur_model, dur_config = duration_mod.load_model_from_checkpoint(
        args.duration_checkpoint, device
    )
    dur_model.eval()
    if str(dur_config.get("architecture") or "duration_conv") != "duration_conv":
        raise SystemExit(f"unsupported duration architecture: {dur_config.get('architecture')!r}")
    d_vocab = int(dur_config["vocab_size"])
    d_hidden = int(dur_config["hidden"])
    d_depth = int(dur_config["depth"])
    d_kernel = int(dur_config["kernel_size"])
    d_max_tokens = int(dur_config["max_tokens"])
    d_max_duration = int(dur_config.get("max_duration", 80))

    ac_model, ac_config = latent_mod.load_model_from_checkpoint(
        args.acoustic_checkpoint, device
    )
    ac_model.eval()
    base_config = ac_config.get("base_config") if isinstance(
        ac_config.get("base_config"), dict) else ac_config
    ac_base, ac_prefix, adapter, adapter_mode = check_acoustic_supported(ac_model, ac_config)
    a_vocab = int(base_config["vocab_size"])
    a_hidden = int(base_config["hidden"])
    a_token_depth = int(base_config["token_depth"])
    a_depth = int(base_config["depth"])
    a_kernel = int(base_config["kernel_size"])
    a_out = int(base_config["out_channels"])
    adapter_kernel = int(adapter.kernel_size) if adapter is not None else 0
    adapter_rank = int(adapter.rank) if adapter is not None else 0

    # ---- weight export, fixed slot order --------------------------------
    dur_sd = {k: v.detach().float() for k, v in dur_model.state_dict().items()}
    ac_sd = {k: v.detach().float() for k, v in ac_model.state_dict().items()}
    order: list[tuple[str, torch.Tensor]] = []

    def add(sd: dict, name: str, shape: tuple) -> None:
        if name not in sd:
            raise SystemExit(f"missing state-dict tensor: {name}")
        t = sd[name].reshape(-1) if shape == (1,) and sd[name].dim() == 0 else sd[name]
        if tuple(t.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(t.shape)} != expected {shape}")
        order.append((name, t.contiguous()))

    def add_blocks(sd: dict, prefix: str, count: int, h: int, k: int) -> None:
        for b in range(count):
            add(sd, f"{prefix}.{b}.scale", (1,))
            add(sd, f"{prefix}.{b}.net.0.weight", (h, h, k))
            add(sd, f"{prefix}.{b}.net.0.bias", (h,))
            add(sd, f"{prefix}.{b}.net.2.weight", (h, h, k))
            add(sd, f"{prefix}.{b}.net.2.bias", (h,))

    add(dur_sd, "embedding.weight", (d_vocab, d_hidden))
    add(dur_sd, "input_proj.weight", (d_hidden, d_hidden + 3, 1))
    add(dur_sd, "input_proj.bias", (d_hidden,))
    add_blocks(dur_sd, "blocks", d_depth, d_hidden, d_kernel)
    add(dur_sd, "output.weight", (1, d_hidden, 1))
    add(dur_sd, "output.bias", (1,))
    n_dur_slots = len(order)

    p = ac_prefix
    add(ac_sd, f"{p}embedding.weight", (a_vocab, a_hidden))
    add(ac_sd, f"{p}token_input_proj.weight", (a_hidden, a_hidden + 2, 1))
    add(ac_sd, f"{p}token_input_proj.bias", (a_hidden,))
    add_blocks(ac_sd, f"{p}token_blocks", a_token_depth, a_hidden, a_kernel)
    add(ac_sd, f"{p}frame_input_proj.weight", (a_hidden, a_hidden + 3, 1))
    add(ac_sd, f"{p}frame_input_proj.bias", (a_hidden,))
    add_blocks(ac_sd, f"{p}frame_blocks", a_depth, a_hidden, a_kernel)
    add(ac_sd, f"{p}output.weight", (a_out, a_hidden, 1))
    add(ac_sd, f"{p}output.bias", (a_out,))
    if adapter_mode in (2, 4):
        add(ac_sd, "adapter.depthwise.weight", (a_out, 1, adapter_kernel))
    if adapter_mode in (3, 4):
        add(ac_sd, "adapter.lowrank_down.weight", (adapter_rank, a_out, 1))
        add(ac_sd, "adapter.lowrank_down.bias", (adapter_rank,))
        add(ac_sd, "adapter.lowrank_up.weight", (a_out, adapter_rank, 1))
        add(ac_sd, "adapter.lowrank_up.bias", (a_out,))
    if adapter_mode > 0:
        add(ac_sd, "adapter.scale", (a_out,))
        add(ac_sd, "adapter.bias", (a_out,))

    # duration + acoustic keys can collide by name; check coverage per-dict
    dur_left = sorted(set(dur_sd) - {n for n, _ in order[:n_dur_slots]})
    ac_left = sorted(set(ac_sd) - {n for n, _ in order[n_dur_slots:]})
    if dur_left or ac_left:
        raise SystemExit(
            f"state dict tensors not covered by export order: duration={dur_left} acoustic={ac_left}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    offsets: list[tuple[str, int, int]] = []
    cursor = 0
    with (args.out / "front_weights_f32.bin").open("wb") as fh:
        for name, t in order:
            arr = t.numpy().astype("<f4")
            offsets.append((name, cursor, arr.size))
            fh.write(arr.tobytes())
            cursor += arr.size

    header = [MAGIC, VERSION,
              d_vocab, d_hidden, d_depth, d_kernel, d_max_tokens, d_max_duration,
              a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out,
              adapter_mode, adapter_kernel, adapter_rank,
              len(offsets)]
    with (args.out / "meta.bin").open("wb") as fh:
        fh.write(struct.pack(f"<{len(header)}i", *header))
        for _, off, size in offsets:
            fh.write(struct.pack("<2i", off, size))

    def cname(name: str) -> str:
        return name.upper().replace(".", "_")

    lines = ["/* generated by tools/export_front_golden.py -- do not edit */",
             "#pragma once", "",
             f"#define FRONT_D_VOCAB {d_vocab}",
             f"#define FRONT_D_HIDDEN {d_hidden}",
             f"#define FRONT_D_DEPTH {d_depth}",
             f"#define FRONT_D_KERNEL {d_kernel}",
             f"#define FRONT_D_MAX_TOKENS {d_max_tokens}",
             f"#define FRONT_D_MAX_DURATION {d_max_duration}",
             f"#define FRONT_A_VOCAB {a_vocab}",
             f"#define FRONT_A_HIDDEN {a_hidden}",
             f"#define FRONT_A_TOKEN_DEPTH {a_token_depth}",
             f"#define FRONT_A_DEPTH {a_depth}",
             f"#define FRONT_A_KERNEL {a_kernel}",
             f"#define FRONT_A_OUT_CHANNELS {a_out}",
             f"#define FRONT_ADAPTER_MODE {adapter_mode}",
             f"#define FRONT_ADAPTER_KERNEL {adapter_kernel}",
             f"#define FRONT_ADAPTER_RANK {adapter_rank}", ""]
    prefix_tag = ["DUR"] * n_dur_slots + ["AC"] * (len(offsets) - n_dur_slots)
    for tag, (name, off, size) in zip(prefix_tag, offsets):
        lines.append(f"#define FRONT_OFF_{tag}_{cname(name)} {off} /* {size} floats */")
    lines.append(f"\n#define FRONT_WEIGHT_FLOATS {cursor}")
    (args.out / "front_meta.h").write_text("\n".join(lines) + "\n")

    # ---- golden inputs ---------------------------------------------------
    if args.pack is not None:
        ids_np, text = golden_ids_from_pack(args.pack, args.chunk_row)
        id_source = {"kind": "pack", "pack": str(args.pack), "chunk_row": args.chunk_row,
                     "text": text}
    else:
        ids_np = golden_ids_from_piper(args.piper_model, args.piper_config, args.text)
        id_source = {"kind": "piper", "model": str(args.piper_model), "text": args.text}
    max_id = int(ids_np.max())
    if max_id >= d_vocab or max_id >= a_vocab:
        raise SystemExit(
            f"golden ids exceed vocab: max id {max_id}, duration vocab {d_vocab}, "
            f"acoustic vocab {a_vocab}"
        )
    (args.out / "ids.bin").write_bytes(ids_np.astype("<i4").tobytes())
    n_tokens = int(ids_np.size)

    # ---- golden durations (dashboard student inference path) --------------
    ids_t = torch.as_tensor(ids_np[None, :], dtype=torch.long, device=device)
    mask_t = torch.ones_like(ids_t, dtype=torch.bool)
    with torch.no_grad():
        dur_log = dur_model(ids_t, mask_t).squeeze(0)
        durations = {}
        for tag, ls in (("durations", 1.0), ("durations_ls125", 1.25)):
            pred = duration_mod.predict_durations(
                dur_model, ids_t, mask_t,
                max_duration=d_max_duration, length_scale=ls,
            ).squeeze(0).cpu().numpy().astype(np.int64)
            if pred.shape != ids_np.shape or np.any(pred <= 0):
                raise SystemExit(f"bad golden durations at length_scale {ls}")
            durations[tag] = pred
            (args.out / f"{tag}.bin").write_bytes(pred.astype("<i4").tobytes())
    (args.out / "dur_log.bin").write_bytes(
        dur_log.detach().cpu().numpy().astype("<f4").tobytes())
    positions = torch.linspace(0.0, 1.0, n_tokens)
    length_hint = torch.log1p(torch.tensor([float(n_tokens)])) / math.log1p(float(d_max_tokens))
    dur_feats = torch.stack([positions,
                             length_hint.expand(n_tokens),
                             torch.ones(n_tokens)], dim=0)
    (args.out / "dur_feats.bin").write_bytes(
        dur_feats.numpy().astype("<f4").tobytes())

    # ---- golden latent (dashboard predict_chunk path) ---------------------
    dur_np = durations["durations"]
    frame_count = int(dur_np.sum())
    sample = latent_mod.ChunkSample(
        row_id="front-golden",
        row_index=1,
        text=str(id_source.get("text") or ""),
        chunk_index=0,
        phoneme_ids=ids_np,
        durations=dur_np,
        target=np.zeros((frame_count, a_out), dtype=np.float32),
        tensor_path=pathlib.Path("front-golden"),
        audio_samples=frame_count * 256,
    )
    features = latent_mod.expand_features(sample, device)
    frame_feats = torch.stack(
        [features.frame_pos, features.token_pos, features.duration_pos], dim=0)
    (args.out / "frame_feats.bin").write_bytes(
        frame_feats.detach().cpu().numpy().astype("<f4").tobytes())

    tok_ctx_holder: list[torch.Tensor] = []

    def tok_hook(_module, _inputs, output):
        tok_ctx_holder.append(output.detach())

    hook = ac_base.token_blocks[-1].register_forward_hook(tok_hook)
    with torch.no_grad():
        latent = latent_mod.predict_latent_tensor(ac_model, features)  # [T, C]
        if adapter is not None:
            latent_base = latent_mod.predict_latent_tensor(ac_base, features)
        else:
            latent_base = None
    hook.remove()
    if latent.shape != (frame_count, a_out):
        raise SystemExit(f"latent shape {tuple(latent.shape)} != ({frame_count}, {a_out})")
    if not tok_ctx_holder:
        raise SystemExit("token-context hook never fired")
    tok_ctx = tok_ctx_holder[0].squeeze(0)  # [hidden, N]
    if tuple(tok_ctx.shape) != (a_hidden, n_tokens):
        raise SystemExit(f"tok_ctx shape {tuple(tok_ctx.shape)} != ({a_hidden}, {n_tokens})")
    (args.out / "tok_ctx.bin").write_bytes(
        tok_ctx.detach().cpu().numpy().astype("<f4").tobytes())
    latent_cm = latent.transpose(0, 1).contiguous()  # [C, T], decoder layout
    (args.out / "latent.bin").write_bytes(
        latent_cm.detach().cpu().numpy().astype("<f4").tobytes())
    if latent_base is not None:
        (args.out / "latent_base.bin").write_bytes(
            latent_base.transpose(0, 1).contiguous().detach().cpu().numpy()
            .astype("<f4").tobytes())

    dims = {"d_vocab": d_vocab, "d_hidden": d_hidden, "d_depth": d_depth,
            "d_kernel": d_kernel, "d_max_tokens": d_max_tokens,
            "d_max_duration": d_max_duration,
            "a_vocab": a_vocab, "a_hidden": a_hidden, "a_token_depth": a_token_depth,
            "a_depth": a_depth, "a_kernel": a_kernel, "a_out_channels": a_out,
            "adapter_mode": adapter_mode, "adapter_kernel": adapter_kernel,
            "adapter_rank": adapter_rank, "weight_floats": cursor}
    (args.out / "golden.json").write_text(json.dumps(
        {"duration_checkpoint": str(args.duration_checkpoint),
         "acoustic_checkpoint": str(args.acoustic_checkpoint),
         "id_source": id_source,
         "dims": dims,
         "n_tokens": n_tokens,
         "frames": frame_count,
         "frames_ls125": int(durations["durations_ls125"].sum()),
         "tensors": [{"name": n, "offset": o, "floats": s} for n, o, s in offsets]},
        indent=1))
    print(json.dumps({"out": str(args.out), "weight_floats": cursor,
                      "n_tokens": n_tokens, "frames": frame_count, "dims": dims}))


if __name__ == "__main__":
    main()
