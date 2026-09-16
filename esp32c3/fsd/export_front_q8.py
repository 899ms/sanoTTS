"""Export duration + acoustic students (int8) and an end-to-end golden.

The E2E golden is the full deployed R7 c-line stack on the heldout12 row-1
sentence: phoneme IDs -> duration student (length scale 1.08) -> acoustic
(token_context) -> LrcEncoder c -> lrc decoder audio. Embeddings stay f32;
all convs are per-output-channel symmetric int8 with rows padded to n16.
k5 convs export as [out][in*k] rows matching a ch-major/k-minor gather.

Run from repo root: .venv/bin/python esp32c3/fsd/export_front_q8.py
"""

import json
import math
import pathlib
import sys

pathlib.WindowsPath = pathlib.PosixPath

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import importlib.util


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dm = load_mod("dm", REPO / "tools/train_roota_piper_decoder_student.py")
latent = load_mod("latent_mod", REPO / "tools/train_roota_piper_latent_student.py")
duration = load_mod("duration_mod", REPO / "tools/train_roota_piper_duration_student.py")

B = REPO / "artifacts/sub10m-search/root-a-piper-vits"
DEC_CK = B / "en_US-kristin-u600-r7-14k-20260703/joint-14k-step-60000/decoder-student.pt"
AC_CK = B / "en_US-kristin-u600-r7-14k-20260703/joint-14k-step-60000/latent-student.pt"
DUR_CK = B / "en_US-kristin-medium-u600-dur-h32d3-train2048-4000-20260702/duration-student.pt"
PACK = B / "en_US-kristin-medium-heldout12-decoder-piper-native-20260702"
OUT = pathlib.Path(__file__).resolve().parent / "golden"
LENGTH_SCALE = 1.08
KEYS = {
    "in_channels", "channels", "res_layers", "variant", "rank_ratio", "activation",
    "post_filter_channels", "post_filter_layers", "post_filter_kernel",
    "post_filter_scale", "istft_n_fft", "fsd_dim", "fsd_blocks", "fsd_film_rank",
    "fsd_head_rank", "stage_projection_bottlenecks",
}


def n16(x: int) -> int:
    return (x + 15) // 16 * 16


blob = bytearray()
offsets: dict[str, int] = {}
meta: dict[str, int | float] = {}


def align(k: int) -> None:
    while len(blob) % k:
        blob.append(0)


def add_q8(name: str, w: torch.Tensor, b: torch.Tensor) -> None:
    wf = w.detach().float().reshape(w.shape[0], -1).numpy()
    out_ch, in_flat = wf.shape
    pad = n16(in_flat)
    scales = np.abs(wf).max(axis=1) / 127.0
    scales[scales == 0.0] = 1.0
    q = np.clip(np.round(wf / scales[:, None]), -127, 127).astype(np.int8)
    qp = np.zeros((out_ch, pad), dtype=np.int8)
    qp[:, :in_flat] = q
    align(16)
    offsets[f"{name}_w8"] = len(blob)
    blob.extend(qp.tobytes())
    align(4)
    offsets[f"{name}_scale"] = len(blob)
    blob.extend(scales.astype("<f4").tobytes())
    offsets[f"{name}_bias"] = len(blob)
    blob.extend(b.detach().float().numpy().astype("<f4").tobytes())
    meta[f"{name.upper()}_N16"] = pad


def add_f32(name: str, t: torch.Tensor) -> None:
    align(4)
    offsets[f"{name}_f32"] = len(blob)
    blob.extend(t.detach().float().contiguous().numpy().astype("<f4").tobytes())


def main() -> None:
    ac, ac_cfg = latent.load_model_from_checkpoint(AC_CK, torch.device("cpu"))
    ac.eval()
    dur, dur_cfg = duration.load_model_from_checkpoint(DUR_CK, torch.device("cpu"))
    dur.eval()
    ck = torch.load(DEC_CK, map_location="cpu", weights_only=False)
    dec = dm.DecoderStudent(**{k: v for k, v in ck["config"].items() if k in KEYS})
    dec.load_state_dict(ck["model_state_dict"], strict=True)
    enc = dm.LrcEncoder(in_channels=192, hidden=64, code_dim=40)
    enc.load_state_dict(ck["lrc_encoder_state_dict"], strict=True)
    dec.eval()
    enc.eval()

    meta["AC_VOCAB"] = int(ac_cfg["vocab_size"])
    meta["AC_HIDDEN"] = int(ac_cfg["hidden"])
    meta["AC_DEPTH"] = int(ac_cfg["depth"])
    meta["AC_TOKEN_DEPTH"] = int(ac_cfg["token_depth"])
    meta["AC_KERNEL"] = int(ac_cfg["kernel_size"])
    meta["AC_OUT"] = int(ac_cfg["out_channels"])
    meta["DUR_VOCAB"] = int(dur_cfg["vocab_size"])
    meta["DUR_HIDDEN"] = int(dur_cfg["hidden"])
    meta["DUR_DEPTH"] = int(dur_cfg["depth"])
    meta["DUR_KERNEL"] = int(dur_cfg["kernel_size"])
    meta["DUR_MAX_TOKENS"] = int(dur_cfg["max_tokens"])
    meta["DUR_MAX_DURATION"] = int(dur_cfg.get("max_duration", 80))

    # duration student
    add_f32("dur_emb", dur.embedding.weight)
    add_q8("dur_proj", dur.input_proj.weight.squeeze(-1), dur.input_proj.bias)
    for i, blk in enumerate(dur.blocks):
        add_q8(f"dur_b{i}_c0", blk.net[0].weight, blk.net[0].bias)
        add_q8(f"dur_b{i}_c1", blk.net[2].weight, blk.net[2].bias)
        add_f32(f"dur_b{i}_scale", blk.scale.reshape(1))
    add_q8("dur_out", dur.output.weight.squeeze(-1), dur.output.bias)

    # acoustic student
    add_f32("ac_emb", ac.embedding.weight)
    add_q8("ac_tproj", ac.token_input_proj.weight.squeeze(-1), ac.token_input_proj.bias)
    for i, blk in enumerate(ac.token_blocks):
        add_q8(f"ac_tb{i}_c0", blk.net[0].weight, blk.net[0].bias)
        add_q8(f"ac_tb{i}_c1", blk.net[2].weight, blk.net[2].bias)
        add_f32(f"ac_tb{i}_scale", blk.scale.reshape(1))
    add_q8("ac_fproj", ac.frame_input_proj.weight.squeeze(-1), ac.frame_input_proj.bias)
    for i, blk in enumerate(ac.frame_blocks):
        add_q8(f"ac_fb{i}_c0", blk.net[0].weight, blk.net[0].bias)
        add_q8(f"ac_fb{i}_c1", blk.net[2].weight, blk.net[2].bias)
        add_f32(f"ac_fb{i}_scale", blk.scale.reshape(1))
    add_q8("ac_out", ac.output.weight.squeeze(-1), ac.output.bias)

    (OUT / "front_q8.bin").write_bytes(bytes(blob))
    lines = ["/* generated by export_front_q8.py -- byte offsets into front_q8.bin */",
             "#pragma once", ""]
    for key, value in meta.items():
        lines.append(f"#define FRONT_{key} {value}")
    lines.append("")
    for name, off in offsets.items():
        lines.append(f"#define FOFF_{name.upper()} {off}")
    lines.append(f"\n#define FRONT_MODEL_BYTES {len(blob)}")
    (OUT / "front_q8_meta.h").write_text("\n".join(lines) + "\n")

    # ---- E2E golden ----
    rows = json.loads((PACK / "rows.json").read_text())
    chunk = rows[0]["chunks"][0]
    ids_list = None
    for key in ("phoneme_ids", "piper_ids", "ids"):
        if key in chunk:
            ids_list = list(chunk[key])
            break
    if ids_list is None:
        npz = np.load(REPO / chunk["tensor_npz"])
        ids_list = np.asarray(npz["phoneme_ids"]).reshape(-1).tolist()
    ids = torch.as_tensor([ids_list], dtype=torch.long)
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        durs = duration.predict_durations(
            dur, ids, mask,
            max_duration=int(meta["DUR_MAX_DURATION"]),
            length_scale=LENGTH_SCALE,
        ).squeeze(0)
        sample = latent.ChunkSample(
            row_id="e2e", row_index=1, text=str(rows[0]["text"]), chunk_index=0,
            phoneme_ids=np.asarray(ids_list, dtype=np.int64),
            durations=durs.numpy().astype(np.int64),
            target=np.zeros((int(durs.sum()), int(meta["AC_OUT"])), dtype=np.float32),
            tensor_path=pathlib.Path("e2e"), audio_samples=0,
        )
        feats = latent.expand_features(sample, torch.device("cpu"))
        z = ac(feats)                          # [frames, out]
        c = z.transpose(0, 1).unsqueeze(0)     # [1, out, T]
        if int(c.shape[1]) != 40:
            c = enc(c)                         # legacy 192-out acoustics
        audio = dec(c)

    (OUT / "e2e_ids.bin").write_bytes(np.asarray(ids_list, dtype="<i4").tobytes())
    (OUT / "e2e_durs.bin").write_bytes(durs.numpy().astype("<i4").tobytes())
    (OUT / "e2e_c.bin").write_bytes(c.squeeze(0).numpy().astype("<f4").tobytes())
    (OUT / "e2e_audio.bin").write_bytes(audio.squeeze().numpy().astype("<f4").tobytes())
    print(json.dumps({
        "front_q8_bytes": len(blob), "tokens": len(ids_list),
        "frames": int(durs.sum()), "audio_samples": int(audio.numel()),
        "text": str(rows[0]["text"])[:80], "meta": {k: v for k, v in meta.items() if not k.endswith("_N16")},
    }))


if __name__ == "__main__":
    main()
