"""Export a piperlite front half (duration + acoustic student) as int8.

Companion to tools/export_front_golden.py -- run that FIRST into the same
--out dir. It writes the fp32 reference (meta.bin, front_weights_f32.bin,
ids.bin, durations.bin, latent.bin) that both the C gate and this exporter's
own self-check compare against. This script adds:

  front_weights_q8.bin -- pure int8 weight payload, same slot order as the
                          fp32 exporter (this is the distribution blob)
  front_meta_q8.bin    -- dims + adapter shape + STATIC activation clips
                          (calibrated over real phoneme ids) + per-slot
                          records:
                            kind 0 = int8 conv weights (offset/size into
                                     front_weights_q8.bin, per-output-channel
                                     fp32 scales in the pool)
                            kind 1 = fp32 data (bias / learned scalar, pool)
                            kind 2 = int8 embedding table (per-ROW scales;
                                     one scale per vocabulary entry)
  front_calib_q8.json  -- calibrated ranges + the exporter's own numpy
                          self-check, for the record

Why the SAME code path serves R7 and piperlite: esp32c3/fsd/export_front_q8.py
proved the int8 front on the R7 lineage, but baked every dimension into a
generated header, so it only ever fit the one model it was written for. The
two lineages are the same architecture family -- a `token_context` acoustic
student (embedding -> token_input_proj -> token blocks -> repeat_interleave ->
frame_input_proj -> frame blocks -> output 1x1) plus a conv DurationStudent --
differing only in dims (R7: a_vocab 157 / a_hidden 48 / a_depth 5 / a_out 40;
amy: 145 / 64 / 4 / 192) and in piperlite's optional output adapter. So this is
a parameterisation of the same quantisation, with the dims moved into the blob
where mcu/src/snt_front_q8.c reads them at runtime.

front_meta_q8.bin layout (little-endian), parsed by mcu/src/snt_front_q8.c:
  i32 magic 0x534E4651 'SNFQ', i32 version=1,
  i32 d_vocab, d_hidden, d_depth, d_kernel, d_max_tokens, d_max_duration,
  i32 a_vocab, a_hidden, a_token_depth, a_depth, a_kernel, a_out,
  i32 adapter_mode, adapter_kernel, adapter_rank,
  i32 n_tensors, i32 n_act,
  f32 act_clip[n_act],
  n_tensors x (i32 kind, i32 offset, i32 size, i32 aux_off, i32 aux_n),
  i32 pool_n, f32 pool[pool_n].

act_clip holds the CALIBRATED CLIP in real units, not a quantisation step:
the runtime divides by its own lane maximum. That keeps the blob independent
of how wide the runtime's activation lane happens to be (12-bit by default,
see mcu/include/snt_front_q8.h), so a blob exported today still means the
same thing if the lane changes.

Activation clip order (n_act = 6 + 2*(d_depth + a_token_depth + a_depth)):
  [0] d_feat   duration input plane (embedding rows + 3 positional channels)
  [1] d_in     after input_proj
  block b at 2+2b:  +0 silu output, +1 block output (after the residual add)
  then, at 2+2*d_depth:
  [+0] a_tfeat token input plane (embedding rows + token_pos + duration hint)
  [+1] a_tin   after token_input_proj
  token block b: +0 silu output, +1 block output
  then, after the token blocks:
  [+0] a_ffeat frame input plane (expanded token context + 3 positional)
  [+1] a_fin   after frame_input_proj
  frame block b: +0 silu output, +1 block output
The duration logits and the acoustic output 1x1 dequantise straight to fp32
(they are the last op before, respectively, exp/round and the decoder), and
the optional output adapter runs fp32 with its weights dequantised on the
fly -- the same call export_piperlite_q8.py makes for the waveform post
filter, and for the same reason: it is a handful of channels' worth of work
and it sits where the error would be least recoverable.

Inputs. Either a pair of checkpoints, or a shipped roota.raw-fp16.v1 package
(--package), which is repacked in memory by tools/repack_package_to_checkpoints
so a user holding only a published voice can still get here.

Calibration ids come from a real eval pack (--pack) or from a piper teacher
phonemizing --calib-texts, matching how the dashboard produces ids.

Run from repo root, e.g.:
  venv/bin/python tools/export_front_golden.py DUR.pt AC.pt --pack PACK --out OUT
  venv/bin/python tools/export_front_q8.py DUR.pt AC.pt --pack PACK --out OUT
or straight from a package:
  venv/bin/python tools/export_front_q8.py --package artifacts/voices/de_DE/package \\
      --piper-model models/teachers/de_DE-thorsten-medium/de_DE-thorsten-medium.onnx \\
      --calib-texts texts.json --out OUT
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


def _load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


golden = _load_module("front_golden", REPO / "tools/export_front_golden.py")
duration_mod = golden.duration_mod
latent_mod = golden.latent_mod
plq = _load_module("piperlite_q8", REPO / "tools/export_piperlite_q8.py")
repack = _load_module("repack_pkg", REPO / "tools/repack_package_to_checkpoints.py")

MAGIC = 0x534E4651  # 'SNFQ'
VERSION = 1
KIND_W8 = 0
KIND_F32 = 1
KIND_EMB8 = 2
# The reference lane the exporter calibrates against. The runtime carries its
# own copy (SNT_FRONT_Q8_QMAX) and the two are gated against each other by
# mcu/test/front_q8_golden_test.c; the blob stores clips, not steps, so a
# mismatch degrades nothing silently -- it simply is not what was calibrated.
CALIB_QMAX = 2047


# --------------------------------------------------------------------------
# weight quantisation
# --------------------------------------------------------------------------

def quant_per_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 with one scale per row of the leading axis.

    For a Conv1d weight [out, in, K] that is per-output-channel; for an
    embedding table [vocab, hidden] it is per-vocabulary-entry, which is the
    granularity the runtime gathers at anyway, so it costs nothing to keep.
    """
    flat = w.reshape(w.shape[0], -1)
    amax = np.abs(flat).max(axis=1)
    scales = amax / 127.0
    scales[scales == 0.0] = 1.0
    q = np.clip(np.round(flat / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def dequant_per_row(q: np.ndarray, scales: np.ndarray, shape: tuple) -> np.ndarray:
    return (q.astype(np.float32) * scales[:, None]).reshape(shape)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def act_names(d_depth: int, a_token_depth: int, a_depth: int) -> list[str]:
    """The planes the int8 runtime carries, in blob order.

    The projection outputs are named `*_in`, NOT `*_x0`: an earlier draft
    called them d_x0/a_tx0/a_fx0, which collides with block 0's own output
    name, and the calibration dict then silently merged two different tensors
    onto one clip. Keep these distinct.
    """
    names = ["d_feat", "d_in"]
    for b in range(d_depth):
        names += [f"d_s{b}", f"d_x{b}"]
    names += ["a_tfeat", "a_tin"]
    for b in range(a_token_depth):
        names += [f"a_ts{b}", f"a_tx{b}"]
    names += ["a_ffeat", "a_fin"]
    for b in range(a_depth):
        names += [f"a_fs{b}", f"a_fx{b}"]
    if len(names) != len(set(names)):
        raise SystemExit(f"duplicate activation names: {names}")
    return names


def _hook_front(dur_model, ac_base, note):
    """Register the taps that name every plane the int8 runtime carries."""
    handles = []

    def out_hook(name):
        return lambda _m, _i, out: note(name, out)

    def in_hook(name):
        return lambda _m, inp: note(name, inp[0])

    handles.append(dur_model.input_proj.register_forward_pre_hook(in_hook("d_feat")))
    handles.append(dur_model.input_proj.register_forward_hook(out_hook("d_in")))
    for b, blk in enumerate(dur_model.blocks):
        handles.append(blk.net[1].register_forward_hook(out_hook(f"d_s{b}")))
        handles.append(blk.register_forward_hook(out_hook(f"d_x{b}")))
    handles.append(ac_base.token_input_proj.register_forward_pre_hook(in_hook("a_tfeat")))
    handles.append(ac_base.token_input_proj.register_forward_hook(out_hook("a_tin")))
    for b, blk in enumerate(ac_base.token_blocks):
        handles.append(blk.net[1].register_forward_hook(out_hook(f"a_ts{b}")))
        handles.append(blk.register_forward_hook(out_hook(f"a_tx{b}")))
    handles.append(ac_base.frame_input_proj.register_forward_pre_hook(in_hook("a_ffeat")))
    handles.append(ac_base.frame_input_proj.register_forward_hook(out_hook("a_fin")))
    for b, blk in enumerate(ac_base.frame_blocks):
        handles.append(blk.net[1].register_forward_hook(out_hook(f"a_fs{b}")))
        handles.append(blk.register_forward_hook(out_hook(f"a_fx{b}")))
    return handles


SUBSAMPLE_STRIDE = 97  # prime, decorrelates from channel/time layout


def calibrate(dur_model, ac_model, ac_base, id_rows: list[np.ndarray],
              max_duration: int, length_scale: float, a_out: int,
              expected: list[str]) -> tuple[dict, dict]:
    amax: dict[str, float] = {}
    samples: dict[str, list[np.ndarray]] = {}

    def note(name: str, t: torch.Tensor) -> None:
        flat = t.detach().reshape(-1)
        v = float(flat.abs().max().item())
        if v > amax.get(name, 0.0):
            amax[name] = v
        samples.setdefault(name, []).append(
            flat[::SUBSAMPLE_STRIDE].abs().numpy().astype(np.float32))

    handles = _hook_front(dur_model, ac_base, note)
    device = torch.device("cpu")
    try:
        with torch.no_grad():
            for ids_np in id_rows:
                ids_t = torch.as_tensor(ids_np[None, :], dtype=torch.long)
                mask_t = torch.ones_like(ids_t, dtype=torch.bool)
                durs = duration_mod.predict_durations(
                    dur_model, ids_t, mask_t, max_duration=max_duration,
                    length_scale=length_scale).squeeze(0).cpu().numpy().astype(np.int64)
                frames = int(durs.sum())
                sample = latent_mod.ChunkSample(
                    row_id="calib", row_index=0, text="", chunk_index=0,
                    phoneme_ids=ids_np.astype(np.int64), durations=durs,
                    target=np.zeros((frames, a_out), dtype=np.float32),
                    tensor_path=pathlib.Path("calib"), audio_samples=frames * 256)
                features = latent_mod.expand_features(sample, device)
                latent_mod.predict_latent_tensor(ac_model, features)
    finally:
        for h in handles:
            h.remove()
    missing = [n for n in expected if n not in amax]
    if missing:
        raise SystemExit(f"calibration never saw these tensors: {missing}")
    extra = sorted(set(amax) - set(expected))
    if extra:
        raise SystemExit(f"calibration captured unexpected tensors: {extra}")
    return amax, {k: np.concatenate(v) for k, v in samples.items()}


# --------------------------------------------------------------------------
# numpy reference of the int8 runtime (the exporter's own self-check)
# --------------------------------------------------------------------------

def _q(x: np.ndarray, clip: float, qmax: int) -> np.ndarray:
    s = clip / qmax
    return np.clip(np.rint(x / s), -qmax, qmax) * s


def _conv_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    out_ch, in_ch, k = w.shape
    pad = k // 2
    T = x.shape[1]
    xp = np.pad(x, ((0, 0), (pad, pad)))
    out = np.repeat(b.reshape(-1, 1), T, axis=1).astype(np.float64)
    for j in range(k):
        out += w[:, :, j].astype(np.float64) @ xp[:, j:j + T]
    return out.astype(np.float32)


def _silu(x):
    return x / (1.0 + np.exp(-x))


def simulate_front_q8(deq: dict, dims: dict, clips: dict, ids: np.ndarray,
                      durations: np.ndarray, qmax: int) -> np.ndarray:
    """Latent [C, T] from the dequantised weights with quantised planes.

    This is not the C runtime; it is an independent numpy model of the same
    arithmetic, so a disagreement between the two is a bug in one of them
    rather than a shared assumption.
    """
    h = dims["a_hidden"]
    N = int(ids.size)
    T = int(durations.sum())

    emb = deq["ac.embedding.weight"]                      # [vocab, h]
    tok = emb[ids].T                                      # [h, N]
    token_pos = np.linspace(0.0, 1.0, N, dtype=np.float32) if N > 1 else np.zeros(1, np.float32)
    maxd = max(1.0, float(durations.max()))
    dur_hint = (np.log1p(durations.astype(np.float32)) / np.log1p(maxd)).astype(np.float32)
    feat = np.concatenate([tok, token_pos[None, :], dur_hint[None, :]], axis=0)
    feat = _q(feat, clips["a_tfeat"], qmax)
    x = _conv_same(feat, deq["ac.token_input_proj.weight"], deq["ac.token_input_proj.bias"])
    x = _q(x, clips["a_tin"], qmax)
    for b in range(dims["a_token_depth"]):
        t = _conv_same(x, deq[f"ac.token_blocks.{b}.net.0.weight"],
                       deq[f"ac.token_blocks.{b}.net.0.bias"])
        t = _q(_silu(t), clips[f"a_ts{b}"], qmax)
        u = _conv_same(t, deq[f"ac.token_blocks.{b}.net.2.weight"],
                       deq[f"ac.token_blocks.{b}.net.2.bias"])
        x = _q(x + deq[f"ac.token_blocks.{b}.scale"][0] * u, clips[f"a_tx{b}"], qmax)

    ctx = np.repeat(x, durations, axis=1)                 # [h, T]
    frame_pos = np.linspace(0.0, 1.0, T, dtype=np.float32) if T > 1 else np.zeros(1, np.float32)
    tcount = N - 1 if N > 1 else 1
    tp = np.repeat((np.arange(N, dtype=np.float64) / tcount).astype(np.float32), durations)
    dp = np.concatenate([
        np.zeros(1, np.float32) if d == 1 else
        (np.arange(d, dtype=np.float64) / (d - 1)).astype(np.float32)
        for d in durations])
    feat = np.concatenate([ctx, frame_pos[None, :], tp[None, :], dp[None, :]], axis=0)
    feat = _q(feat, clips["a_ffeat"], qmax)
    x = _conv_same(feat, deq["ac.frame_input_proj.weight"], deq["ac.frame_input_proj.bias"])
    x = _q(x, clips["a_fin"], qmax)
    for b in range(dims["a_depth"]):
        t = _conv_same(x, deq[f"ac.frame_blocks.{b}.net.0.weight"],
                       deq[f"ac.frame_blocks.{b}.net.0.bias"])
        t = _q(_silu(t), clips[f"a_fs{b}"], qmax)
        u = _conv_same(t, deq[f"ac.frame_blocks.{b}.net.2.weight"],
                       deq[f"ac.frame_blocks.{b}.net.2.bias"])
        x = _q(x + deq[f"ac.frame_blocks.{b}.scale"][0] * u, clips[f"a_fx{b}"], qmax)

    latent = _conv_same(x, deq["ac.output.weight"], deq["ac.output.bias"])

    mode = dims["adapter_mode"]
    if mode != 0:
        C = dims["a_out"]
        cur = latent
        if mode in (2, 4):
            dw = deq["adapter.depthwise.weight"]           # [C,1,K]
            k = dw.shape[2]
            pad = k // 2
            xp = np.pad(cur, ((0, 0), (pad, pad)))
            acc = np.zeros_like(cur)
            for j in range(k):
                acc += dw[:, 0, j:j + 1] * xp[:, j:j + T]
            cur = acc
        if mode in (3, 4):
            dwn = deq["adapter.lowrank_down.weight"].reshape(-1, C)
            dup = deq["adapter.lowrank_up.weight"].reshape(C, -1)
            r = np.tanh(dwn @ cur + deq["adapter.lowrank_down.bias"][:, None])
            cur = cur + dup @ r + deq["adapter.lowrank_up.bias"][:, None]
        latent = cur * deq["adapter.scale"][:, None] + deq["adapter.bias"][:, None]
    return latent.astype(np.float32)


def simulate_durations_q8(deq: dict, dims: dict, clips: dict, ids: np.ndarray,
                          length_scale: float, qmax: int) -> np.ndarray:
    h = dims["d_hidden"]
    N = int(ids.size)
    emb = deq["dur.embedding.weight"]
    tok = emb[ids].T
    pos = np.linspace(0.0, 1.0, N, dtype=np.float32) if N > 1 else np.zeros(1, np.float32)
    hint = np.full(N, float(np.float32(math.log1p(N) / math.log1p(dims["d_max_tokens"]))),
                   dtype=np.float32)
    feat = np.concatenate([tok, pos[None, :], hint[None, :],
                           np.ones((1, N), np.float32)], axis=0)
    feat = _q(feat, clips["d_feat"], qmax)
    x = _conv_same(feat, deq["dur.input_proj.weight"], deq["dur.input_proj.bias"])
    x = _q(x, clips["d_in"], qmax)
    for b in range(dims["d_depth"]):
        t = _conv_same(x, deq[f"dur.blocks.{b}.net.0.weight"], deq[f"dur.blocks.{b}.net.0.bias"])
        t = _q(_silu(t), clips[f"d_s{b}"], qmax)
        u = _conv_same(t, deq[f"dur.blocks.{b}.net.2.weight"], deq[f"dur.blocks.{b}.net.2.bias"])
        x = _q(x + deq[f"dur.blocks.{b}.scale"][0] * u, clips[f"d_x{b}"], qmax)
    logits = _conv_same(x, deq["dur.output.weight"], deq["dur.output.bias"]).reshape(-1)
    v = np.maximum(np.exp(logits), 1.0) * length_scale
    v = np.rint(v)
    return np.clip(v, 1.0, float(dims["d_max_duration"])).astype(np.int64)


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------

def load_from_package(package: pathlib.Path):
    """Rebuild duration/acoustic state dicts + configs from a shipped package."""
    manifest = json.loads((package / "manifest.json").read_text())
    if manifest.get("format") != "roota.raw-fp16.v1":
        raise SystemExit(f"unsupported package format {manifest.get('format')!r}")
    blob = (package / manifest["weights_file"]).read_bytes()
    if len(blob) != int(manifest["weights_size_bytes"]):
        raise SystemExit(f"{package}: weights blob size != manifest")
    out = {}
    for name in ("duration", "acoustic"):
        component = manifest["components"][name]
        state = repack.load_component(blob, component, name)
        params = sum(t.numel() for t in state.values())
        if params != int(component["parameters"]):
            raise SystemExit(f"{name}: rebuilt {params} params, manifest says "
                             f"{component['parameters']}")
        out[name] = (state, dict(component["config"]))
    return out, manifest


SUPPORTED_ACOUSTIC = ("token_context", "calibrated")


def _guard_acoustic_config(config: dict, where: str) -> None:
    """Refuse an acoustic config this exporter cannot read, in one line.

    Without this, handing the tool the wrong checkpoint (a decoder, say) blows
    up inside the model factory and prints two kilobytes of training config,
    which reads like a crash rather than like a refusal. Quantising a model you
    are misreading is the failure mode worth spending code on: it produces
    weights that load and a voice that is wrong.
    """
    arch = str(config.get("architecture") or "")
    if arch not in SUPPORTED_ACOUSTIC:
        raise SystemExit(
            f"{where}: acoustic architecture {arch or '(none)'!r} is not supported "
            f"by the C front runtime (want one of {list(SUPPORTED_ACOUSTIC)}). "
            "If this is a decoder checkpoint, it belongs to "
            "tools/export_piperlite_q8.py, not here.")


def _guard_duration_config(config: dict, where: str) -> None:
    arch = str(config.get("architecture") or "duration_conv")
    if arch != "duration_conv":
        raise SystemExit(f"{where}: duration architecture {arch!r} is not supported "
                         "by the C front runtime (want 'duration_conv')")


def read_texts(path: pathlib.Path, lang: str | None = None) -> list[str]:
    """Sentences from a JSON list, a {lang: [...]} map, a JSONL file whose
    objects carry "text", or one sentence per line."""
    raw = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        loaded = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                if "text" not in obj:
                    raise SystemExit(f"{path}: JSONL row without a 'text' field")
                loaded.append(obj["text"])
            else:
                loaded.append(line)
    if isinstance(loaded, dict):
        if lang is None:
            raise SystemExit(f"{path} is a {{lang: [...]}} map; pass a language")
        if lang not in loaded:
            raise SystemExit(f"language {lang!r} not in {sorted(loaded)}")
        loaded = loaded[lang]
    out = [str(s) for s in loaded if str(s).strip()]
    if not out:
        raise SystemExit(f"{path}: no sentences")
    return out


def ids_from_texts(piper_model: pathlib.Path, piper_config: pathlib.Path,
                   texts: list[str]) -> list[np.ndarray]:
    rows = []
    for text in texts:
        ids = golden.golden_ids_from_piper(piper_model, piper_config, text)
        if ids.size:
            rows.append(ids)
    if not rows:
        raise SystemExit("no calibration ids produced from --calib-texts")
    return rows


def ids_from_pack(pack: pathlib.Path, n: int) -> list[np.ndarray]:
    rows_json = json.loads((pack / "rows.json").read_text())
    rows = []
    for i in range(min(n, len(rows_json))):
        ids, _text = golden.golden_ids_from_pack(pack, i)
        rows.append(ids)
    if not rows:
        raise SystemExit(f"{pack}: no rows")
    return rows


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("duration_checkpoint", type=pathlib.Path, nargs="?")
    ap.add_argument("acoustic_checkpoint", type=pathlib.Path, nargs="?")
    ap.add_argument("--package", type=pathlib.Path, default=None,
                    help="roota.raw-fp16.v1 package dir; repacked in memory "
                         "instead of reading checkpoints")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--pack", type=pathlib.Path, default=None,
                    help="eval pack dir with rows.json (calibration ids)")
    ap.add_argument("--piper-model", type=pathlib.Path, default=None)
    ap.add_argument("--piper-config", type=pathlib.Path, default=None)
    ap.add_argument("--lang", type=str, default=None,
                    help="key to pick from a {lang: [...]} --calib-texts map")
    ap.add_argument("--calib-texts", type=pathlib.Path, default=None,
                    help="JSON list of sentences (or newline-separated text) "
                         "phonemized by --piper-model for calibration")
    ap.add_argument("--calib-n", type=int, default=16)
    ap.add_argument("--length-scale", type=float, default=1.0,
                    help="length scale used while calibrating (affects frame "
                         "counts, and so the frame-stage statistics)")
    ap.add_argument("--headroom", type=float, default=1.0)
    ap.add_argument("--calib-mode", choices=("max", "mse"), default="max",
                    help="max (default): clip at the observed absolute maximum. "
                         "mse: the SQNR clip search export_piperlite_q8.py uses. "
                         "The decoder wants mse because its trunk tensors are "
                         "heavy-tailed (clip/rms 7..21) and clipping buys real "
                         "resolution. The FRONT does not: measured on amy, mse "
                         "clips d_in to 0.56x and d_s0 to 0.63x of their true "
                         "range for no latent-correlation gain (0.999956 either "
                         "way) and costs 4 of 157 tokens their exact frame "
                         "count -- because the duration head is exp() then "
                         "round-half-to-even, a hard decision boundary where a "
                         "clipped tail flips a frame instead of degrading.")
    ap.add_argument("--embed", choices=("int8", "f32"), default="int8",
                    help="embedding tables: int8 with one scale per vocabulary "
                         "row (default) or fp32")
    ap.add_argument("--qmax", type=int, default=CALIB_QMAX,
                    help="activation lane maximum the clips are optimised for "
                         "(must match SNT_FRONT_Q8_QMAX in the runtime)")
    args = ap.parse_args()

    if (args.package is None) == (args.duration_checkpoint is None):
        raise SystemExit("provide either both checkpoints or --package, not both/neither")
    if args.package is None and args.acoustic_checkpoint is None:
        raise SystemExit("acoustic_checkpoint is required alongside duration_checkpoint")
    if (args.pack is None) == (args.calib_texts is None):
        raise SystemExit("provide exactly one of --pack or --calib-texts")
    if args.calib_texts is not None and (args.piper_model is None or args.piper_config is None):
        raise SystemExit("--calib-texts requires --piper-model and --piper-config")
    if args.qmax < 127 or args.qmax > 32767:
        raise SystemExit(f"--qmax {args.qmax} outside the sane 127..32767 range")

    torch.manual_seed(0)
    device = torch.device("cpu")

    # ---- models ---------------------------------------------------------
    provenance: dict[str, object]
    if args.package is not None:
        components, manifest = load_from_package(args.package)
        dur_state, dur_config = components["duration"]
        ac_state, ac_config = components["acoustic"]
        _guard_duration_config(dur_config, f"{args.package} duration component")
        _guard_acoustic_config(ac_config, f"{args.package} acoustic component")
        dur_model = duration_mod.DurationStudent(
            vocab_size=int(dur_config["vocab_size"]),
            hidden=int(dur_config["hidden"]),
            depth=int(dur_config["depth"]),
            kernel_size=int(dur_config["kernel_size"]),
            max_tokens=int(dur_config["max_tokens"]))
        dur_model.load_state_dict(dur_state, strict=True)
        ac_model = latent_mod.create_model_from_config(ac_config)
        ac_model.load_state_dict(ac_state, strict=True)
        provenance = {"source": "package", "package": str(args.package),
                      "voice": manifest.get("voice"),
                      "precision": "float16 widened to float32"}
    else:
        dur_ck = torch.load(args.duration_checkpoint, map_location="cpu",
                            weights_only=False)
        _guard_duration_config(dur_ck.get("config") or {}, str(args.duration_checkpoint))
        ac_ck = torch.load(args.acoustic_checkpoint, map_location="cpu",
                           weights_only=False)
        _guard_acoustic_config(ac_ck.get("config") or {}, str(args.acoustic_checkpoint))
        del dur_ck, ac_ck
        dur_model, dur_config = duration_mod.load_model_from_checkpoint(
            args.duration_checkpoint, device)
        ac_model, ac_config = latent_mod.load_model_from_checkpoint(
            args.acoustic_checkpoint, device)
        provenance = {"source": "checkpoints",
                      "duration_checkpoint": str(args.duration_checkpoint),
                      "acoustic_checkpoint": str(args.acoustic_checkpoint)}
    dur_model.eval()
    ac_model.eval()

    _guard_duration_config(dur_config, "duration config")
    ac_base, ac_prefix, adapter, adapter_mode = golden.check_acoustic_supported(
        ac_model, ac_config)
    base_config = ac_config.get("base_config") if isinstance(
        ac_config.get("base_config"), dict) else ac_config

    dims = {
        "d_vocab": int(dur_config["vocab_size"]),
        "d_hidden": int(dur_config["hidden"]),
        "d_depth": int(dur_config["depth"]),
        "d_kernel": int(dur_config["kernel_size"]),
        "d_max_tokens": int(dur_config["max_tokens"]),
        "d_max_duration": int(dur_config.get("max_duration", 80)),
        "a_vocab": int(base_config["vocab_size"]),
        "a_hidden": int(base_config["hidden"]),
        "a_token_depth": int(base_config["token_depth"]),
        "a_depth": int(base_config["depth"]),
        "a_kernel": int(base_config["kernel_size"]),
        "a_out": int(base_config["out_channels"]),
        "adapter_mode": int(adapter_mode),
        "adapter_kernel": int(adapter.kernel_size) if adapter is not None else 0,
        "adapter_rank": int(adapter.rank) if adapter is not None else 0,
    }
    # The adapter's `mode` is a STRING on the module and an INT in every blob
    # and meta.json this project ships. check_acoustic_supported() is the one
    # place that maps between them; nothing downstream ever sees the string.
    adapter_mode_name = str(adapter.mode) if adapter is not None else "none"

    # ---- calibration ids -------------------------------------------------
    if args.pack is not None:
        id_rows = ids_from_pack(args.pack, args.calib_n)
        id_source = {"kind": "pack", "pack": str(args.pack), "rows": len(id_rows)}
    else:
        texts = read_texts(args.calib_texts, args.lang)[: args.calib_n]
        id_rows = ids_from_texts(args.piper_model, args.piper_config, texts)
        id_source = {"kind": "piper", "model": str(args.piper_model),
                     "rows": len(id_rows)}
    if len(id_rows) < 4:
        raise SystemExit(f"need at least 4 calibration rows, got {len(id_rows)}")
    for ids in id_rows:
        top = int(ids.max())
        if top >= dims["d_vocab"] or top >= dims["a_vocab"]:
            raise SystemExit(f"calibration id {top} exceeds vocab "
                             f"(d {dims['d_vocab']}, a {dims['a_vocab']})")

    names = act_names(dims["d_depth"], dims["a_token_depth"], dims["a_depth"])
    raw_amax, samples = calibrate(dur_model, ac_model, ac_base, id_rows,
                                  dims["d_max_duration"], args.length_scale,
                                  dims["a_out"], names)
    if args.calib_mode == "mse":
        amax = {k: plq.mse_optimal_amax(samples[k], v, qmax=args.qmax)
                for k, v in raw_amax.items()}
    else:
        amax = dict(raw_amax)
    clips = {k: (v * args.headroom if v > 0.0 else 1.0) for k, v in amax.items()}
    act_clip = [clips[n] for n in names]

    # ---- weight quantisation, slot order == export_front_golden.py -------
    dur_sd = {k: v.detach().float() for k, v in dur_model.state_dict().items()}
    ac_sd = {k: v.detach().float() for k, v in ac_model.state_dict().items()}

    records: list[tuple[int, int, int, int, int]] = []
    blob = bytearray()
    pool: list[float] = []
    slot_names: list[str] = []
    dequant: dict[str, np.ndarray] = {}

    def add_w(sd: dict, name: str, shape: tuple, label: str) -> None:
        if name not in sd:
            raise SystemExit(f"missing state-dict tensor: {name}")
        t = sd[name]
        if tuple(t.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(t.shape)} != expected {shape}")
        w = t.numpy().astype(np.float32)
        q, scales = quant_per_row(w)
        aux_off = len(pool)
        pool.extend(scales.tolist())
        records.append((KIND_W8, len(blob), int(q.size), aux_off, int(scales.size)))
        blob.extend(np.ascontiguousarray(q).tobytes())
        slot_names.append(name)
        dequant[label] = dequant_per_row(q, scales, shape)

    def add_f(sd: dict, name: str, shape: tuple, label: str) -> None:
        if name not in sd:
            raise SystemExit(f"missing state-dict tensor: {name}")
        t = sd[name]
        t = t.reshape(-1) if shape == (1,) and t.dim() == 0 else t
        if tuple(t.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(t.shape)} != expected {shape}")
        arr = t.numpy().astype(np.float32).reshape(-1)
        records.append((KIND_F32, len(pool), int(arr.size), -1, 0))
        pool.extend(arr.tolist())
        slot_names.append(name)
        dequant[label] = arr

    def add_emb(sd: dict, name: str, shape: tuple, label: str) -> None:
        t = sd[name]
        if tuple(t.shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(t.shape)} != expected {shape}")
        w = t.numpy().astype(np.float32)
        if args.embed == "f32":
            arr = w.reshape(-1)
            records.append((KIND_F32, len(pool), int(arr.size), -1, 0))
            pool.extend(arr.tolist())
            dequant[label] = w
        else:
            q, scales = quant_per_row(w)
            aux_off = len(pool)
            pool.extend(scales.tolist())
            records.append((KIND_EMB8, len(blob), int(q.size), aux_off, int(scales.size)))
            blob.extend(np.ascontiguousarray(q).tobytes())
            dequant[label] = dequant_per_row(q, scales, shape)
        slot_names.append(name)

    def add_blocks(sd: dict, prefix: str, label_prefix: str, count: int,
                   h: int, k: int) -> None:
        for b in range(count):
            add_f(sd, f"{prefix}.{b}.scale", (1,), f"{label_prefix}.{b}.scale")
            add_w(sd, f"{prefix}.{b}.net.0.weight", (h, h, k), f"{label_prefix}.{b}.net.0.weight")
            add_f(sd, f"{prefix}.{b}.net.0.bias", (h,), f"{label_prefix}.{b}.net.0.bias")
            add_w(sd, f"{prefix}.{b}.net.2.weight", (h, h, k), f"{label_prefix}.{b}.net.2.weight")
            add_f(sd, f"{prefix}.{b}.net.2.bias", (h,), f"{label_prefix}.{b}.net.2.bias")

    dh, dk = dims["d_hidden"], dims["d_kernel"]
    add_emb(dur_sd, "embedding.weight", (dims["d_vocab"], dh), "dur.embedding.weight")
    add_w(dur_sd, "input_proj.weight", (dh, dh + 3, 1), "dur.input_proj.weight")
    add_f(dur_sd, "input_proj.bias", (dh,), "dur.input_proj.bias")
    add_blocks(dur_sd, "blocks", "dur.blocks", dims["d_depth"], dh, dk)
    add_w(dur_sd, "output.weight", (1, dh, 1), "dur.output.weight")
    add_f(dur_sd, "output.bias", (1,), "dur.output.bias")
    n_dur_slots = len(records)

    p = ac_prefix
    ah, ak, aC = dims["a_hidden"], dims["a_kernel"], dims["a_out"]
    add_emb(ac_sd, f"{p}embedding.weight", (dims["a_vocab"], ah), "ac.embedding.weight")
    add_w(ac_sd, f"{p}token_input_proj.weight", (ah, ah + 2, 1), "ac.token_input_proj.weight")
    add_f(ac_sd, f"{p}token_input_proj.bias", (ah,), "ac.token_input_proj.bias")
    add_blocks(ac_sd, f"{p}token_blocks", "ac.token_blocks", dims["a_token_depth"], ah, ak)
    add_w(ac_sd, f"{p}frame_input_proj.weight", (ah, ah + 3, 1), "ac.frame_input_proj.weight")
    add_f(ac_sd, f"{p}frame_input_proj.bias", (ah,), "ac.frame_input_proj.bias")
    add_blocks(ac_sd, f"{p}frame_blocks", "ac.frame_blocks", dims["a_depth"], ah, ak)
    add_w(ac_sd, f"{p}output.weight", (aC, ah, 1), "ac.output.weight")
    add_f(ac_sd, f"{p}output.bias", (aC,), "ac.output.bias")
    if adapter_mode in (2, 4):
        add_w(ac_sd, "adapter.depthwise.weight", (aC, 1, dims["adapter_kernel"]),
              "adapter.depthwise.weight")
    if adapter_mode in (3, 4):
        add_w(ac_sd, "adapter.lowrank_down.weight", (dims["adapter_rank"], aC, 1),
              "adapter.lowrank_down.weight")
        add_f(ac_sd, "adapter.lowrank_down.bias", (dims["adapter_rank"],),
              "adapter.lowrank_down.bias")
        add_w(ac_sd, "adapter.lowrank_up.weight", (aC, dims["adapter_rank"], 1),
              "adapter.lowrank_up.weight")
        add_f(ac_sd, "adapter.lowrank_up.bias", (aC,), "adapter.lowrank_up.bias")
    if adapter_mode > 0:
        add_f(ac_sd, "adapter.scale", (aC,), "adapter.scale")
        add_f(ac_sd, "adapter.bias", (aC,), "adapter.bias")

    dur_left = sorted(set(dur_sd) - set(slot_names[:n_dur_slots]))
    ac_left = sorted(set(ac_sd) - set(slot_names[n_dur_slots:]))
    if dur_left or ac_left:
        raise SystemExit("state dict tensors not covered by the export order: "
                         f"duration={dur_left} acoustic={ac_left}")

    # ---- write ----------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "front_weights_q8.bin").write_bytes(bytes(blob))
    header = [MAGIC, VERSION,
              dims["d_vocab"], dims["d_hidden"], dims["d_depth"], dims["d_kernel"],
              dims["d_max_tokens"], dims["d_max_duration"],
              dims["a_vocab"], dims["a_hidden"], dims["a_token_depth"],
              dims["a_depth"], dims["a_kernel"], dims["a_out"],
              dims["adapter_mode"], dims["adapter_kernel"], dims["adapter_rank"],
              len(records), len(act_clip)]
    with (args.out / "front_meta_q8.bin").open("wb") as fh:
        fh.write(struct.pack(f"<{len(header)}i", *header))
        fh.write(struct.pack(f"<{len(act_clip)}f", *act_clip))
        for rec in records:
            fh.write(struct.pack("<5i", *rec))
        fh.write(struct.pack("<i", len(pool)))
        fh.write(np.asarray(pool, dtype="<f4").tobytes())

    # ---- self-check against the fp32 model on the golden ids -------------
    check: dict[str, object] = {}
    ids_path = args.out / "ids.bin"
    if ids_path.is_file():
        ids = np.frombuffer(ids_path.read_bytes(), dtype="<i4").astype(np.int64)
        gold_dur_path = args.out / "durations.bin"
        # export_front_golden.py writes durations.bin at length_scale 1.0,
        # so the self-check has to ask for 1.0 regardless of what the
        # calibration pass ran at.
        q_dur = simulate_durations_q8(dequant, dims, clips, ids, 1.0, args.qmax)
        if gold_dur_path.is_file():
            gold_dur = np.frombuffer(gold_dur_path.read_bytes(), dtype="<i4").astype(np.int64)
            check["duration_exact"] = int((q_dur == gold_dur).sum())
            check["duration_tokens"] = int(gold_dur.size)
            check["duration_frames_fp32"] = int(gold_dur.sum())
            check["duration_frames_int8"] = int(q_dur.sum())
            use_dur = gold_dur
        else:
            use_dur = q_dur
        lat = simulate_front_q8(dequant, dims, clips, ids, use_dur, args.qmax)
        gold_lat_path = args.out / "latent.bin"
        if gold_lat_path.is_file():
            gold = np.frombuffer(gold_lat_path.read_bytes(), dtype="<f4")
            if gold.size == lat.size:
                a = lat.reshape(-1).astype(np.float64)
                b = gold.astype(np.float64)
                corr = float(np.corrcoef(a, b)[0, 1])
                check["latent_corr_numpy_sim"] = corr
                check["latent_max_abs_diff"] = float(np.abs(a - b).max())
            else:
                check["latent_corr_numpy_sim"] = None
                check["note"] = (f"latent.bin has {gold.size} floats, simulation "
                                 f"produced {lat.size}; durations differ")

    meta_bytes = (args.out / "front_meta_q8.bin").stat().st_size
    summary = {
        "out": str(args.out),
        "provenance": provenance,
        "id_source": id_source,
        "dims": dims,
        "adapter_mode_name": adapter_mode_name,
        "embed": args.embed,
        "qmax": args.qmax,
        "calib_mode": args.calib_mode,
        "headroom": args.headroom,
        "length_scale": args.length_scale,
        "n_tensors": len(records),
        "n_act": len(act_clip),
        "weights_q8_bytes": len(blob),
        "meta_q8_bytes": meta_bytes,
        "total_q8_bytes": len(blob) + meta_bytes,
        "pool_floats": len(pool),
        "self_check": check,
    }
    (args.out / "front_calib_q8.json").write_text(json.dumps(
        dict(summary, act_clip={n: clips[n] for n in names},
             act_raw_amax={n: raw_amax[n] for n in names}), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
