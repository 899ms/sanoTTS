"""Build a piperlite calibration pack from TEXT, using the voice's own front.

tools/export_piperlite_golden.py and tools/export_piperlite_q8.py both want a
pack: rows.json plus per-row npz files carrying `generator_input`, the latent
the decoder consumes. Those packs are a by-product of training, and a user who
holds only a shipped package does not have one. tools/make_pack_from_text.py
solves the same problem for the NANO lineage, but it is Kokoro-specific --
misaki G2P into a 62-entry corpus vocabulary, an 80-bin log-mel
`generator_input` -- and none of that is what a piperlite decoder eats.

So this is the piperlite equivalent: phonemize with the voice's teacher, run
the voice's OWN duration + acoustic students, and write the latent they
produce.

That is also the better calibration set, not merely the available one. A
training pack's `generator_input` is the TEACHER's latent; the decoder on the
device is fed the STUDENT's. Calibrating activation clips on the teacher's
distribution and then running the student's is a mismatch nobody measures
until the voice sounds wrong.

  python tools/make_front_latent_pack.py \\
      --duration DUR.pt --acoustic AC.pt \\
      --piper-model models/teachers/en_US-amy-medium/en_US-amy-medium.onnx \\
      --texts flores24.json --lang en --rows 16 --length-scale 1.08 \\
      --out artifacts/packs/amy-calib

Either --duration/--acoustic checkpoints or --package (repacked in memory).
"""

import argparse
import importlib.util
import json
import pathlib
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


fq8 = _load_module("front_q8_export", REPO / "tools/export_front_q8.py")
golden = fq8.golden
duration_mod = fq8.duration_mod
latent_mod = fq8.latent_mod


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=pathlib.Path, default=None)
    ap.add_argument("--acoustic", type=pathlib.Path, default=None)
    ap.add_argument("--package", type=pathlib.Path, default=None)
    ap.add_argument("--piper-model", type=pathlib.Path, required=True)
    ap.add_argument("--piper-config", type=pathlib.Path, default=None)
    ap.add_argument("--texts", type=pathlib.Path, required=True,
                    help="JSON list, JSON {lang: [..]} with --lang, or one "
                         "sentence per line")
    ap.add_argument("--lang", type=str, default=None)
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    if (args.package is None) == (args.duration is None):
        raise SystemExit("provide either --duration/--acoustic or --package")
    config_path = args.piper_config or pathlib.Path(str(args.piper_model) + ".json")
    if not config_path.is_file():
        raise SystemExit(f"missing piper config {config_path}")

    if args.package is not None:
        components, _manifest = fq8.load_from_package(args.package)
        dur_state, dur_config = components["duration"]
        ac_state, ac_config = components["acoustic"]
        dur_model = duration_mod.DurationStudent(
            vocab_size=int(dur_config["vocab_size"]),
            hidden=int(dur_config["hidden"]),
            depth=int(dur_config["depth"]),
            kernel_size=int(dur_config["kernel_size"]),
            max_tokens=int(dur_config["max_tokens"]))
        dur_model.load_state_dict(dur_state, strict=True)
        ac_model = latent_mod.create_model_from_config(ac_config)
        ac_model.load_state_dict(ac_state, strict=True)
    else:
        if args.acoustic is None:
            raise SystemExit("--acoustic is required alongside --duration")
        dur_model, dur_config = duration_mod.load_model_from_checkpoint(
            args.duration, torch.device("cpu"))
        ac_model, ac_config = latent_mod.load_model_from_checkpoint(
            args.acoustic, torch.device("cpu"))
    dur_model.eval()
    ac_model.eval()
    golden.check_acoustic_supported(ac_model, ac_config)
    base_config = ac_config.get("base_config") if isinstance(
        ac_config.get("base_config"), dict) else ac_config
    a_out = int(base_config["out_channels"])
    max_duration = int(dur_config.get("max_duration", 80))

    texts = fq8.read_texts(args.texts, args.lang)[: args.rows]
    if len(texts) < 4:
        raise SystemExit(f"need at least 4 sentences, got {len(texts)}")

    tensors_dir = args.out / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids_np = golden.golden_ids_from_piper(args.piper_model, config_path, text)
            ids_t = torch.as_tensor(ids_np[None, :], dtype=torch.long)
            mask_t = torch.ones_like(ids_t, dtype=torch.bool)
            durs = duration_mod.predict_durations(
                dur_model, ids_t, mask_t, max_duration=max_duration,
                length_scale=args.length_scale).squeeze(0).cpu().numpy().astype(np.int64)
            frames = int(durs.sum())
            sample = latent_mod.ChunkSample(
                row_id=f"calib{i:03d}", row_index=i, text=text, chunk_index=0,
                phoneme_ids=ids_np.astype(np.int64), durations=durs,
                target=np.zeros((frames, a_out), dtype=np.float32),
                tensor_path=pathlib.Path("calib"), audio_samples=frames * 256)
            features = latent_mod.expand_features(sample, torch.device("cpu"))
            latent = latent_mod.predict_latent_tensor(ac_model, features)  # [T, C]
            if tuple(latent.shape) != (frames, a_out):
                raise SystemExit(f"latent {tuple(latent.shape)} != ({frames}, {a_out})")
            npz_path = tensors_dir / f"calib{i:03d}.npz"
            np.savez(npz_path,
                     generator_input=latent.transpose(0, 1).contiguous().numpy()
                         .astype(np.float32),        # [C, T], decoder layout
                     phoneme_ids=ids_np.astype(np.int64),
                     durations=durs,
                     w_ceil=durs.astype(np.int64))
            try:
                rel = str(npz_path.resolve().relative_to(REPO))
            except ValueError:
                rel = str(npz_path.resolve())
            rows.append({"row_id": f"calib{i:03d}", "index": i, "text": text,
                         "chunks": [{"tensor_npz": rel,
                                     "phoneme_ids": ids_np.astype(int).tolist()}],
                         "frames": frames, "piper_id_count": int(ids_np.size)})
    (args.out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps({"out": str(args.out), "rows": len(rows), "a_out": a_out,
                      "length_scale": args.length_scale,
                      "frames": [r["frames"] for r in rows]}, indent=1))


if __name__ == "__main__":
    main()
