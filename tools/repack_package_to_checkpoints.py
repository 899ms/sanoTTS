"""Rebuild loadable .pt checkpoints from a roota.raw-fp16.v1 package.

Two of the ten languages added on 2026-09-08 -- fr_FR and ar_JO -- exist on k2
only as their shipped package (manifest.json + weights.fp16.bin); the training
checkpoints the manifest names are gone. The package is a lossless dump of the
three state_dicts (see tools/export_roota_self_contained_package.py: every
tensor is written with its state_dict name, shape and a sha256, and every
component's training config is copied into the manifest verbatim), so the
checkpoint tools/export_voice_bundle.py wants can be reconstructed from it.

The one thing that does NOT survive is precision: the package stores float16,
so the rebuilt checkpoint is the fp16 weights widened back to float32, not the
original float32. That is the same artifact every CER number in
experiments/evidence/{ten-languages-asr,ood-tatoeba}-20260908.json was measured
on -- those were all rendered from the packages -- so the browser bundle built
from it is the model that was measured, not an approximation of it.

Every tensor is checked against the sha256 the manifest records before it is
used, so a truncated or mismatched blob fails here rather than turning into a
voice that sounds plausible and is wrong.

  python tools/repack_package_to_checkpoints.py artifacts/voices/ar_JO/package
      -> artifacts/voices/ar_JO/from-package/{duration,latent,decoder}-student.pt
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

# state_dict name -> file the exporter expects, mirroring the layout of the
# languages that do still have their training checkpoints.
COMPONENT_FILES = {
    "duration": "duration-student.pt",
    "acoustic": "latent-student.pt",
    "decoder": "decoder-student.pt",
}
NUMPY_DTYPES = {"float16": "<f2", "float32": "<f4", "int64": "<i8", "int32": "<i4"}


def load_component(blob: bytes, component: dict, name: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for entry in component["tensors"]:
        off = int(entry["offset_bytes"])
        n = int(entry["nbytes"])
        raw = blob[off:off + n]
        if len(raw) != n:
            raise SystemExit(
                f"{name}/{entry['name']}: wanted {n} bytes at {off}, blob has {len(raw)}"
            )
        got = hashlib.sha256(raw).hexdigest()
        if got != entry["sha256"]:
            raise SystemExit(
                f"{name}/{entry['name']}: sha256 {got} != manifest {entry['sha256']}"
            )
        dtype = NUMPY_DTYPES.get(entry["dtype"])
        if dtype is None:
            raise SystemExit(f"{name}/{entry['name']}: unsupported dtype {entry['dtype']!r}")
        arr = np.frombuffer(raw, dtype=dtype).reshape(tuple(entry["shape"]))
        if entry["dtype"] == "float16":
            arr = arr.astype("<f4")            # widen; the model runs in float32
        state[entry["name"]] = torch.from_numpy(np.ascontiguousarray(arr))
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", type=pathlib.Path, help="directory holding manifest.json")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output directory (default: <package>/../from-package)")
    args = ap.parse_args()

    manifest = json.loads((args.package / "manifest.json").read_text())
    if manifest.get("format") != "roota.raw-fp16.v1":
        raise SystemExit(f"unsupported package format {manifest.get('format')!r}")
    blob_path = args.package / manifest["weights_file"]
    blob = blob_path.read_bytes()
    if len(blob) != int(manifest["weights_size_bytes"]):
        raise SystemExit(
            f"{blob_path}: {len(blob)} bytes, manifest says {manifest['weights_size_bytes']}"
        )
    got = hashlib.sha256(blob).hexdigest()
    if got != manifest["weights_sha256"]:
        raise SystemExit(f"{blob_path}: sha256 {got} != manifest {manifest['weights_sha256']}")

    out_dir = args.out or (args.package.parent / "from-package")
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for name, filename in COMPONENT_FILES.items():
        component = manifest["components"][name]
        state = load_component(blob, component, name)
        params = sum(t.numel() for t in state.values())
        if params != int(component["parameters"]):
            raise SystemExit(
                f"{name}: rebuilt {params} parameters, manifest says {component['parameters']}"
            )
        path = out_dir / filename
        torch.save({"model_state_dict": state, "config": component["config"],
                    "rebuilt_from": {"package": str(args.package),
                                     "weights_sha256": manifest["weights_sha256"],
                                     "precision": "float16 widened to float32"}}, path)
        written.append({"component": name, "path": str(path), "tensors": len(state),
                        "parameters": params})

    print(json.dumps({"package": str(args.package), "voice": manifest["voice"],
                      "out_dir": str(out_dir), "components": written}, indent=1))


if __name__ == "__main__":
    main()
