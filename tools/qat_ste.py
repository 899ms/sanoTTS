"""Minimal stand-in for the author's tools/qat_ste.py (quantization-aware training).

The upstream repo publishes train_roota_piper_latent_student.py which does
`import qat_ste` and calls `qat_ste.enable_dense_conv1d_qat(model)`, but that
file is NOT committed to GitHub (verified: not tracked in git, not present in
tools/). This stub lets the latent trainer import and run.

NOTE: this disables quantization-awareness (it is a no-op). For a production
voice you should obtain the real tools/qat_ste.py from the author and replace
this file; the final int8 export/golden-gate may depend on real STE behavior.
"""

from __future__ import annotations

import torch  # noqa: F401  (kept so the module feels like the original)


def enable_dense_conv1d_qat(model):
    """No-op wrapper: returns the model unchanged (no QAT).

    Upstream returns the model wrapped with per-channel int8 straight-through
    estimation. For measuring throughput / a first training run a no-op is
    sufficient; quality runs should use the real implementation.
    """
    return model
