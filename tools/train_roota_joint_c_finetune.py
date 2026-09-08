"""Shared helpers for the joint z finetune (reconstructed joint_common).

The published repo ships train_roota_joint_z_finetune.py which does
`import train_roota_joint_c_finetune as joint_common`, but that module is NOT in
GitHub (verified). This reconstruction provides the names z_finetune needs by
delegating to the two trainer modules that ARE present and complete:

    train_roota_piper_decoder_student  (DecoderStudent, MultiPeriodDiscriminator,
                                        adversarial loss helpers, ChunkSample)
    train_roota_piper_latent_student   (create_model_from_config, expand_features,
                                        predict_latent_tensor, load checkpoint)

The two adversarial runners rebuild the exact inline logic from the decoder
trainer's training loop (target_energy_gate + LSGAN losses + gate/quantile).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn  # noqa: F401  (kept for parity with the upstream module)

import train_roota_piper_decoder_student as decoder_trainer
import train_roota_piper_latent_student as latent_trainer


# --------------------------------------------------------------------------
# trivial utils
# --------------------------------------------------------------------------
def require_dir(path: Path | str, label: str) -> None:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{label} not found: {p}")


def require_file(path: Path | str, label: str) -> None:
    decoder_trainer.require_file(path, label)


def pick_device(requested: str) -> torch.device:
    return decoder_trainer.pick_device(requested)


def finite_or_raise(value: torch.Tensor | float, label: str, step: int) -> None:
    v = float(value.detach().cpu()) if hasattr(value, "detach") else float(value)
    if not math.isfinite(v):
        raise RuntimeError(f"non-finite {label} at step {step}")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], Path):
            out[k] = [str(x) for x in v]
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------
# model building
# --------------------------------------------------------------------------
_DECODER_PARAMS = set(
    inspect.signature(decoder_trainer.DecoderStudent.__init__).parameters
) - {"self"}


def decoder_from_config(config: dict[str, Any]) -> decoder_trainer.DecoderStudent:
    """Build a DecoderStudent from a checkpoint 'config' dict (list->tuple)."""
    kwargs: dict[str, Any] = {}
    for k, v in config.items():
        if k not in _DECODER_PARAMS:
            continue
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return decoder_trainer.DecoderStudent(**kwargs)


def load_acoustic_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    return latent_trainer.load_model_from_checkpoint(checkpoint_path, device)


# --------------------------------------------------------------------------
# acoustic shim: bridge a decoder ChunkSample to latent_trainer.expand_features
# --------------------------------------------------------------------------
class _AcousticShim:
    def __init__(self, sample: decoder_trainer.ChunkSample, frames: int) -> None:
        self.row_id = sample.row_id
        self.chunk_index = sample.chunk_index
        self.phoneme_ids = sample.phoneme_ids
        self.durations = sample.durations
        # expand_features only reads target.shape[0] (length contract); values unused.
        self.target = np.zeros((frames,), dtype=np.float32)
        self.latent = sample.latent


def make_acoustic_shim(sample: decoder_trainer.ChunkSample, code_dim: int) -> _AcousticShim:
    frames = int(sample.latent.shape[-1])
    return _AcousticShim(sample, frames)


# --------------------------------------------------------------------------
# random crops
# --------------------------------------------------------------------------
@dataclass
class _Crop:
    sample: decoder_trainer.ChunkSample
    start: int
    end: int


def select_crops(
    samples: list[decoder_trainer.ChunkSample],
    *,
    batch_size: int,
    crop_frames: int,
) -> list[_Crop]:
    # The joint finetune stacks crops of a fixed width, so only sample chunks
    # that are at least crop_frames long (matching the acoustic model's latent
    # frame count) are eligible; short chunks would yield a ragged crop.
    eligible = [s for s in samples if int(s.latent.shape[-1]) >= crop_frames]
    if not eligible:
        raise RuntimeError(f"no samples with at least {crop_frames} frames")
    crops: list[_Crop] = []
    for _ in range(batch_size):
        sample = eligible[random.randrange(len(eligible))]
        nframes = int(sample.latent.shape[-1])
        max_start = int(nframes) - int(crop_frames)
        start = random.randint(0, max_start)
        crops.append(_Crop(sample=sample, start=start, end=start + crop_frames))
    return crops


# --------------------------------------------------------------------------
# adversarial runners (reconstructed from the decoder trainer inline loop)
# --------------------------------------------------------------------------
def run_waveform_adversarial(
    *,
    args: argparse.Namespace,
    step: int,
    discriminator: nn.Module | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    gen = prediction.new_tensor(0.0)
    feat = prediction.new_tensor(0.0)
    disc = prediction.new_tensor(0.0)
    gate_mean = 1.0
    grad_norm = 0.0
    if discriminator is not None and discriminator_optimizer is not None and step >= int(args.adv_start_step):
        disc_pred = prediction
        disc_target = target.detach()
        if getattr(args, "adv_gate_mode", "target-energy") == "target-energy":
            gate = decoder_trainer.target_energy_gate(
                target,
                quantile=float(args.adv_gate_quantile),
                sharpness=float(args.adv_gate_sharpness),
                frame_size=int(args.adv_gate_frame_size),
                frame_hop=int(args.adv_gate_frame_hop),
            )
            gate_mean = float(gate.detach().mean().cpu())
            disc_pred = prediction * gate
            disc_target = disc_target * gate.detach()
        decoder_trainer.set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        real_scores, _real_features = discriminator(disc_target)
        fake_scores, _fake_features = discriminator(disc_pred.detach())
        disc = decoder_trainer.discriminator_lsgan_loss(real_scores, fake_scores)
        if not torch.isfinite(disc):
            raise RuntimeError(f"non-finite discriminator loss at step {step}")
        disc.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0).detach().cpu()
        )
        discriminator_optimizer.step()

        decoder_trainer.set_requires_grad(discriminator, False)
        fake_scores_for_gen, fake_features_for_gen = discriminator(disc_pred)
        _real_scores_for_gen, real_features_for_gen = discriminator(disc_target)
        gen = decoder_trainer.generator_lsgan_loss(fake_scores_for_gen)
        feat = decoder_trainer.discriminator_feature_matching_loss(
            real_features_for_gen,
            fake_features_for_gen,
        )
        decoder_trainer.set_requires_grad(discriminator, True)
        if not torch.isfinite(gen):
            raise RuntimeError(f"non-finite generator adversarial loss at step {step}")
        if not torch.isfinite(feat):
            raise RuntimeError(f"non-finite adversarial feature loss at step {step}")
    return gen, feat, disc, gate_mean, grad_norm


def run_delta_adversarial(
    *,
    args: argparse.Namespace,
    step: int,
    discriminator: nn.Module | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    gen = prediction.new_tensor(0.0)
    feat = prediction.new_tensor(0.0)
    disc = prediction.new_tensor(0.0)
    gate_mean = 1.0
    grad_norm = 0.0
    if discriminator is not None and discriminator_optimizer is not None and step >= int(args.adv_start_step):
        delta_gate = None
        if getattr(args, "adv_gate_mode", "target-energy") == "target-energy":
            delta_gate = decoder_trainer.target_energy_gate(
                target,
                quantile=float(args.adv_gate_quantile),
                sharpness=float(args.adv_gate_sharpness),
                frame_size=int(args.adv_gate_frame_size),
                frame_hop=int(args.adv_gate_frame_hop),
            )
            gate_mean = float(delta_gate.detach().mean().cpu())
        delta_prediction, delta_target = decoder_trainer.first_difference_discriminator_audio(
            prediction, target, delta_gate
        )
        decoder_trainer.set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)
        real_scores, _real_features = discriminator(delta_target)
        fake_scores, _fake_features = discriminator(delta_prediction.detach())
        disc = decoder_trainer.discriminator_lsgan_loss(real_scores, fake_scores)
        if not torch.isfinite(disc):
            raise RuntimeError(f"non-finite delta discriminator loss at step {step}")
        disc.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0).detach().cpu()
        )
        discriminator_optimizer.step()

        decoder_trainer.set_requires_grad(discriminator, False)
        fake_scores_for_gen, fake_features_for_gen = discriminator(delta_prediction)
        _real_scores_for_gen, real_features_for_gen = discriminator(delta_target)
        gen = decoder_trainer.generator_lsgan_loss(fake_scores_for_gen)
        feat = decoder_trainer.discriminator_feature_matching_loss(
            real_features_for_gen,
            fake_features_for_gen,
        )
        decoder_trainer.set_requires_grad(discriminator, True)
        if not torch.isfinite(gen):
            raise RuntimeError(f"non-finite delta generator adversarial loss at step {step}")
        if not torch.isfinite(feat):
            raise RuntimeError(f"non-finite delta adversarial feature loss at step {step}")
    return gen, feat, disc, gate_mean, grad_norm
