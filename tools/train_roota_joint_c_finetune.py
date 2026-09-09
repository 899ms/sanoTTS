#!/usr/bin/env python3
"""Joint fine-tune the c-acoustic student and LRC decoder on waveform loss.

This is the interface repair pass for the compact Kristin stack:

    phonemes + teacher w_ceil -> c-acoustic -> c_hat -> LRC decoder -> waveform

The training-only LRC encoder E(z) supplies the c-anchor target from the
teacher generator_input latent.  The standalone acoustic and decoder checkpoints
written by this tool keep the existing loader contracts unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_roota_piper_decoder_student as decoder_trainer
import train_roota_piper_latent_student as latent_trainer


BASE_ARTIFACT_DIR = ROOT / "artifacts" / "sub10m-search" / "root-a-piper-vits"
DEFAULT_PACK_DIR = BASE_ARTIFACT_DIR / "en_US-kristin-medium-train2048-decoder-piper-native-20260702"
DEFAULT_TEACHER_DECODER = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-medium-decoder-cut-20260702"
    / "en_US-kristin-medium-decoder-from-generator-input.onnx"
)
DEFAULT_ACOUSTIC_CHECKPOINT = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-u600-acoustic-c-stage-lrc-r3-c40-pre6000-refine2000-20260703"
    / "refine2000"
    / "latent-student.pt"
)
DEFAULT_DECODER_CHECKPOINT = (
    BASE_ARTIFACT_DIR
    / "en_US-kristin-u600-lrc-r4-cmix-finetune-30k-3050-20260703"
    / "step-30000"
    / "decoder-student.pt"
)
DEFAULT_OUT_DIR = BASE_ARTIFACT_DIR / "en_US-kristin-u600-joint-c-finetune-smoke-20260703"
HOP_LENGTH = decoder_trainer.HOP_LENGTH


@dataclass(frozen=True)
class Crop:
    sample: decoder_trainer.ChunkSample
    start: int
    end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--teacher-decoder", type=Path, default=DEFAULT_TEACHER_DECODER)
    parser.add_argument("--acoustic-checkpoint", type=Path, default=DEFAULT_ACOUSTIC_CHECKPOINT)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_DECODER_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=64)
    parser.add_argument("--acoustic-lr", type=float, default=2e-5)
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--encoder-lr", type=float, default=5e-6)
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the training-only LRC encoder E(z). Use --no-freeze-encoder for low-LR E updates.",
    )
    parser.add_argument("--waveform-l1-weight", type=float, default=0.1)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--c-anchor-weight", type=float, default=0.5)
    parser.add_argument("--adv-weight", type=float, default=0.0)
    parser.add_argument("--adv-feature-weight", type=float, default=0.0)
    parser.add_argument("--adv-delta-weight", type=float, default=0.025)
    parser.add_argument("--adv-delta-feature-weight", type=float, default=0.25)
    parser.add_argument("--adv-start-step", type=int, default=1)
    parser.add_argument("--adv-lr", type=float, default=2e-4)
    parser.add_argument("--adv-periods", type=str, default="2,3,5,7,11")
    parser.add_argument("--adv-channels", type=str, default="8,16,32,64")
    parser.add_argument(
        "--adv-gate-mode",
        choices=("none", "target-energy"),
        default="none",
        help="Optionally gate discriminator audio with a teacher-energy mask before adversarial losses.",
    )
    parser.add_argument("--adv-gate-quantile", type=float, default=0.40)
    parser.add_argument("--adv-gate-sharpness", type=float, default=24.0)
    parser.add_argument("--adv-gate-frame-size", type=int, default=1024)
    parser.add_argument("--adv-gate-frame-hop", type=int, default=256)
    parser.add_argument("--duration-params", type=int, default=36164)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--verify-standalone-reload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After saving, prove standalone checkpoints reload through the original loaders.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def validate_args(args: argparse.Namespace) -> None:
    require_dir(args.pack_dir, "pack directory")
    require_file(args.teacher_decoder, "teacher decoder ONNX")
    require_file(args.acoustic_checkpoint, "c-acoustic checkpoint")
    require_file(args.decoder_checkpoint, "LRC decoder checkpoint")
    if int(args.steps) < 1:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if int(args.batch_size) < 1:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    if int(args.crop_frames) < 1:
        raise ValueError(f"--crop-frames must be positive, got {args.crop_frames}")
    for name in ("acoustic_lr", "decoder_lr", "encoder_lr", "adv_lr"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive, got {value!r}")
    for name in (
        "waveform_l1_weight",
        "spectral_weight",
        "c_anchor_weight",
        "adv_weight",
        "adv_feature_weight",
        "adv_delta_weight",
        "adv_delta_feature_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative, got {value!r}")
    if float(args.waveform_l1_weight) == 0.0 and float(args.spectral_weight) == 0.0:
        raise ValueError("at least one of --waveform-l1-weight or --spectral-weight must be positive")
    if float(args.c_anchor_weight) == 0.0:
        raise ValueError("--c-anchor-weight must be positive for the joint interface contract")
    if int(args.adv_start_step) < 1:
        raise ValueError(f"--adv-start-step must be >= 1, got {args.adv_start_step}")
    if not (0.0 < float(args.adv_gate_quantile) < 1.0):
        raise ValueError(f"--adv-gate-quantile must be in (0, 1), got {args.adv_gate_quantile}")
    if float(args.adv_gate_sharpness) <= 0.0:
        raise ValueError(f"--adv-gate-sharpness must be positive, got {args.adv_gate_sharpness}")
    if int(args.adv_gate_frame_size) <= 1:
        raise ValueError(f"--adv-gate-frame-size must be greater than 1, got {args.adv_gate_frame_size}")
    if int(args.adv_gate_frame_hop) <= 0:
        raise ValueError(f"--adv-gate-frame-hop must be positive, got {args.adv_gate_frame_hop}")
    if int(args.duration_params) < 0:
        raise ValueError(f"--duration-params must be non-negative, got {args.duration_params}")


def pick_device(requested: str) -> torch.device:
    return decoder_trainer.pick_device(requested)


def state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, Path) else value
    return result


def load_acoustic_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = decoder_trainer.load_torch_checkpoint(checkpoint_path, "c-acoustic checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing acoustic checkpoint config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint_path}: missing acoustic model_state_dict")
    model = latent_trainer.create_model_from_config(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model, dict(config)


def decoder_from_config(config: dict[str, Any]) -> decoder_trainer.DecoderStudent:
    raw_channels = config.get("channels")
    if not isinstance(raw_channels, (list, tuple)):
        raise RuntimeError(f"decoder config missing channels list: {config!r}")
    return decoder_trainer.DecoderStudent(
        in_channels=int(config.get("in_channels") or 0),
        channels=tuple(int(value) for value in raw_channels),
        res_layers=int(config.get("res_layers", 1)),
        variant=str(config.get("variant") or "dense"),
        rank_ratio=float(config.get("rank_ratio", 0.5)),
        activation=str(config.get("activation") or "leaky_relu"),
        stage_affine=bool(config.get("stage_affine", False)),
        factorized_pre_rank=int(config.get("factorized_pre_rank", 0)),
        piper_res_factor_rank_ratio=float(config.get("piper_res_factor_rank_ratio", 0.0)),
        res_bank_scale_mode=str(config.get("res_bank_scale_mode", "kept")),
        stage0_branches=tuple(int(value) for value in config.get("stage0_branches", [0, 1, 2])),
        stage1_branches=tuple(int(value) for value in config.get("stage1_branches", [0, 1, 2])),
        stage2_branches=tuple(int(value) for value in config.get("stage2_branches", [0, 1, 2])),
        stage3_branches=tuple(int(value) for value in config.get("stage3_branches", [0, 1, 2])),
        post_filter_channels=int(config.get("post_filter_channels", 0)),
        post_filter_layers=int(config.get("post_filter_layers", 0)),
        post_filter_kernel=int(config.get("post_filter_kernel", 9)),
        post_filter_scale=float(config.get("post_filter_scale", 0.25)),
        pre_tanh_repair_channels=int(config.get("pre_tanh_repair_channels", 0)),
        pre_tanh_repair_layers=int(config.get("pre_tanh_repair_layers", 0)),
        pre_tanh_repair_kernel=int(config.get("pre_tanh_repair_kernel", 7)),
        pre_tanh_repair_scale=float(config.get("pre_tanh_repair_scale", 0.15)),
        istft_n_fft=int(config.get("istft_n_fft", 512)),
        fsd_dim=int(config.get("fsd_dim", 72)),
        fsd_blocks=int(config.get("fsd_blocks", 5)),
        fsd_film_rank=int(config.get("fsd_film_rank", 12)),
        fsd_head_rank=int(config.get("fsd_head_rank", 48)),
        wavehax_channels=int(config.get("wavehax_channels", 32)),
        wavehax_blocks=int(config.get("wavehax_blocks", 8)),
        wavehax_mult_channels=int(config.get("wavehax_mult_channels", 2)),
        wavehax_kernel_freq=int(config.get("wavehax_kernel_freq", 7)),
        wavehax_kernel_time=int(config.get("wavehax_kernel_time", 7)),
        wavehax_f0_channel=int(config.get("wavehax_f0_channel", 80)),
        wavehax_voiced_channel=int(config.get("wavehax_voiced_channel", 81)),
        wavehax_sample_rate=int(config.get("wavehax_sample_rate", 24000)),
        wavehax_prior_power=float(config.get("wavehax_prior_power", 0.1)),
        wavehax_prior_noise=float(config.get("wavehax_prior_noise", 0.01)),
        stage_projection_bottlenecks=tuple(int(value) for value in config.get("stage_projection_bottlenecks", [])),
    )


def load_decoder_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[decoder_trainer.DecoderStudent, decoder_trainer.LrcEncoder, dict[str, Any]]:
    checkpoint = decoder_trainer.load_torch_checkpoint(checkpoint_path, "LRC decoder checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing decoder checkpoint config")
    if str(config.get("variant") or "") != "lrc":
        raise RuntimeError(f"{checkpoint_path}: joint c fine-tune requires decoder variant lrc, got {config.get('variant')!r}")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint_path}: missing decoder model_state_dict")
    encoder_state = checkpoint.get("lrc_encoder_state_dict")
    if not isinstance(encoder_state, dict):
        raise RuntimeError(f"{checkpoint_path}: missing lrc_encoder_state_dict")
    model = decoder_from_config(config)
    model.load_state_dict(state, strict=True)
    model.to(device)
    source_in_channels = int(config.get("source_in_channels") or 0)
    code_dim = int(config.get("lrc_code_dim") or config.get("in_channels") or 0)
    encoder_hidden = int(config.get("lrc_encoder_hidden") or 0)
    if source_in_channels <= 0 or code_dim <= 0 or encoder_hidden <= 0:
        raise RuntimeError(f"{checkpoint_path}: invalid LRC encoder dimensions in config: {config!r}")
    lrc_encoder = decoder_trainer.LrcEncoder(
        in_channels=source_in_channels,
        hidden=encoder_hidden,
        code_dim=code_dim,
    )
    lrc_encoder.load_state_dict(encoder_state, strict=True)
    lrc_encoder.to(device)
    return model, lrc_encoder, dict(config)


def make_acoustic_shim(
    sample: decoder_trainer.ChunkSample,
    *,
    code_dim: int,
) -> latent_trainer.ChunkSample:
    frames = int(sample.latent.shape[2])
    return latent_trainer.ChunkSample(
        row_id=sample.row_id,
        row_index=sample.row_index,
        text=sample.text,
        chunk_index=sample.chunk_index,
        phoneme_ids=sample.phoneme_ids,
        durations=sample.durations,
        target=np.zeros((frames, int(code_dim)), dtype=np.float32),
        tensor_path=sample.tensor_path,
        audio_samples=int(sample.teacher_audio.size),
    )


def select_crops(samples: list[decoder_trainer.ChunkSample], batch_size: int, crop_frames: int) -> list[Crop]:
    eligible = [sample for sample in samples if int(sample.latent.shape[2]) >= int(crop_frames)]
    if not eligible:
        raise RuntimeError(f"no samples have at least {crop_frames} frames")
    crops: list[Crop] = []
    for _ in range(int(batch_size)):
        sample = random.choice(eligible)
        frames = int(sample.latent.shape[2])
        start = random.randint(0, frames - int(crop_frames))
        crops.append(Crop(sample=sample, start=start, end=start + int(crop_frames)))
    return crops


def joint_crop_batch(
    samples: list[decoder_trainer.ChunkSample],
    *,
    acoustic_model: nn.Module,
    lrc_encoder: decoder_trainer.LrcEncoder,
    code_dim: int,
    batch_size: int,
    crop_frames: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    crops = select_crops(samples, batch_size=batch_size, crop_frames=crop_frames)
    c_hat_values: list[torch.Tensor] = []
    latent_values: list[np.ndarray] = []
    audio_values: list[np.ndarray] = []
    for crop in crops:
        sample = crop.sample
        shim = make_acoustic_shim(sample, code_dim=int(code_dim))
        features = latent_trainer.expand_features(shim, device)
        c_hat_full = latent_trainer.predict_latent_tensor(acoustic_model, features)
        if c_hat_full.ndim != 2:
            raise RuntimeError(f"c-acoustic prediction must be [T, C], got {c_hat_full.shape}")
        if int(c_hat_full.shape[0]) != int(sample.latent.shape[2]) or int(c_hat_full.shape[1]) != int(code_dim):
            raise RuntimeError(
                f"{sample.row_id} chunk {sample.chunk_index}: c-acoustic prediction shape "
                f"{tuple(c_hat_full.shape)} != ({int(sample.latent.shape[2])}, {int(code_dim)})"
            )
        c_hat_values.append(c_hat_full[crop.start : crop.end, :].transpose(0, 1))
        latent_values.append(sample.latent[:, :, crop.start : crop.end])
        audio_start = crop.start * HOP_LENGTH
        audio_end = crop.end * HOP_LENGTH
        audio_values.append(sample.teacher_audio[audio_start:audio_end].reshape(1, -1))
    c_hat = torch.stack(c_hat_values, dim=0).to(dtype=torch.float32)
    teacher_latent = torch.as_tensor(np.concatenate(latent_values, axis=0), dtype=torch.float32, device=device)
    target = torch.as_tensor(np.stack(audio_values, axis=0), dtype=torch.float32, device=device)
    exact_c = lrc_encoder(teacher_latent)
    if c_hat.shape != exact_c.shape:
        raise RuntimeError(f"c_hat shape {c_hat.shape} != E(z) shape {exact_c.shape}")
    return c_hat, exact_c, teacher_latent, target


def finite_or_raise(value: torch.Tensor, label: str, step: int) -> None:
    if not torch.isfinite(value):
        raise RuntimeError(f"non-finite {label} at step {step}")


def maybe_target_energy_gate(args: argparse.Namespace, target: torch.Tensor) -> torch.Tensor | None:
    if args.adv_gate_mode != "target-energy":
        return None
    return decoder_trainer.target_energy_gate(
        target,
        quantile=float(args.adv_gate_quantile),
        sharpness=float(args.adv_gate_sharpness),
        frame_size=int(args.adv_gate_frame_size),
        frame_hop=int(args.adv_gate_frame_hop),
    )


def run_waveform_adversarial(
    *,
    args: argparse.Namespace,
    step: int,
    discriminator: decoder_trainer.MultiPeriodDiscriminator | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    zero = prediction.new_tensor(0.0)
    if discriminator is None or discriminator_optimizer is None or step < int(args.adv_start_step):
        return zero, zero, zero, 1.0, 0.0
    gate = maybe_target_energy_gate(args, target)
    discriminator_prediction = prediction if gate is None else prediction * gate
    discriminator_target = target.detach() if gate is None else target.detach() * gate.detach()
    gate_mean = 1.0 if gate is None else float(gate.detach().mean().cpu())

    decoder_trainer.set_requires_grad(discriminator, True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    real_scores, _real_features = discriminator(discriminator_target)
    fake_scores, _fake_features = discriminator(discriminator_prediction.detach())
    discriminator_loss = decoder_trainer.discriminator_lsgan_loss(real_scores, fake_scores)
    finite_or_raise(discriminator_loss, "discriminator loss", step)
    discriminator_loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0).detach().cpu())
    discriminator_optimizer.step()

    decoder_trainer.set_requires_grad(discriminator, False)
    fake_scores_for_generator, fake_features_for_generator = discriminator(discriminator_prediction)
    _real_scores_for_generator, real_features_for_generator = discriminator(discriminator_target)
    generator_loss = decoder_trainer.generator_lsgan_loss(fake_scores_for_generator)
    feature_loss = decoder_trainer.discriminator_feature_matching_loss(
        real_features_for_generator,
        fake_features_for_generator,
    )
    decoder_trainer.set_requires_grad(discriminator, True)
    finite_or_raise(generator_loss, "generator adversarial loss", step)
    finite_or_raise(feature_loss, "adversarial feature loss", step)
    return generator_loss, feature_loss, discriminator_loss.detach(), gate_mean, grad_norm


def run_delta_adversarial(
    *,
    args: argparse.Namespace,
    step: int,
    discriminator: decoder_trainer.MultiPeriodDiscriminator | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    zero = prediction.new_tensor(0.0)
    if discriminator is None or discriminator_optimizer is None or step < int(args.adv_start_step):
        return zero, zero, zero, 1.0, 0.0
    gate = maybe_target_energy_gate(args, target)
    gate_mean = 1.0 if gate is None else float(gate.detach().mean().cpu())
    delta_prediction, delta_target = decoder_trainer.first_difference_discriminator_audio(
        prediction,
        target,
        gate,
    )

    decoder_trainer.set_requires_grad(discriminator, True)
    discriminator_optimizer.zero_grad(set_to_none=True)
    real_scores, _real_features = discriminator(delta_target)
    fake_scores, _fake_features = discriminator(delta_prediction.detach())
    discriminator_loss = decoder_trainer.discriminator_lsgan_loss(real_scores, fake_scores)
    finite_or_raise(discriminator_loss, "delta discriminator loss", step)
    discriminator_loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0).detach().cpu())
    discriminator_optimizer.step()

    decoder_trainer.set_requires_grad(discriminator, False)
    fake_scores_for_generator, fake_features_for_generator = discriminator(delta_prediction)
    _real_scores_for_generator, real_features_for_generator = discriminator(delta_target)
    generator_loss = decoder_trainer.generator_lsgan_loss(fake_scores_for_generator)
    feature_loss = decoder_trainer.discriminator_feature_matching_loss(
        real_features_for_generator,
        fake_features_for_generator,
    )
    decoder_trainer.set_requires_grad(discriminator, True)
    finite_or_raise(generator_loss, "delta generator adversarial loss", step)
    finite_or_raise(feature_loss, "delta adversarial feature loss", step)
    return generator_loss, feature_loss, discriminator_loss.detach(), gate_mean, grad_norm


def parameter_accounting(
    *,
    args: argparse.Namespace,
    acoustic_model: nn.Module,
    decoder_model: nn.Module,
    lrc_encoder: nn.Module,
) -> dict[str, Any]:
    acoustic_parameters = latent_trainer.count_parameters(acoustic_model)
    decoder_parameters = decoder_trainer.count_parameters(decoder_model)
    lrc_encoder_parameters = decoder_trainer.count_parameters(lrc_encoder)
    lrc_encoder_trainable = int(sum(param.numel() for param in lrc_encoder.parameters() if param.requires_grad))
    acoustic_trainable = int(sum(param.numel() for param in acoustic_model.parameters() if param.requires_grad))
    decoder_trainable = int(sum(param.numel() for param in decoder_model.parameters() if param.requires_grad))
    duration_parameters = int(args.duration_params)
    return {
        "duration_student_parameters": duration_parameters,
        "acoustic_parameters": int(acoustic_parameters),
        "decoder_parameters": int(decoder_parameters),
        "lrc_encoder_training_only_parameters": int(lrc_encoder_parameters),
        "acoustic_trainable_parameters": int(acoustic_trainable),
        "decoder_trainable_parameters": int(decoder_trainable),
        "lrc_encoder_trainable_parameters": int(lrc_encoder_trainable),
        "freeze_encoder": bool(args.freeze_encoder),
        "inference_total_parameters": int(duration_parameters + acoustic_parameters + decoder_parameters),
        "training_total_parameters": int(duration_parameters + acoustic_parameters + decoder_parameters + lrc_encoder_parameters),
        "trainable_parameters": int(acoustic_trainable + decoder_trainable + lrc_encoder_trainable),
    }


def save_checkpoints(
    *,
    args: argparse.Namespace,
    acoustic_model: nn.Module,
    acoustic_config: dict[str, Any],
    decoder_model: decoder_trainer.DecoderStudent,
    decoder_config: dict[str, Any],
    lrc_encoder: decoder_trainer.LrcEncoder,
    logs: list[dict[str, Any]],
    accounting: dict[str, Any],
) -> dict[str, Path]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    acoustic_state = state_dict_cpu(acoustic_model)
    decoder_state = state_dict_cpu(decoder_model)
    lrc_encoder_state = state_dict_cpu(lrc_encoder)
    train_args = jsonable_args(args)

    joint_checkpoint = args.out_dir / "joint-c-finetune.pt"
    acoustic_checkpoint = args.out_dir / "latent-student.pt"
    decoder_checkpoint = args.out_dir / "decoder-student.pt"

    torch.save(
        {
            "format": "roota_joint_c_finetune_v1",
            "acoustic_state_dict": acoustic_state,
            "decoder_model_state_dict": decoder_state,
            "lrc_encoder_state_dict": lrc_encoder_state,
            "acoustic_config": acoustic_config,
            "decoder_config": decoder_config,
            "joint_config": train_args,
            "parameter_accounting": accounting,
            "logs": logs,
        },
        joint_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": acoustic_state,
            "config": acoustic_config,
            "train_args": train_args,
            "joint_checkpoint": str(joint_checkpoint),
            "joint_logs": logs,
            "student_parameters": int(accounting["acoustic_parameters"]),
        },
        acoustic_checkpoint,
    )
    torch.save(
        {
            "model_state_dict": decoder_state,
            "lrc_encoder_state_dict": lrc_encoder_state,
            "config": decoder_config,
            "train_args": train_args,
            "joint_checkpoint": str(joint_checkpoint),
            "joint_logs": logs,
            "decoder_parameters": int(accounting["decoder_parameters"]),
            "lrc_encoder_training_only_parameters": int(accounting["lrc_encoder_training_only_parameters"]),
            "adversarial_discriminator_parameters": 0,
            "adversarial_delta_discriminator_parameters": 0,
        },
        decoder_checkpoint,
    )
    return {
        "joint_checkpoint": joint_checkpoint,
        "acoustic_checkpoint": acoustic_checkpoint,
        "decoder_checkpoint": decoder_checkpoint,
    }


def verify_standalone_reload(paths: dict[str, Path], device: torch.device) -> dict[str, Any]:
    acoustic_model, acoustic_config = latent_trainer.load_model_from_checkpoint(
        paths["acoustic_checkpoint"],
        device,
    )
    acoustic_parameters = latent_trainer.count_parameters(acoustic_model)
    decoder_checkpoint = decoder_trainer.load_torch_checkpoint(
        paths["decoder_checkpoint"],
        "saved standalone decoder checkpoint",
    )
    decoder_config = decoder_checkpoint.get("config")
    if not isinstance(decoder_config, dict):
        raise RuntimeError(f"{paths['decoder_checkpoint']}: saved standalone decoder missing config")
    decoder_state = decoder_checkpoint.get("model_state_dict")
    if not isinstance(decoder_state, dict):
        raise RuntimeError(f"{paths['decoder_checkpoint']}: saved standalone decoder missing model_state_dict")
    decoder_model = decoder_from_config(decoder_config).to(device)
    incompatible = decoder_model.load_state_dict(decoder_state, strict=True)
    lrc_encoder_state = decoder_checkpoint.get("lrc_encoder_state_dict")
    if not isinstance(lrc_encoder_state, dict):
        raise RuntimeError(f"{paths['decoder_checkpoint']}: saved standalone decoder missing lrc_encoder_state_dict")
    lrc_encoder = decoder_trainer.LrcEncoder(
        in_channels=int(decoder_config.get("source_in_channels") or 0),
        hidden=int(decoder_config.get("lrc_encoder_hidden") or 0),
        code_dim=int(decoder_config.get("lrc_code_dim") or decoder_config.get("in_channels") or 0),
    ).to(device)
    lrc_incompatible = lrc_encoder.load_state_dict(lrc_encoder_state, strict=True)
    return {
        "acoustic_load_model_from_checkpoint": {
            "ok": True,
            "checkpoint": str(paths["acoustic_checkpoint"]),
            "parameters": int(acoustic_parameters),
            "config_out_channels": int(acoustic_config.get("out_channels") or 0),
        },
        "decoder_DecoderStudent_load_state_dict_strict": {
            "ok": True,
            "checkpoint": str(paths["decoder_checkpoint"]),
            "parameters": int(decoder_trainer.count_parameters(decoder_model)),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        },
        "lrc_encoder_load_state_dict_strict": {
            "ok": True,
            "parameters": int(decoder_trainer.count_parameters(lrc_encoder)),
            "missing_keys": list(lrc_incompatible.missing_keys),
            "unexpected_keys": list(lrc_incompatible.unexpected_keys),
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = pick_device(str(args.device))
    rows, samples, source_in_channels = decoder_trainer.load_samples(args.pack_dir, args.teacher_decoder)
    if int(source_in_channels) != 192:
        raise RuntimeError(f"{args.pack_dir}: expected 192-channel generator_input, got {source_in_channels}")

    acoustic_model, acoustic_config = load_acoustic_checkpoint(args.acoustic_checkpoint, device)
    decoder_model, lrc_encoder, decoder_config = load_decoder_checkpoint(args.decoder_checkpoint, device)
    code_dim = int(decoder_config.get("lrc_code_dim") or decoder_config.get("in_channels") or 0)
    if int(acoustic_config.get("out_channels") or 0) != code_dim:
        raise RuntimeError(
            f"{args.acoustic_checkpoint}: acoustic out_channels {acoustic_config.get('out_channels')} "
            f"!= decoder LRC code_dim {code_dim}"
        )
    if int(decoder_config.get("source_in_channels") or 0) != int(source_in_channels):
        raise RuntimeError(
            f"{args.decoder_checkpoint}: source_in_channels {decoder_config.get('source_in_channels')} "
            f"!= pack channels {source_in_channels}"
        )

    if bool(args.freeze_encoder):
        for parameter in lrc_encoder.parameters():
            parameter.requires_grad_(False)
        lrc_encoder.eval()
    else:
        lrc_encoder.train()
    acoustic_model.train()
    decoder_model.train()

    accounting = parameter_accounting(
        args=args,
        acoustic_model=acoustic_model,
        decoder_model=decoder_model,
        lrc_encoder=lrc_encoder,
    )
    print(json.dumps({"joint_parameter_accounting": accounting}, ensure_ascii=False), flush=True)

    optimizer_groups: list[dict[str, Any]] = [
        {
            "params": [param for param in acoustic_model.parameters() if param.requires_grad],
            "lr": float(args.acoustic_lr),
            "name": "acoustic",
        },
        {
            "params": [param for param in decoder_model.parameters() if param.requires_grad],
            "lr": float(args.decoder_lr),
            "name": "decoder",
        },
    ]
    if not bool(args.freeze_encoder):
        optimizer_groups.append(
            {
                "params": [param for param in lrc_encoder.parameters() if param.requires_grad],
                "lr": float(args.encoder_lr),
                "name": "lrc_encoder",
            }
        )
    optimizer_groups = [group for group in optimizer_groups if group["params"]]
    if not optimizer_groups:
        raise RuntimeError("joint model has no trainable parameters")
    trainable_parameters = [param for group in optimizer_groups for param in group["params"]]
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)

    adv_periods = decoder_trainer.parse_positive_int_tuple(str(args.adv_periods), label="--adv-periods", min_value=2)
    adv_channels = decoder_trainer.parse_positive_int_tuple(str(args.adv_channels), label="--adv-channels")
    discriminator: decoder_trainer.MultiPeriodDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    discriminator_parameter_count = 0
    if float(args.adv_weight) > 0.0 or float(args.adv_feature_weight) > 0.0:
        discriminator = decoder_trainer.MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        discriminator_parameter_count = decoder_trainer.count_parameters(discriminator)
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(args.adv_lr), weight_decay=1e-5)
    delta_discriminator: decoder_trainer.MultiPeriodDiscriminator | None = None
    delta_discriminator_optimizer: torch.optim.Optimizer | None = None
    delta_discriminator_parameter_count = 0
    if float(args.adv_delta_weight) > 0.0 or float(args.adv_delta_feature_weight) > 0.0:
        delta_discriminator = decoder_trainer.MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        delta_discriminator_parameter_count = decoder_trainer.count_parameters(delta_discriminator)
        delta_discriminator_optimizer = torch.optim.AdamW(
            delta_discriminator.parameters(),
            lr=float(args.adv_lr),
            weight_decay=1e-5,
        )
    print(
        json.dumps(
            {
                "joint_adversarial_parameters": {
                    "adversarial_discriminator_parameters": int(discriminator_parameter_count),
                    "adversarial_delta_discriminator_parameters": int(delta_discriminator_parameter_count),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    logs: list[dict[str, Any]] = []
    for step in range(1, int(args.steps) + 1):
        c_hat, exact_c, _teacher_latent, target = joint_crop_batch(
            samples,
            acoustic_model=acoustic_model,
            lrc_encoder=lrc_encoder,
            code_dim=code_dim,
            batch_size=int(args.batch_size),
            crop_frames=int(args.crop_frames),
            device=device,
        )
        prediction_value = decoder_model(c_hat)
        if isinstance(prediction_value, tuple):
            raise RuntimeError("joint decoder returned features unexpectedly")
        prediction = prediction_value
        if prediction.shape != target.shape:
            raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")

        waveform_l1 = F.l1_loss(prediction, target)
        spectral = (
            decoder_trainer.multi_resolution_stft_loss(prediction, target)
            if float(args.spectral_weight) > 0.0
            else prediction.new_tensor(0.0)
        )
        c_anchor = F.l1_loss(c_hat, exact_c)
        finite_or_raise(waveform_l1, "waveform_l1", step)
        if float(args.spectral_weight) > 0.0:
            finite_or_raise(spectral, "spectral", step)
        finite_or_raise(c_anchor, "c_anchor", step)

        (
            adversarial_generator,
            adversarial_feature,
            adversarial_discriminator,
            adversarial_gate_mean,
            discriminator_grad_norm,
        ) = run_waveform_adversarial(
            args=args,
            step=step,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            prediction=prediction,
            target=target,
        )
        (
            adversarial_delta_generator,
            adversarial_delta_feature,
            adversarial_delta_discriminator,
            adversarial_delta_gate_mean,
            delta_discriminator_grad_norm,
        ) = run_delta_adversarial(
            args=args,
            step=step,
            discriminator=delta_discriminator,
            discriminator_optimizer=delta_discriminator_optimizer,
            prediction=prediction,
            target=target,
        )

        loss = (
            float(args.waveform_l1_weight) * waveform_l1
            + float(args.spectral_weight) * spectral
            + float(args.c_anchor_weight) * c_anchor
            + float(args.adv_weight) * adversarial_generator
            + float(args.adv_feature_weight) * adversarial_feature
            + float(args.adv_delta_weight) * adversarial_delta_generator
            + float(args.adv_delta_feature_weight) * adversarial_delta_feature
        )
        finite_or_raise(loss, "loss", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0).detach().cpu())
        optimizer.step()

        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "waveform_l1": float(waveform_l1.detach().cpu()),
                "waveform_l1_weight": float(args.waveform_l1_weight),
                "spectral": float(spectral.detach().cpu()),
                "spectral_weight": float(args.spectral_weight),
                "c_anchor": float(c_anchor.detach().cpu()),
                "c_anchor_weight": float(args.c_anchor_weight),
                "adversarial_generator": float(adversarial_generator.detach().cpu()),
                "adversarial_feature": float(adversarial_feature.detach().cpu()),
                "adversarial_discriminator": float(adversarial_discriminator.detach().cpu()),
                "adversarial_gate_mean": float(adversarial_gate_mean),
                "adversarial_delta_generator": float(adversarial_delta_generator.detach().cpu()),
                "adversarial_delta_feature": float(adversarial_delta_feature.detach().cpu()),
                "adversarial_delta_discriminator": float(adversarial_delta_discriminator.detach().cpu()),
                "adversarial_delta_gate_mean": float(adversarial_delta_gate_mean),
                "grad_norm": float(grad_norm),
                "discriminator_grad_norm": float(discriminator_grad_norm),
                "delta_discriminator_grad_norm": float(delta_discriminator_grad_norm),
            }
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False), flush=True)

    accounting = parameter_accounting(
        args=args,
        acoustic_model=acoustic_model,
        decoder_model=decoder_model,
        lrc_encoder=lrc_encoder,
    )
    paths = save_checkpoints(
        args=args,
        acoustic_model=acoustic_model,
        acoustic_config=acoustic_config,
        decoder_model=decoder_model,
        decoder_config=decoder_config,
        lrc_encoder=lrc_encoder,
        logs=logs,
        accounting=accounting,
    )
    reload_proof = verify_standalone_reload(paths, device) if bool(args.verify_standalone_reload) else None
    if reload_proof is not None:
        print(json.dumps({"standalone_reload_proof": reload_proof}, ensure_ascii=False), flush=True)

    report = {
        "passed": True,
        "pack_dir": str(args.pack_dir),
        "teacher_decoder": str(args.teacher_decoder),
        "acoustic_checkpoint": str(args.acoustic_checkpoint),
        "decoder_checkpoint": str(args.decoder_checkpoint),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "crop_frames": int(args.crop_frames),
        "source_in_channels": int(source_in_channels),
        "code_dim": int(code_dim),
        "rows": int(len(rows)),
        "chunks": int(len(samples)),
        "parameter_accounting": accounting,
        "optimizer_groups": [
            {
                "name": str(group["name"]),
                "lr": float(group["lr"]),
                "parameters": int(sum(param.numel() for param in group["params"])),
            }
            for group in optimizer_groups
        ],
        "adversarial_discriminator_parameters": int(discriminator_parameter_count),
        "adversarial_delta_discriminator_parameters": int(delta_discriminator_parameter_count),
        "paths": {key: str(value) for key, value in paths.items()},
        "standalone_reload_proof": reload_proof,
        "logs": logs,
        "train_args": jsonable_args(args),
    }
    decoder_trainer.write_json(args.out_dir / "train-report.json", report)
    return report


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
