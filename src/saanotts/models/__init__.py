"""Reusable model components extracted from historical training commands."""

from saanotts.models.fsd import HOP_LENGTH
from saanotts.models.fsd import FactorizedSpectralHead
from saanotts.models.fsd import FsdConvNeXtBlock
from saanotts.models.fsd import initialize_factorized_spectral_head
from saanotts.models.fsd import logmag_phase_synthesize
from saanotts.models.rift_tts import RiftAcousticFront
from saanotts.models.rift_tts import RiftConfig
from saanotts.models.rift_tts import RiftFrontConfig
from saanotts.models.rift_tts import RiftTTS
from saanotts.models.rift_tts import add_zero_initialized_front_mel_residual
from saanotts.models.rift_tts import expand_front_shared_cell_net2wider
from saanotts.models.recurrent_transplant import RecurrentTransplantConfig
from saanotts.models.recurrent_transplant import RecurrentTransplantHead
from saanotts.models.recurrent_transplant import recurrent_transplant_parameter_breakdown
from saanotts.models.phaseflow_vocoder import LowRankConvNeXtBlock1d
from saanotts.models.phaseflow_vocoder import PhaseFlowVocoder
from saanotts.models.phaseflow_vocoder import PhaseFlowVocoderConfig
from saanotts.models.phaseflow_vocoder import phaseflow_parameter_count
from saanotts.models.pitch_lattice_acoustic import PitchLatticeAcoustic
from saanotts.models.pitch_lattice_acoustic import PitchLatticeAcousticConfig
from saanotts.models.pitch_lattice_acoustic import pitch_lattice_acoustic_parameter_breakdown
from saanotts.models.pitch_lattice_vocoder import PitchLatticeConfig
from saanotts.models.pitch_lattice_vocoder import PitchLatticeVocoder
from saanotts.models.pitch_lattice_vocoder import pitch_lattice_parameter_count
from saanotts.models.pretrained_fargan import PretrainedFargan
from saanotts.models.pretrained_fargan import PretrainedFarganConfig
from saanotts.models.pretrained_fargan import pretrained_fargan_parameter_breakdown
from saanotts.models.splitband_trellis import SplitBandTrellis
from saanotts.models.splitband_trellis import SplitBandTrellisConfig
from saanotts.models.splitband_trellis import splitband_parameter_breakdown
from saanotts.models.sharedtensor_vocoder import SharedTensorBlock
from saanotts.models.sharedtensor_vocoder import SharedTensorCore
from saanotts.models.sharedtensor_vocoder import SharedTensorVocoder
from saanotts.models.sharedtensor_vocoder import SharedTensorVocoderConfig
from saanotts.models.sharedtensor_vocoder import sharedtensor_parameter_breakdown
from saanotts.models.trellis_rift import FactorizedTemporalConv1d
from saanotts.models.trellis_rift import TrellisRift
from saanotts.models.trellis_rift import TrellisRiftConfig
from saanotts.models.trellis_rift import trellis_rift_parameter_breakdown
from saanotts.models.widebasis_vocoder import WideBasisBlock
from saanotts.models.widebasis_vocoder import WideBasisVocoder
from saanotts.models.widebasis_vocoder import WideBasisVocoderConfig
from saanotts.models.widebasis_vocoder import widebasis_parameter_breakdown
# nirmal_tts is research code that is not part of the published tree, and it is
# not needed by the distillation pipeline. Importing it unconditionally made
# `import saanotts.models` fail outright for anyone who installed the public
# repo -- reported as Ampixa/sanoTTS#7. Guard it, and drop its names from
# __all__ when it is absent, so the modules that ARE shipped still load.
try:
    from saanotts.models.nirmal_tts import NirmalAdaptiveMidbandEnhancer
    from saanotts.models.nirmal_tts import NirmalConfig
    from saanotts.models.nirmal_tts import NirmalDecoderOutput
    from saanotts.models.nirmal_tts import NirmalFrameDeterministicPosteriorEncoder
    from saanotts.models.nirmal_tts import NirmalFrameDeterministicTextPrior
    from saanotts.models.nirmal_tts import NirmalFrameDeterministicTTS
    from saanotts.models.nirmal_tts import NirmalFrameNormalizedPosteriorEncoder
    from saanotts.models.nirmal_tts import NirmalFrameNormalizedTextPrior
    from saanotts.models.nirmal_tts import NirmalFrameNormalizedTTS
    from saanotts.models.nirmal_tts import NirmalGrainDecoder
    from saanotts.models.nirmal_tts import NirmalWaveDecoder
    from saanotts.models.nirmal_tts import NirmalPosteriorEncoder
    from saanotts.models.nirmal_tts import NirmalTTS
    from saanotts.models.nirmal_tts import NirmalTextPrior
    from saanotts.models.nirmal_tts import adjacent_grain_seam_loss
    from saanotts.models.nirmal_tts import estimate_inference_macs_per_second
    from saanotts.models.nirmal_tts import extract_overlapping_grains
    from saanotts.models.nirmal_tts import nirmal_parameter_breakdown
    from saanotts.models.nirmal_tts import nirmal_adaptive_midband_parameter_breakdown
    from saanotts.models.nirmal_tts import nirmal_frame_deterministic_parameter_breakdown
    from saanotts.models.nirmal_tts import nirmal_frame_normalized_parameter_breakdown
    from saanotts.models.nirmal_tts import normalized_overlap_add
    _NIRMAL_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on which tree you installed
    _NIRMAL_AVAILABLE = False

__all__ = [
    "HOP_LENGTH",
    "FactorizedSpectralHead",
    "FsdConvNeXtBlock",
    "initialize_factorized_spectral_head",
    "logmag_phase_synthesize",
    "RiftAcousticFront",
    "RiftConfig",
    "RiftFrontConfig",
    "RiftTTS",
    "add_zero_initialized_front_mel_residual",
    "expand_front_shared_cell_net2wider",
    "RecurrentTransplantConfig",
    "RecurrentTransplantHead",
    "recurrent_transplant_parameter_breakdown",
    "LowRankConvNeXtBlock1d",
    "PhaseFlowVocoder",
    "PhaseFlowVocoderConfig",
    "phaseflow_parameter_count",
    "PitchLatticeAcoustic",
    "PitchLatticeAcousticConfig",
    "pitch_lattice_acoustic_parameter_breakdown",
    "PitchLatticeConfig",
    "PitchLatticeVocoder",
    "pitch_lattice_parameter_count",
    "PretrainedFargan",
    "PretrainedFarganConfig",
    "pretrained_fargan_parameter_breakdown",
    "SplitBandTrellis",
    "SplitBandTrellisConfig",
    "splitband_parameter_breakdown",
    "SharedTensorBlock",
    "SharedTensorCore",
    "SharedTensorVocoder",
    "SharedTensorVocoderConfig",
    "sharedtensor_parameter_breakdown",
    "FactorizedTemporalConv1d",
    "TrellisRift",
    "TrellisRiftConfig",
    "trellis_rift_parameter_breakdown",
    "WideBasisBlock",
    "WideBasisVocoder",
    "WideBasisVocoderConfig",
    "widebasis_parameter_breakdown",
    "NirmalConfig",
    "NirmalAdaptiveMidbandEnhancer",
    "NirmalDecoderOutput",
    "NirmalFrameDeterministicPosteriorEncoder",
    "NirmalFrameDeterministicTextPrior",
    "NirmalFrameDeterministicTTS",
    "NirmalFrameNormalizedPosteriorEncoder",
    "NirmalFrameNormalizedTextPrior",
    "NirmalFrameNormalizedTTS",
    "NirmalGrainDecoder",
    "NirmalWaveDecoder",
    "NirmalPosteriorEncoder",
    "NirmalTTS",
    "NirmalTextPrior",
    "adjacent_grain_seam_loss",
    "estimate_inference_macs_per_second",
    "extract_overlapping_grains",
    "nirmal_parameter_breakdown",
    "nirmal_adaptive_midband_parameter_breakdown",
    "nirmal_frame_deterministic_parameter_breakdown",
    "nirmal_frame_normalized_parameter_breakdown",
    "normalized_overlap_add",
]

if not _NIRMAL_AVAILABLE:
    _NIRMAL_NAMES = {
        "NirmalAdaptiveMidbandEnhancer",
        "NirmalConfig",
        "NirmalDecoderOutput",
        "NirmalFrameDeterministicPosteriorEncoder",
        "NirmalFrameDeterministicTextPrior",
        "NirmalFrameDeterministicTTS",
        "NirmalFrameNormalizedPosteriorEncoder",
        "NirmalFrameNormalizedTextPrior",
        "NirmalFrameNormalizedTTS",
        "NirmalGrainDecoder",
        "NirmalWaveDecoder",
        "NirmalPosteriorEncoder",
        "NirmalTTS",
        "NirmalTextPrior",
        "adjacent_grain_seam_loss",
        "estimate_inference_macs_per_second",
        "extract_overlapping_grains",
        "nirmal_parameter_breakdown",
        "nirmal_adaptive_midband_parameter_breakdown",
        "nirmal_frame_deterministic_parameter_breakdown",
        "nirmal_frame_normalized_parameter_breakdown",
        "normalized_overlap_add",
    }
    __all__ = [_n for _n in __all__ if _n not in _NIRMAL_NAMES]
