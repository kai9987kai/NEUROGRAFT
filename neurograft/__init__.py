"""NEUROGRAFT: provenance-preserving experimental neural circuits."""

from .archaeology import ArchaeologyReport, MemoryArchaeologyAssay, ResectionSpec
from .bioelectric import BioelectricConfig, BioelectricMorphogeneticSubstrate, SparseGapJunctionDiffuser
from .byte_bridge import ByteBridgeConfig, ConnectomeByteBridge
from .self_model import ConnectomeForwardSelfModel, SelfModelConfig
from .spectral_diffusion import ChebyshevSpectralDiffusion
from .stp import STPConfig, TsodyksMarkramSubstrate

__version__ = "0.10.0"

__all__ = [
    "ArchaeologyReport",
    "BioelectricConfig",
    "BioelectricMorphogeneticSubstrate",
    "ByteBridgeConfig",
    "ChebyshevSpectralDiffusion",
    "ConnectomeByteBridge",
    "ConnectomeForwardSelfModel",
    "MemoryArchaeologyAssay",
    "ResectionSpec",
    "SelfModelConfig",
    "SparseGapJunctionDiffuser",
    "STPConfig",
    "TsodyksMarkramSubstrate",
    "__version__",
]
