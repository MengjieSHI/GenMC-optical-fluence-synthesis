"""GenMC: a generative Monte-Carlo surrogate for optical-fluence synthesis.

GenMC is a conditional GAN that predicts spatially-resolved optical-fluence
distributions from optical-property-encoded tissue anatomy, enabling real-time
fluence compensation for quantitative photoacoustic imaging.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .losses import DiscriminatorLoss, GeneratorLoss
from .models import (
    GenMCGenerator,
    NLayerDiscriminator,
    PatchDiscriminator,
    UNet,
    UnetGenerator,
    build_discriminator,
    build_generator,
)

__all__ = [
    "__version__",
    "GenMCGenerator",
    "PatchDiscriminator",
    "UnetGenerator",
    "NLayerDiscriminator",
    "UNet",
    "build_generator",
    "build_discriminator",
    "GeneratorLoss",
    "DiscriminatorLoss",
]
