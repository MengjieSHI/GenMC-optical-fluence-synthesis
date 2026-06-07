"""Model definitions for GenMC and the U-Net / Pix2Pix baselines."""

from __future__ import annotations

from .genmc import GenMCGenerator, PatchDiscriminator
from .pix2pix import NLayerDiscriminator, UnetGenerator
from .unet import UNet

__all__ = [
    "GenMCGenerator",
    "PatchDiscriminator",
    "UnetGenerator",
    "NLayerDiscriminator",
    "UNet",
    "build_generator",
    "build_discriminator",
]


def build_generator(model: str = "genmc", in_channels: int = 3, out_channels: int = 1):
    """Construct a generator by name (``genmc``, ``pix2pix`` or ``unet``)."""
    model = model.lower().strip()
    if model == "genmc":
        return GenMCGenerator(in_channels=in_channels, out_channels=out_channels)
    if model == "pix2pix":
        return UnetGenerator(input_nc=in_channels, output_nc=out_channels)
    if model == "unet":
        return UNet(n_channels=in_channels, n_classes=out_channels)
    raise ValueError(f"Unknown model '{model}'. Choose from: genmc, pix2pix, unet.")


def build_discriminator(model: str = "genmc", fluence_channels: int = 1, cond_channels: int = 3):
    """Construct the discriminator for an adversarial model (``genmc`` or ``pix2pix``).

    Returns ``None`` for ``unet``, which is trained without a discriminator.
    """
    model = model.lower().strip()
    if model == "genmc":
        return PatchDiscriminator(fluence_channels=fluence_channels, cond_channels=cond_channels)
    if model == "pix2pix":
        return NLayerDiscriminator(fluence_channels=fluence_channels, cond_channels=cond_channels)
    if model == "unet":
        return None
    raise ValueError(f"Unknown model '{model}'. Choose from: genmc, pix2pix, unet.")
