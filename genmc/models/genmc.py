"""GenMC model: conditional GAN for optical-fluence synthesis.

The framework follows a conditional GAN (cGAN) formulation (Isola et al.,
"Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017):

* :class:`GenMCGenerator` -- a U-Net encoder/decoder (Ronneberger et al., 2015)
  whose decoder batch-normalisation is replaced by SPADE and whose input is
  augmented with 2D coordinate channels. It maps an optical-property-encoded
  tissue anatomy map to the spatially-resolved optical-fluence distribution.

* :class:`PatchDiscriminator` -- a PatchGAN that classifies local image patches
  (16x16 output map) as real (Monte-Carlo derived) or fake (generated), enforcing
  high-frequency realism.

Conventions
-----------
* Generator input  : optical-property-encoded anatomy, shape (B, in_channels, H, W)
                     (default 3 channels: absorption, scattering, Grueneisen).
* Generator output : optical fluence, shape (B, out_channels, H, W) (default 1).
* Discriminator    : receives the channel-wise concatenation of the fluence map
                     (real or generated) and the conditioning optical-property map.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .layers import AddCoords, DoubleConv, Down, SPADEUp


class GenMCGenerator(nn.Module):
    """U-Net generator with SPADE-modulated decoder and coordinate input.

    The encoder is a standard U-Net contracting path: an initial double
    convolution followed by five downscaling stages (max-pool + double conv),
    reducing a 256x256 input to an 8x8 bottleneck. The decoder mirrors the
    encoder with five SPADE-up stages, each conditioned on the input
    optical-property map, and produces a single-channel, non-negative fluence map.

    Args:
        in_channels: channels of the optical-property map (default 3).
        out_channels: channels of the predicted fluence (default 1).
        base_channels: channel width of the first encoder level (default 64).
        use_coords: prepend 2D coordinate channels to the input (default True).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 base_channels: int = 64, use_coords: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_coords = use_coords

        # 2D coordinate layers (CoordConv) on the network input.
        self.add_coords = AddCoords() if use_coords else None
        stem_in = in_channels + (2 if use_coords else 0)

        c = base_channels
        # Encoder (contracting path): 5 downscaling stages -> 8x8 bottleneck.
        self.inc = DoubleConv(stem_in, c)            # 256 ->  64ch
        self.down1 = Down(c, c * 2)                  # 128 -> 128ch
        self.down2 = Down(c * 2, c * 4)              #  64 -> 256ch
        self.down3 = Down(c * 4, c * 8)              #  32 -> 512ch
        self.down4 = Down(c * 8, c * 8)              #  16 -> 512ch
        self.down5 = Down(c * 8, c * 8)              #   8 -> 512ch (bottleneck)

        # Decoder (expanding path): SPADE conditioned on the input property map.
        label_nc = in_channels
        self.up1 = SPADEUp(c * 8, c * 8, c * 8, label_nc)   # 8  -> 16,  512ch
        self.up2 = SPADEUp(c * 8, c * 8, c * 4, label_nc)   # 16 -> 32,  256ch
        self.up3 = SPADEUp(c * 4, c * 4, c * 2, label_nc)   # 32 -> 64,  128ch
        self.up4 = SPADEUp(c * 2, c * 2, c, label_nc)       # 64 -> 128,  64ch
        self.up5 = SPADEUp(c, c, c, label_nc)               # 128-> 256,  64ch

        self.outc = nn.Conv2d(c, out_channels, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True)  # fluence is non-negative

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segmap = x  # SPADE is conditioned on the raw optical-property map
        h = self.add_coords(x) if self.add_coords is not None else x

        e0 = self.inc(h)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        b = self.down5(e4)

        d = self.up1(b, e4, segmap)
        d = self.up2(d, e3, segmap)
        d = self.up3(d, e2, segmap)
        d = self.up4(d, e1, segmap)
        d = self.up5(d, e0, segmap)

        return self.out_act(self.outc(d))


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator operating on local 16x16 patches.

    The fluence map (real or generated) is concatenated channel-wise with the
    conditioning optical-property map and passed through four strided
    convolutions, reducing a 256x256 input to a 16x16 field of patch logits.

    Args:
        fluence_channels: channels of the fluence map (default 1).
        cond_channels: channels of the conditioning optical-property map (default 3).
        base_channels: channel width of the first convolution (default 64).
    """

    def __init__(self, fluence_channels: int = 1, cond_channels: int = 3, base_channels: int = 64):
        super().__init__()
        in_ch = fluence_channels + cond_channels
        c = base_channels

        def block(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(cin, cout, kernel_size=4, stride=2, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_ch, c, norm=False),     # 256 -> 128
            block(c, c * 2),                 # 128 ->  64
            block(c * 2, c * 4),             #  64 ->  32
            block(c * 4, c * 8),             #  32 ->  16
            nn.Conv2d(c * 8, 1, kernel_size=1),  # 16 -> 16 patch logits
        )

    def forward(self, fluence: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([fluence, cond], dim=1))


# Backwards-compatible aliases.
Generator = GenMCGenerator
Discriminator = PatchDiscriminator
