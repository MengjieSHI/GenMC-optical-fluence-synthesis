"""Building blocks for the GenMC generator.

This module implements the two architectural components that distinguish the
GenMC generator from a vanilla U-Net, as described in the Methods (Sec. 5.2)
of the paper:

* :class:`AddCoords` / :class:`CoordConv2d` -- 2D coordinate layers that append
  normalised pixel coordinates to a feature map, improving the network's spatial
  awareness of the diffusion-driven nature of light propagation
  (Liu et al., "An Intriguing Failing of Convolutional Neural Networks and the
  CoordConv Solution", NeurIPS 2018).

* :class:`SPADE` -- spatially-adaptive (de)normalisation, which replaces batch
  normalisation in the decoder so that activations are modulated by the spatial
  layout of the conditioning optical-property map, preserving structural fidelity
  at tissue boundaries (Park et al., "Semantic Image Synthesis with
  Spatially-Adaptive Normalization", CVPR 2019).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    """Append normalised coordinate channels to a feature map.

    Two channels are added by default, holding the normalised ``x`` and ``y``
    pixel coordinates in the range ``[-1, 1]``. When ``with_r`` is ``True`` an
    additional radial-distance channel is appended.
    """

    def __init__(self, with_r: bool = False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype)
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype)
        xx = xx.view(1, 1, 1, w).expand(b, 1, h, w)
        yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
        out = torch.cat([x, xx, yy], dim=1)
        if self.with_r:
            rr = torch.sqrt(xx ** 2 + yy ** 2)
            out = torch.cat([out, rr], dim=1)
        return out

    @property
    def extra_channels(self) -> int:
        return 3 if self.with_r else 2


class CoordConv2d(nn.Module):
    """A :class:`~torch.nn.Conv2d` preceded by an :class:`AddCoords` layer."""

    def __init__(self, in_channels: int, out_channels: int, with_r: bool = False, **conv_kwargs):
        super().__init__()
        self.add_coords = AddCoords(with_r=with_r)
        self.conv = nn.Conv2d(in_channels + self.add_coords.extra_channels, out_channels, **conv_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.add_coords(x))


class SPADE(nn.Module):
    """Spatially-adaptive denormalisation.

    The input ``x`` is first normalised with a parameter-free batch norm, then
    modulated by a spatially-varying scale (``gamma``) and bias (``beta``) that
    are predicted from the conditioning map ``segmap`` (here, the optical-property
    encoded tissue anatomy). ``segmap`` is resized to the resolution of ``x`` so
    that a single SPADE module can be reused at every decoder scale.

    Args:
        norm_nc: number of channels of the activation map being modulated.
        label_nc: number of channels of the conditioning map.
        hidden_nc: width of the shared intermediate convolution.
        kernel_size: kernel size of the SPADE convolutions.
    """

    def __init__(self, norm_nc: int, label_nc: int, hidden_nc: int = 128, kernel_size: int = 3):
        super().__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        padding = kernel_size // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)
        self.mlp_beta = nn.Conv2d(hidden_nc, norm_nc, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class DoubleConv(nn.Module):
    """(Conv 3x3 -> BatchNorm -> ReLU) x 2, the standard U-Net encoder unit."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling stage: 2x2 max-pool followed by a :class:`DoubleConv`."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class SPADEUp(nn.Module):
    """Decoder stage of the GenMC generator.

    Following the Methods, each decoder layer (i) bilinearly upsamples the feature
    map from the previous layer, (ii) concatenates it with the corresponding
    encoder features via a skip connection, and (iii) applies two 3x3
    convolutions. Batch normalisation is replaced by :class:`SPADE`, so the two
    convolutions are each followed by SPADE (conditioned on the optical-property
    map) and a ReLU activation.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, label_nc: int,
                 mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.spade1 = SPADE(mid_channels, label_nc)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.spade2 = SPADE(out_channels, label_nc)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one size mismatches from odd spatial dimensions.
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        x = self.act(self.spade1(self.conv1(x), segmap))
        x = self.act(self.spade2(self.conv2(x), segmap))
        return x
