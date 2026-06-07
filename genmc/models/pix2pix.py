"""Pix2Pix baseline (conditional GAN).

A vanilla Pix2Pix (Isola et al., "Image-to-Image Translation with Conditional
Adversarial Networks", CVPR 2017) without the SPADE and coordinate-layer
additions of GenMC, used as a baseline. The U-Net generator and PatchGAN
discriminator follow the canonical reference implementation
(https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _UnetSkipConnectionBlock(nn.Module):
    """Recursive U-Net block with a skip connection.

    Blocks are nested from the innermost (bottleneck) outward, each adding one
    down/up sampling level around its submodule.
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            # ReLU output: optical fluence is non-negative.
            up = [uprelu, upconv, nn.ReLU(True)]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
            if use_dropout:
                model = model + [nn.Dropout(0.5)]

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)  # skip connection


class UnetGenerator(nn.Module):
    """Pix2Pix U-Net generator.

    Args:
        input_nc: number of input channels (default 3, the optical-property map).
        output_nc: number of output channels (default 1, the fluence map).
        num_downs: number of downsamplings; 8 suits a 256x256 input.
        ngf: number of generator filters in the last conv layer.
    """

    def __init__(self, input_nc: int = 3, output_nc: int = 1, num_downs: int = 8,
                 ngf: int = 64, norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        super().__init__()
        block = _UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                         innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            block = _UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=block,
                                             norm_layer=norm_layer, use_dropout=use_dropout)
        block = _UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = _UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=block, norm_layer=norm_layer)
        block = _UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=block, norm_layer=norm_layer)
        self.model = _UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=block,
                                              outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """PatchGAN discriminator (outputs raw patch logits).

    The fluence map (real or generated) is concatenated channel-wise with the
    conditioning optical-property map before being classified.

    Args:
        fluence_channels: channels of the fluence map (default 1).
        cond_channels: channels of the conditioning map (default 3).
        ndf: number of discriminator filters in the first conv layer.
        n_layers: number of strided convolution layers.
    """

    def __init__(self, fluence_channels: int = 1, cond_channels: int = 3,
                 ndf: int = 64, n_layers: int = 3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        input_nc = fluence_channels + cond_channels

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev, nf_mult = nf_mult, min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        nf_mult_prev, nf_mult = nf_mult, min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, fluence: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([fluence, cond], dim=1))
