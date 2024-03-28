import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    '''
    ResidualBlock Class
    Values
         channels: the number of channels throughout the residual block: a scalar
    '''
    def __init__(self, nc):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, nc, kernel_size=3, padding=0),
            #nn.InstanceNorm2d(nc, affine=False),
            nn.BatchNorm2d(nc, affine=False),

            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, nc, kernel_size=3, padding=0),
            #nn.InstanceNorm2d(nc, affine=False),
            nn.BatchNorm2d(nc, affine=False),

        )
        def forward(self, x):
            return x + self.layers(x)

class GlobalGenerator(nn.Module):
    '''
    global generator for transfering styles at low resolution
    values:
    in_nc: number of input channels
    out_nc: number of output channels
    base_nc: number of channels in the first convolutional layer
    fb_blocks: number of frontend blocks
    res_blocks: number of residual blocks

    '''
    def __init__(self, in_nc, out_nc, base_nc, fb_blocks, res_blocks):
        super(GlobalGenerator, self).__init__()

        # initial convolutional layer
        g1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_nc, base_nc, kernel_size=7, padding=0),
            nn.InstanceNorm2d(base_nc, affine=False),
            #nn.BatchNorm2d(base_nc, affine=False), # whether to scale and shift the normalised value
            nn.ReLU(inplace=True),
        ]
        channels = base_channels
        # frontend blocks
        for _ in range(fb_blocks):
            g1 += [
                nn.Conv2d(channels, 2*channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2*channels, affine=False),
                #nn.BatchNorm2d(2*channels, affine=False),
                nn.ReLU(inplace=True),
            ]
            channels *= 2
        # residual blocks
        for _ in range(res_blocks):
            g1 += [ResidualBlock(channels)]
        # backend blocks
        for _ in range(fb_blocks):
            g1 += [
                # different upsampling methods
                nn.ConvTranspose2d(channels, channels//2, kernel_size=3,  stride=2, padding=1, output_padding=1),
                # nn.Conv2d(channels, channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.UpsamplingBilinear2d(scale_factor=2),
                nn.InstanceNorm2d(channels//2, affine=False),
                # nn.BatchNorm2d(channels//2, affine=False), # spade norm
                nn.ReLU(inplace=True),
            ]
            channels //=2
        # Ouput convolutional layer as its own nn.Sequential since it will be omitted in second train phase
        self.out_layers = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_nc, out_nc, kernel_size=7, padding=0),
            nn.ReLU(),
            #nn.Tanh(),
        )
        self.g1 = nn.Sequential(*g1) #* unpackes a sequence or collection into positional arguments
    def forward(self, x):
        x = self.g1(x)
        x = self.out_layers(x)
        return x

class LocalEnhancer(nn.Module):
    '''
    localenhancer class:
    local enhancer subgenerator for handling larger scale image.
    values:
    in_nc: the number of input channels, a scalar
    out_nc: the number of output channels, a scalar
    base_nc: the number of channels in first conv layer, a scalar
    global_fb_blocks: the number of global generator frontend/backend blocks, a scalar
    global_res_blocks: the number of global generator residual blocks, a scalar
    local_res_blocks: the number of local enhancer residual blocks, a scalar
    '''
    def __init__(self, in_nc, out_nc, base_nc=32, global_fb_blocks=3, global_res_blocks=9, local_res_blocks=3):
        super().__init__()
        global_base_nc = 2 * base_nc

        # downsampling layer for high-res -> low-res input to g1
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        # initialise global generator without its output layers
        self.g1 = GlobalGenerator(
            in_nc, out_nc, base_nc=global_nc, fb_blocks=global_fb_blocks, res_blocks = global_res_blocks,
        ).g1
        self.g2 = nn.ModuleList()

        # initialise local frontend block
        self.g2.append(
            nn.Sequential(
                # Initial convolutional layer
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=0),
                nn.InstanceNorm2d(base_channels, affine=False),
                # nn.BatchNorm2d(base_channels, affine=False),
                nn.ReLU(inplace=True),

                # Frontend block
                nn.Conv2d(base_channels, 2 * base_channels, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(2 * base_channels, affine=False),
                # nn.BatchNorm2d(2 * base_channels, affine=False),
                nn.ReLU(inplace=True),
            )
        )
        # Initialise local residual and backend blocks
        self.g2.append(
            nn.Sequential(
                # Residual blocks
                *[ResidualBlock(2 * base_channels) for _ in range(local_res_blocks)],

                # Backend blocks
                nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=3, stride=2, padding=1,
                                   output_padding=1),
                # nn.Conv2d(2 * base_channels, base_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                # nn.UpsamplingBilinear2d(scale_factor=2),
                nn.InstanceNorm2d(base_channels, affine=False),
                # nn.BatchNorm2d(base_channels, affine=False), # spade norm
                nn.ReLU(inplace=True),

                # Output convolutional layer
                nn.ReflectionPad2d(3),
                nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=0),
                nn.ReLU(),
                # nn.Tanh(),
            )
        )
        def forward(self, x):
            # Get output from g1_B
            x_g1 = self.downsample(x)
            x_g1 = self.g1(x_g1)
            # Get output from g2_F
            x_g2 = self.g2[0](x)
            # Get final output from g2_B
            return self.g2[1](x_g1 + x_g2)

