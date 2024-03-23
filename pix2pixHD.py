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
