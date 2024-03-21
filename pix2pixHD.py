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
        self.layers = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, nc, kernel_size=3, padding=0),
            #nn.InstanceNorm2d(nc, affine=False),
            nn.BatchNorm2d(nc, affine=False),

            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(nc, nc, kernel_size=3, padding=0),
            nn.InstanceNorm2d(nc, affine=False),
            nn.BatchNorm2d(nc, affine=False),

        )
        def forward(self, x):
            return x + self.layers(x)

