import torch
from torch import nn
from torch.nn import functional as F

" we adapt torch codes from xx and urls"

class EncoderBlock(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, norm_layer=True):
        if norm_layer:
            a = self.bn(self.conv(x))
        a = self.conv(x)
        return self.leakyrelu(a)

class DecoderBlock(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding='same')
        self.upsampling2d = nn.UpsamplingBilinear2d(size=2)
        #self.upsampling2d = F.interpolate(size=2, mode='bilinear')
        #chose to use spade norm or batch norm
        #TBD
        #self.spadenorm =
        self.bn = nn.BatchNorm2d()
        self.dropout = nn.Dropout2d(p=0.2, inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, drop_out=True):
        a = self.bn(self.upsampling2d(self.conv(x)))
        if drop_out:
            return self.leakyrelu(self.dropout(a))
        return self.leakyrelu(a)

class UnetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder1 = EncoderBlock(3, 64, norm_layer=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.decoder1 = DecoderBlock(512, 512)
        self.decoder2 = DecoderBlock(512+512, 512)
        self.decoder3 = DecoderBlock(512+256, 256)
        self.decoder4 = DecoderBlock(256+128, 128)
        self.decoder5 = DecoderBlock(128+64, 64)
        self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding='same')
        self.upsampling2d = nn.UpsamplingBilinear2d(size=2)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # encoder forward:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.relu1(self.encoder6(e5))
        # decoder forward with skip connections
        d1 = self.decoder1(e6)
        d1 = torch.cat([d1, e5], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e4], dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e3], dim=1)
        d4 = self.decoder4(d3)
        d5 = torch.cat([d4, e2], dim=1)
        d6 = self.decoder5(d5)
        output = self.relu2(self.upsampling2d(self.conv(d6)))

        return output

class BasicBlock(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, norm_layer=True):
        if norm_layer:
            a = self.bn(self.conv(x))
        a = self.conv(x)
        return self.leakyrelu(a)


class Discriminator(nn.Module):
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm_layer=False)
        self.block2 = BasicBlock(64, 128)
        self.block3 = BasicBlock(128, 256)
        self.block4 = BasicBlock(256, 512)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=4, stride=1, padding='same')
        self.bn = nn.BatchNorm2d()
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.block1(x)
        a = self.block2(a)
        a = self.block3(a)
        a = self.block4(a)
        a = self.leakyrelu(self.bn(self.conv1(a)))
        a = self.conv2(a)
        return self.sigmoid(a)








