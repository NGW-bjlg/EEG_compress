import torch
import torch.nn as nn
import math
import torch.nn.init as init
import os
import numpy

class _ResBLockDB(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(_ResBLockDB, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, stride, 1, bias=True)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        out = self.layers(x)
        residual = x
        out = torch.add(residual, out)
        return out

class _DeblurringMoudle(nn.Module):
    def __init__(self):
        super(_DeblurringMoudle, self).__init__()
        self.conv1     = nn.Conv2d(1, 8, (7, 7), 1, padding=3)
        self.relu      = nn.LeakyReLU(inplace=True)
        self.resBlock1 = self._makelayers(8, 8, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.resBlock2 = self._makelayers(16, 16, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), 2, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.resBlock3 = self._makelayers(32, 32, 1)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, (4, 4), 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, (7, 7), 1, padding=3),
            #nn.Conv2d(8, 8, (3, 1), 1, 0)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, (3, 3), 1, 1),
            # 恢复 23*n
            nn.Conv2d(3, 1, (2, 1), 1, 0)
        )
        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1   = self.relu(self.conv1(x))
        con2   = self.conv2(con1)
        con3   = self.conv3(con2)
        decon1 = self.deconv1(con3)
        deblur_feature = self.deconv2(decon1)
        #deblur_out = self.convout(torch.add(deblur_feature, con1))
        return deblur_feature

class _SRMoudle(nn.Module):
    def __init__(self):
        super(_SRMoudle, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (7, 7), 1, padding=3)
        self.relu = nn.LeakyReLU(inplace=True)
        self.resBlock = self._makelayers(8, 8, 1, 1)
        self.conv2 = nn.Conv2d(8, 8, (3, 3), 1, 1)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def _makelayers(self, inchannel, outchannel, block_num, stride=1):
        layers = []
        for i in range(0, block_num):
            layers.append(_ResBLockDB(inchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        res1 = self.resBlock(con1)
        con2 = self.conv2(res1)
        sr_feature = torch.add(con2, con1)
        return sr_feature

class _GateMoudle(nn.Module):
    def __init__(self):
        super(_GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(17,  8, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 1, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap

class ResBlk(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlk, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(out_size, in_size, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(1)
        )

    def forward(self, x):
        out = self.net(x)
        out = x + out
        return out

class AE2(nn.Module):
    def __init__(self):
        super(AE2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(32, 1), padding=0),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=2, padding=0, groups=22),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=2, padding=0, groups=22),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=2, padding=0, groups=22),
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=2, padding=0, groups=22),
            nn.LeakyReLU(),

        )

        self.deblurMoudle = _DeblurringMoudle()
        self.srMoudle = _SRMoudle()
        self.geteMoudle = _GateMoudle()

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(1, 2), stride=2),
            ResBlk(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=(1, 2), stride=2),
            ResBlk(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=(1, 2), stride=2),
            ResBlk(32, 32),
            nn.ConvTranspose2d(32, 32, kernel_size=(1, 2), stride=2),
            ResBlk(32, 32),
        )

        self.decoder2 = nn.Sequential(
            ResBlk(1, 16),
            nn.Upsample(scale_factor=(1, 2), mode='bilinear'),
            ResBlk(1, 16),
            nn.Upsample(scale_factor=(1, 2), mode='bilinear'),
            ResBlk(1, 16),
            nn.Upsample(scale_factor=(1, 2), mode='bilinear'),
            ResBlk(1, 16),
            nn.Upsample(scale_factor=(1, 2), mode='bilinear'),
            ResBlk(1, 16),
        )

        self.conv1 = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x):
        batchsz = x.size(0)
        time_len = x.size(3)
        x = self.encoder(x)
        leng = x.size(3)

        x = x.view(batchsz, 1, 32, leng)
        deblur_feature = self.deblurMoudle(x)
        sr_feature = self.srMoudle(x)
        x_c = torch.cat([x, deblur_feature, sr_feature], dim=1)
        x_f = self.geteMoudle(x_c)
        x1 = x_f.view(batchsz, 32, 1, leng)
        x1 = self.decoder1(x1)
        x1 = x1.view(batchsz, 1, 32, time_len)
        x2 = self.decoder2(x_f)
        x3 = torch.cat([x1, x2], dim=1)
        x3 = self.conv1(x3)
        x3 = x3.view(batchsz, 1, 32, time_len)
        return x3

if __name__ == '__main__':
    x = numpy.random.rand(8, 1, 23, 2560)
    x = x.astype(numpy.float32)
    x = torch.from_numpy(x)
    #deb = _DeblurringMoudle()
    #sr = _SRMoudle()
    sr = AE2()
    x2 = sr(x)
    #x1 = deb(x)
    print(x)