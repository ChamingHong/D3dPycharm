from math import sqrt
import matplotlib.pyplot as plt
from dcn.modules.deform_conv import *
import functools
import torch.nn.functional as F
# import torch.utils.data


class Net(nn.Module):
    def __init__(self, upscale_factor, in_channel=1, out_channel=1, nf=64):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.in_channel = in_channel

        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.residual_layer = self.make_layer(functools.partial(ResBlock_3d, nf), 5)
        self.TA = nn.Conv2d(7 * nf, nf, 1, 1, bias=True)
        ### reconstruct
        self.reconstruct1 = self.make_layer(functools.partial(ResBlock1, nf), 4)
        self.reconstruct2 = self.make_layer(functools.partial(ResBlock2, nf), 3)
        ###upscale
        self.upscale = nn.Sequential(
            nn.Conv2d(nf, nf * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(nf, out_channel, 3, 1, 1, bias=False),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        b, c, n, h, w = x.size()
        residual = F.interpolate(x[:, :, n // 2, :, :], scale_factor=self.upscale_factor, mode='bilinear',
                                 align_corners=False)
        out = self.input(x)
        out = self.residual_layer(out)
        out = self.TA(out.permute(0,2,1,3,4).contiguous().view(b, -1, h, w))  # B, C, H, W
        out = self.reconstruct1(out)
        out = self.reconstruct2(out)
        ###upscale
        out = self.upscale(out)
        out = torch.add(out, residual)
        return out


class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x


class PyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, pyconv_kernels, pyconv_groups, stride=1, dilation=1, bias=False):
        super(PyConv2d, self).__init__()

        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels, out_channels[i], kernel_size=pyconv_kernels[i],
                                              stride=stride, padding=pyconv_kernels[i] // 2, groups=pyconv_groups[i],
                                              dilation=dilation, bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self, x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))
        return torch.cat(out, 1)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes//4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0]//2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes//4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1]//2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes//4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2]//2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes//4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3]//2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x), self.conv2_4(x)), dim=1)


class PyConv3(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x), self.conv2_3(x)), dim=1)


class PyConv2(nn.Module):
    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def get_pyconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class ResBlock1(nn.Module):
    def __init__(self, nf):
        super(ResBlock1, self).__init__()
        self.dcn1 = conv1x1(nf, 64)
        self.bn = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.dcn2_4 = get_pyconv(64, 64, pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        # self.dcn2_3 = get_pyconv(64, 64, pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.dcn3 = conv1x1(64, nf)
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        residual = x
        out = self.dcn1(x)       # 降维        nf-->64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数      64

        out = self.dcn2_4(out)   # 金字塔卷积    64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数      64

        out = self.dcn2_4(out)   # 金字塔卷积    64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数     64

        out = self.dcn3(out)     # 升维        64-->nf
        out = self.bn2(out)       # 归一化       nf

        return self.lrelu(out + residual)


class ResBlock2(nn.Module):
    def __init__(self, nf):
        super(ResBlock2, self).__init__()
        self.dcn1 = conv1x1(nf, 64)
        self.bn = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.dcn2_4 = get_pyconv(64, 64, pyconv_kernels=[3, 5, 7, 9], pyconv_groups=[1, 4, 8, 16])
        self.dcn2_3 = get_pyconv(64, 64, pyconv_kernels=[3, 5, 7], pyconv_groups=[1, 4, 8])
        self.dcn3 = conv1x1(64, nf)
        self.bn2 = nn.BatchNorm2d(nf)

    def forward(self, x):
        residual = x
        out = self.dcn1(x)       # 降维        nf-->64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数      64

        out = self.dcn2_3(out)   # 金字塔卷积    64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数      64

        out = self.dcn2_3(out)   # 金字塔卷积    64
        out = self.bn(out)       # 归一化       64
        out = self.lrelu(out)    # 激活函数     64

        out = self.dcn3(out)      # 升维        64-->nf
        out = self.bn2(out)       # 归一化       nf

        return self.lrelu(out + residual)


if __name__ == "__main__":
    net = Net(4).cuda()
    from thop import profile
    input = torch.randn(1, 1, 7, 180, 120).cuda()
    flops, params = profile(net, inputs=(input,))
    total = sum([param.nelement() for param in net.parameters()])
    print('   Number of params: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))


