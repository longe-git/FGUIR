import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class PhaseFusion(nn.Module):
    def __init__(self, dim, ratio=4):
        super(PhaseFusion, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 1)
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim // ratio, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // ratio, dim, 5, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.conv(self.att(y) * x) + x


class FreGuide(nn.Module):
    def __init__(self, dim):
        super(FreGuide, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 1)
        if dim == 96:
            self.em_in = nn.Sequential(
                PatchEmbed(patch_size=2, in_chans=3, embed_dim=dim//2, kernel_size=3),
                PatchEmbed(patch_size=2, in_chans=dim//2, embed_dim=dim, kernel_size=3)
            )
        elif dim == 48:
            self.em_in = PatchEmbed(patch_size=2, in_chans=3, embed_dim=dim, kernel_size=3)
        else:
            self.em_in = PatchEmbed(patch_size=1, in_chans=3, embed_dim=dim, kernel_size=3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
        )
        self.fu = PhaseFusion(dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x, input):
        x_in = self.conv(x)
        input = self.em_in(input)
        x_fft = torch.fft.rfft2(x_in.float(), norm='backward')
        x_amp = torch.abs(x_fft)    # 修改图像振幅
        x_phase = torch.angle(x_fft)    # 输入相位引导合成
        in_fft = torch.fft.rfft2(input.float(), norm='backward')
        # in_amp = torch.abs(in_fft)      # 弃掉输入的振幅
        in_phase = torch.angle(in_fft)  # 保留相位
        x_amp = self.conv1(x_amp)
        x_phase = self.fu(x_phase, in_phase)
        x_fft_out = torch.fft.irfft2(x_amp * torch.exp(1j * x_phase), norm='backward')
        out = self.conv2(x_fft_out)
        return out + x


class ADWConv(nn.Module):
    def __init__(self, dim):
        super(ADWConv, self).__init__()
        self.dw1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.dw2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        self.dw3 = nn.Conv2d(dim//4, dim//4, kernel_size=3, padding=1, stride=1, groups=dim//4)
        self.pw1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, c, _, _ = x.shape
        x1 = self.dw1(x)
        x1, x1_ = torch.split(x1, (c // 2, c // 2), dim=1)
        x2 = self.dw2(x1)
        x2, x2_ = torch.split(x2, (c // 4, c // 4), dim=1)
        x3 = self.dw3(x2)
        gate = self.pw1(x)
        out = gate * torch.cat((x1_, x2_, x3), dim=1)
        return out


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        return self.conv(x)


class CA(nn.Module):
    def __init__(self, inc, ratio=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.MLP = nn.Sequential(
            nn.Conv2d(inc, inc // ratio, 1, bias=True), nn.GELU(),
            nn.Conv2d(inc // ratio, inc, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        x = self.avg_pool(x)
        return self.MLP(x)


class Conv2d_rd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):

        super(Conv2d_rd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            conv_shape = conv_weight.shape
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(conv_shape[0], conv_shape[1], 5 * 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(conv_shape[0], conv_shape[1], 5 * 5)
            conv_weight = Rearrange('c_in c_out k1 k2 -> c_in c_out (k1 k2)')(conv_weight)
            conv_weight_rd[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = conv_weight[:, :, [0, 2, 4, 10, 14, 20, 22, 24]]-conv_weight[:, :, [6, 7, 8, 11, 13, 16, 17, 18]]*self.theta
            conv_weight_rd[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = conv_weight[:, :, [6, 7, 8, 11, 13, 16, 17, 18]]-conv_weight[:, :, [12, 12, 12, 12, 12, 12, 12, 12]]*self.theta
            conv_weight_rd[:, :, 12] = conv_weight[:, :, 12] * (1 - self.theta)
            conv_weight_rd = conv_weight_rd.view(conv_shape[0], conv_shape[1], 5, 5)
            out_diff = nn.functional.conv2d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)

            return out_diff


class DSA(nn.Module):
    def __init__(self, kernel_size=7):
        super(DSA, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=True)
        self.convd = Conv2d_rd(2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x1 = self.conv(x)
        x2 = self.convd(x)
        out = x1 + x2
        return self.sigmoid(out)


class Conv1d_rd(nn.Module):
    def __init__(self, kernel_size=5, stride=1,
                 padding=2, dilation=1, groups=1, bias=False, theta=1.0):
        super(Conv1d_rd, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view([b, 1, c])
        if math.fabs(self.theta - 0.0) < 1e-8:
            out_normal = self.conv(x)
            return out_normal
        else:
            conv_weight = self.conv.weight
            if conv_weight.is_cuda:
                conv_weight_rd = torch.cuda.FloatTensor(1, 1, 5).fill_(0)
            else:
                conv_weight_rd = torch.zeros(1, 1, 5)
            conv_weight_rd[:, :, [0, 4]] = conv_weight[:, :, [0, 4]] - conv_weight[:, :, [1, 3]] * self.theta
            conv_weight_rd[:, :, [1, 3]] = conv_weight[:, :, [1, 3]] - conv_weight[:, :, [2, 2]] * self.theta
            conv_weight_rd[:, :, [2]] = conv_weight[:, :, [2]] * (1-self.theta)
            out_diff = nn.functional.conv1d(input=x, weight=conv_weight_rd, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups)
            out_diff = out_diff.view([b, c, 1, 1])

            return out_diff


class DCA(nn.Module):
    def __init__(self, inc, ratio=8):
        super(DCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.c1 = Conv1d_rd(bias=True)
        self.MLP = nn.Sequential(
            nn.Conv2d(inc * 2, inc//ratio, 3, padding=1, bias=True), nn.GELU(),
            nn.Conv2d(inc // ratio, inc, 1, bias=True), nn.Sigmoid())

    def forward(self, x):
        x = self.avg_pool(x)
        x1 = self.c1(x)
        x2 = torch.cat((x, x1), dim=1)
        return self.MLP(x2)


class FE(nn.Module):
    def __init__(self, dim):
        super(FE, self).__init__()

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.aconv = ADWConv(dim)

        self.sa = DSA()
        self.ca = DCA(dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            # nn.GELU(),
            # nn.Conv2d(dim * 4, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.aconv(x)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.ca(x) * x, self.sa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList(
            [FE(dim=dim) for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_chans=3, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Net(nn.Module):
    def __init__(self, inc=3, outc=6, dims=[24, 48, 96], depths=[2, 2, 4]):
        super(Net, self).__init__()
        self.patch_size = 4
        self.em1 = PatchEmbed(patch_size=1, in_chans=inc, embed_dim=dims[0], kernel_size=3)
        self.em2 = PatchEmbed(patch_size=2, in_chans=dims[0], embed_dim=dims[1], kernel_size=3)
        self.em3 = PatchEmbed(patch_size=2, in_chans=dims[1], embed_dim=dims[2], kernel_size=3)
        self.um3 = PatchUnEmbed(patch_size=2, embed_dim=dims[2], out_chans=dims[1], kernel_size=3)
        self.um2 = PatchUnEmbed(patch_size=2, embed_dim=dims[1], out_chans=dims[0], kernel_size=3)
        self.um1 = PatchUnEmbed(patch_size=1, embed_dim=dims[0], out_chans=outc, kernel_size=3)

        self.layer1 = BasicLayer(dim=dims[0], depth=depths[0])
        self.layer2 = BasicLayer(dim=dims[1], depth=depths[1])
        self.layer3 = BasicLayer(dim=dims[2], depth=depths[2])
        self.layer4 = BasicLayer(dim=dims[1], depth=depths[1])
        self.layer5 = BasicLayer(dim=dims[0], depth=depths[0])
        self.back1 = BasicLayer(dim=dims[0], depth=depths[0]//2)
        self.forward1 = BasicLayer(dim=dims[0], depth=depths[0]//2)
        self.back2 = BasicLayer(dim=dims[1], depth=depths[1]//2)
        self.forward2 = BasicLayer(dim=dims[1], depth=depths[1]//2)

        self.skip1 = nn.Conv2d(dims[0], dims[0], 1)
        self.skip2 = nn.Conv2d(dims[0], dims[0], 1)
        self.skip3 = nn.Conv2d(dims[1], dims[1], 1)
        self.skip4 = nn.Conv2d(dims[1], dims[1], 1)
        self.fu1 = nn.Conv2d(3*dims[0], dims[0], 1)
        self.fu2 = nn.Conv2d(3*dims[1], dims[1], 1)

        self.fre3 = FreGuide(dims[2])
        self.fre2d = FreGuide(dims[1])
        self.fre2e = FreGuide(dims[1])

    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def cc(self, x):
        x = (x + 1)/2
        r, g, b = torch.split(x, (1, 1, 1), dim=1)
        r_ = r + (1-r)*(g-r)*g
        b_ = b + (1-b)*(g-b)*g
        y = torch.cat((r_, g, b_), dim=1)
        y = y*2-1
        return y

    def forward_features(self, x):
        idx = x
        idx = self.cc(idx)
        x = self.em1(x)
        # 编码一
        skip11 = x
        x = self.layer1(x)
        skip12 = x
        back1 = self.skip1(skip11) - self.back1(x)
        f1 = self.forward1(back1) + self.skip2(x)

        # 编码区二
        x = self.em2(x)
        skip21 = x
        x = self.fre2e(x, idx)
        x = self.layer2(x)
        skip22 = x
        back2 = self.skip3(skip21) - self.back2(x)
        f2 = self.forward2(back2) + self.skip4(x)

        # 编码区三
        x = self.em3(x)
        x = self.fre3(x, idx)
        x = self.layer3(x)

        # 解码二
        x = self.um3(x)
        x = self.fu2(torch.cat((x, skip22, f2), dim=1))
        x = self.fre2d(x, idx)
        x = self.layer4(x)

        # 解码一
        x = self.um2(x)
        x = self.fu1(torch.cat((x, skip12, f1), dim=1))
        x = self.layer5(x)

        x = self.um1(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        feat = self.forward_features(x)
        K, B = torch.split(feat, (3, 3), dim=1)
        x = K * x - B + x
        x = x[:, :, :H, :W]

        return x


def Net_s():   # small
    return Net5(depths=[1, 1, 2])


def Net_b():   # big
    return Net5(depths=[2, 2, 4])


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)

    network = Net()
    out = network(x)
    print(out.shape)


