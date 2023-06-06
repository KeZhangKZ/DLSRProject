import torch
from torch import nn
from model.common import *
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return MYNET(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class mini_conv(nn.Module):
    def __init__(self, n_feat=64, n_small_con=4):
        super().__init__()

        m = []
        # Shrink
        m.append(nn.Conv2d(n_feat, n_feat // 4, kernel_size=1))
        m.append(nn.PReLU(n_feat // 4))
        # Map
        for _ in range(n_small_con):
            m.extend([nn.Conv2d(n_feat // 4, n_feat // 4, kernel_size=3, padding=1), nn.PReLU(n_feat // 4)])
        # Expand
        m.extend([nn.Conv2d(n_feat // 4, n_feat, kernel_size=1), nn.PReLU(n_feat)])

        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return x + self.convs(x)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class brm_inner_block(nn.Module):
    def __init__(self, feat):
        super().__init__()
        m = []
        # for _ in range(3):
            # m.append(mini_conv())
        m.append(mini_conv())
        m.append(mini_conv())
        m.append(CALayer(feat))
        m.append(mini_conv())

        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return x + self.convs(x)


class brm(nn.Module):
    def __init__(self, feat, scale):
        super(brm, self).__init__()
        self.feat = feat
        self.scale = scale

        self.pool = nn.AvgPool2d(kernel_size=2)

        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(feat, feat, scale, stride=scale, padding=0),
        #     nn.PReLU()
        # )
        self.low_conv = brm_inner_block(feat)

        # self.down = nn.Sequential(
        #     nn.Conv2d(feat, feat, scale, stride=scale, padding=0),
        #     nn.PReLU()
        # )
        self.high_conv = brm_inner_block(feat)

    def forward(self, x):
        down = self.pool(x)
        low = F.interpolate(down, size = x.size()[-2:], mode='bilinear', align_corners=True)
        high = x - low
        up = self.low_conv(low)
        out = self.high_conv(high)
        # up_out = self.up(x)
        # up = up_out.clone()
        # up = self.up_conv(up)

        # out = x - self.down(up_out.clone())

        # down = out.clone()
        # down = self.down_conv(down)

        # out += down

        return up, out


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class MYNET(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(MYNET, self).__init__()
        feat = args.n_feats
        scale = args.scale[0]
        self.n_resgroups = args.n_brmblocks

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, feat * 4, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),
            # nn.Conv2d(feat * 4, feat, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(feat * 4, feat, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1), 
            nn.PReLU()
        )

        self.brm = nn.ModuleList([brm(feat=feat, scale=scale) for _ in range(self.n_resgroups)])

        self.conv = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride=1, padding=1) for _ in range(self.n_resgroups - 1)])
        self.relu = nn.ModuleList([nn.PReLU() for _ in range(self.n_resgroups - 1)])

        m_tail = [
            common.Upsampler(conv, scale, feat, act=False),
            conv(feat, args.n_colors, 3)
        ]

        # self.tail = nn.Sequential(nn.Conv2d(self.n_resgroups * feat, args.n_colors, 3, stride=1, padding=1), nn.PReLU())
        # TODO: Might add activation function
        self.reduce = nn.Conv2d(self.n_resgroups * feat, feat, 3, stride=1, padding=1)
        self.tail = nn.Sequential(*m_tail)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


    def forward(self, x):
        # print("enter x.shape")
        # print(x.shape)

        x = self.sub_mean(x)
        x = self.head(x)

        up = []
        x2 = x
        for unit in self.brm:
            x1, x2 = unit(x2)
            up.append(x1)

        out = []
        out.append(up[-1])
        for i, conv, relu in zip(range(self.n_resgroups - 1), self.conv, self.relu):
            if i ==0:
                x2 = up[-1] + up[-2]
            else:
                x2 += up[-i-2]
            x2 = conv(x2)
            x2 = relu(x2)
            out.append(x2)
        out = torch.cat(out, dim=1)
        out = self.reduce(out)
        out = self.tail(out)
        out = self.add_mean(out)

        return out


