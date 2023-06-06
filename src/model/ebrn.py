import torch
from torch import nn
from model.common import *
from model import common


def make_model(args, parent=False):
    return EBRN(args)

class brm_inner_block(nn.Module):
    def __init__(self, feat):
        super().__init__()
        m = []
        for _ in range(3):
            m.append(nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1))
            m.append(nn.PReLU())
        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.convs(x)


class brm(nn.Module):
    def __init__(self, feat, scale):
        super(brm, self).__init__()
        self.feat = feat
        self.scale = scale

        self.up = nn.Sequential(
            nn.ConvTranspose2d(feat, feat, scale, stride=scale, padding=0),
            nn.PReLU()
        )
        self.up_conv = brm_inner_block(feat)

        self.down = nn.Sequential(
            nn.Conv2d(feat, feat, scale, stride=scale, padding=0),
            nn.PReLU()
        )
        self.down_conv = brm_inner_block(feat)

    def forward(self, x):
        up_out = self.up(x)
        up = up_out.clone()
        up = self.up_conv(up)

        out = x - self.down(up_out.clone())

        down = out.clone()
        down = self.down_conv(down)

        out += down

        return up, out


class EBRN(nn.Module):
    def __init__(self, args):
        super(EBRN, self).__init__()
        feat = args.n_feats
        scale = args.scale[0]
        self.n_resgroups = args.n_brmblocks

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, feat * 4, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),
            nn.Conv2d(feat * 4, feat, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),
            nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1), 
            nn.PReLU()
        )

        self.brm = nn.ModuleList([brm(feat=feat, scale=scale) for _ in range(self.n_resgroups)])

        self.conv = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride=1, padding=1) for _ in range(self.n_resgroups - 1)])
        self.relu = nn.ModuleList([nn.PReLU() for _ in range(self.n_resgroups - 1)])

        self.tail = nn.Sequential(nn.Conv2d(self.n_resgroups * feat, args.n_colors, 3, stride=1, padding=1), nn.PReLU())
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
        out = self.tail(out)
        out = self.add_mean(out)

        return out


