import torch
from torch import nn
from model.common import *
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return MYNET5(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


# class mini_conv(nn.Module):
#     def __init__(self, n_feat=64, n_small_con=4):
#         super().__init__()

#         m = []
#         # Shrink
#         m.append(nn.Conv2d(n_feat, n_feat // 4, kernel_size=1))
#         m.append(nn.PReLU(n_feat // 4))
#         # Map
#         for _ in range(n_small_con):
#             m.extend([nn.Conv2d(n_feat // 4, n_feat // 4, kernel_size=3, padding=1), nn.PReLU(n_feat // 4)])
#         # Expand
#         m.extend([nn.Conv2d(n_feat // 4, n_feat, kernel_size=1), nn.PReLU(n_feat)])

#         self.convs = nn.Sequential(*m)

#     def forward(self, x):
#         return x + self.convs(x)
   

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output
    

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=64,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        out=x*self.ca(x)
        out=out*self.sa(out)
        # return out+residual
        return out


class brm_low_inner_block(nn.Module):
    def __init__(self, feat):
        super().__init__()
        m = []
        for _ in range(3):
            m.append(nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1))
            m.append(nn.PReLU())
        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.convs(x)


class MKAModule(nn.Module):
    def __init__(self, n_feats=64, alpha=0.1):
        super().__init__()
        self.k1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=5//2),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k4 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(negative_slope=alpha),
                                 nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=5//2),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.reduce = nn.Sequential(nn.Conv2d(n_feats * 4, n_feats, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.CBAM = CBAMBlock(channel=n_feats,reduction=n_feats//4)

    def forward(self, x):
        out1 = self.k1(x)
        out2 = self.k2(x)
        out3 = self.k3(x)
        out4 = self.k4(x)
        # print(x.shape)
        # print(out1.shape)
        # print(out2.shape)
        # print(out3.shape)
        # print(out4.shape)
        cat_out = torch.cat([out1, out2, out3, out4], dim=1)
        inception_out = self.reduce(cat_out)
        out = self.CBAM(inception_out)
        return x + out


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, neg_slope=0.05, inplace=True):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.mkam1 = MKAModule(n_feats=self.distilled_channels)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, stride=1, padding=1)
        self.mkam2 = MKAModule(n_feats=self.distilled_channels)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, stride=1, padding=1)
        self.mkam3 = MKAModule(n_feats=self.distilled_channels)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, stride=1, padding=1)
        self.mkam4 = MKAModule(n_feats=self.distilled_channels)
        self.act = nn.LeakyReLU(neg_slope, inplace)
        self.cbam = CBAMBlock(channel=self.distilled_channels * 4, reduction=self.distilled_channels)
        self.c5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        final_c1 = self.mkam1(distilled_c1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        final_c2 = self.mkam2(distilled_c2)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        final_c3 = self.mkam3(distilled_c3)
        out_c4 = self.c4(remaining_c3)
        final_c4 = self.mkam4(out_c4)
        out = torch.cat([final_c1, final_c2, final_c3, final_c4], dim=1)
        # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cbam(out)) + input

        # print(f"input={input}\n")
        # print(f"out_c1={out_c1}\ndistilled_c1={distilled_c1}remaining_c1={remaining_c1}")
        # print(f"out_c2={out_c2}\ndistilled_c2={distilled_c2}remaining_c2={remaining_c2}")
        # print(f"out_c3={out_c3}\ndistilled_c3={distilled_c3}remaining_c3={remaining_c3}")
        # print(f"out_c4={out_c4}\nout={out}out_fused={out_fused}")

        return out_fused
    

class brm_high_inner_block(nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.imdm = IMDModule(feat)
        # self.mkas = nn.Sequential(MKAModule(), MKAModule())
        # # for _ in range(3):
        #     # m.append(mini_conv())
        # m.append(MKAModule())
        # m.append(MKAModule())

        # self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.imdm(x)


class brm(nn.Module):
    def __init__(self, feat, scale):
        super(brm, self).__init__()
        self.feat = feat
        self.scale = scale

        self.pool = nn.AvgPool2d(kernel_size=2)
        self.low_conv = brm_low_inner_block(feat)
        self.high_conv = brm_high_inner_block(feat)


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

        # out += high

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


class MYNET5(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(MYNET5, self).__init__()
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

        self.conv_local = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(feat * (i + 2), feat, kernel_size=1),
                                nn.PReLU()) 
                                for i in range(self.n_resgroups - 1)])

        self.conv_final = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride=1, padding=1) for _ in range(self.n_resgroups - 1)])
        self.relu = nn.ModuleList([nn.PReLU() for _ in range(self.n_resgroups - 1)])

        m_tail = [
            common.Upsampler(conv, scale, feat, act=False),
            conv(feat, args.n_colors, 3)
        ]

        self.reduce = nn.Conv2d(self.n_resgroups * feat, feat, 3, stride=1, padding=1)
        self.tail = nn.Sequential(*m_tail)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


    def forward(self, x):
        # print("enter x.shape")
        # print(x.shape)

        x = self.sub_mean(x)
        x = self.head(x)

        # up = []
        # x2 = x
        # for unit in self.brm:
        #     x1, x2 = unit(x2)
        #     up.append(x1)

        pre = [x]
        up = []
        high = x
        for i, brm in zip(range(self.n_resgroups), self.brm):
            low, high = brm(high)
            up.append(low)

            if (i != self.n_resgroups - 1):
                pre.append(high)
                # print(torch.cat(pre, dim=1).shape)
                high = self.conv_local[i](torch.cat(pre, dim=1))

        out = []
        out.append(up[-1])
        for i, conv, relu in zip(range(self.n_resgroups - 1), self.conv_final, self.relu):
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


