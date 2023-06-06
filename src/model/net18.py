import torch
from torch import nn
from model.common import *
from model import common
import torch.nn.functional as F


def make_model(args, parent=False):
    return MYNET18(args)


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
            m.append(mini_conv())
        # m.append(mini_conv())
        # m.append(mini_conv())
        # m.append(CALayer(feat))
        # m.append(mini_conv())
        
        
        # m = []
        # for _ in range(3):
        #     m.append(nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1))
        #     m.append(nn.PReLU())      

        # self.convs = nn.Sequential(*m)
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


class DKconv(nn.Module):
    def __init__(self, channels, kernal_size, neg_slope=0.1):
        super(DKconv, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernal_size, stride=1, padding=kernal_size//2),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.m(x)


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25, neg_slope=0.05, inplace=True):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.local1 = DKconv(self.distilled_channels, 1)

        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, stride=1, padding=1)
        self.local2 = DKconv(self.distilled_channels, 3)

        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, stride=1, padding=1)
        self.local3 = DKconv(self.distilled_channels, 5)

        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, stride=1, padding=1)
        self.local4 = nn.Sequential(
            DKconv(self.distilled_channels, 3),
            DKconv(self.distilled_channels, 5))

        self.act = nn.LeakyReLU(neg_slope, inplace)
        self.cbam = CBAMBlock(channel=self.distilled_channels * 4, reduction=self.distilled_channels)
        self.c5 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        final_c1 = self.local1(distilled_c1)

        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        final_c2 = self.local2(distilled_c2)

        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)        
        final_c3 = self.local3(distilled_c3)
        
        out_c4 = self.c4(remaining_c3)
        final_c4 = self.local4(out_c4)

        out = torch.cat([final_c1, final_c2, final_c3, final_c4], dim=1)
        # out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(self.cbam(out)) + input

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
    def __init__(self, feat, scale, isLast=False):
        super(brm, self).__init__()
        self.feat = feat
        self.scale = scale
        self.isLast = isLast

        if isLast:
            self.high_conv = brm_high_inner_block(feat)
            return
        
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.low_conv = brm_low_inner_block(feat)
        self.high_conv = brm_high_inner_block(feat)


    def forward(self, x):
        if self.isLast:
            out = self.high_conv(x)
            return out, out.clone()
        
        down = self.pool(x)
        low = F.interpolate(down, size = x.size()[-2:], mode='bilinear', align_corners=True)
        high = x - low
        up = self.low_conv(low)
        out = self.high_conv(high)

        return up, out


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class MYNET18(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(MYNET18, self).__init__()
        feat = args.n_feats
        scale = args.scale[0]
        self.n_brmblocks = args.n_brmblocks

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, feat * 4, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),
            # nn.Conv2d(feat * 4, feat, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(feat * 4, feat, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1), 
            nn.PReLU()
        )

        self.brm = nn.ModuleList([brm(feat=feat, scale=scale, isLast=True if i == self.n_brmblocks - 1 else False) for i in range(self.n_brmblocks)])

        self.conv_local = nn.ModuleList([nn.Sequential(
                                nn.Conv2d(feat * (i + 2), feat, kernel_size=1),
                                nn.PReLU()) 
                                for i in range(self.n_brmblocks - 1)])

        self.conv_final = nn.ModuleList([nn.Conv2d(feat, feat, 3, stride=1, padding=1) for _ in range(self.n_brmblocks - 1)])
        self.relu = nn.ModuleList([nn.PReLU() for _ in range(self.n_brmblocks - 1)])

        # m_tail = [
        #     common.Upsampler(conv, scale, feat, act=False),
        #     conv(feat, args.n_colors, 3)
        # ]

        self.reduce = nn.Sequential(
            nn.Conv2d(self.n_brmblocks * feat, feat, 1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, 3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, 3, stride=1, padding=1)
            )
        # self.tail = nn.Sequential(*m_tail)
        self.tail = pixelshuffle_block(feat, args.n_colors, upscale_factor=scale, kernel_size=3, stride=1)

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)


    def forward(self, x):
        # print("enter x.shape")
        # print(x.shape)
        x0 = self.head(self.sub_mean(x))

        # up = []
        # x2 = x
        # for unit in self.brm:
        #     x1, x2 = unit(x2)
        #     up.append(x1)

        pre = [x0]
        up = []
        high = x0
        for i, brm in zip(range(self.n_brmblocks), self.brm):
            low, high = brm(high)
            up.append(low)

            if (i != self.n_brmblocks - 1):
                pre.append(high)
                # print(torch.cat(pre, dim=1).shape)
                high = self.conv_local[i](torch.cat(pre, dim=1))

        out = []
        out.append(up[-1])
        for i, conv, relu in zip(range(self.n_brmblocks - 1), self.conv_final, self.relu):
            if i ==0:
                x2 = up[-1] + up[-2]
            else:
                x2 += up[-i-2]
            x2 = conv(x2)
            x2 = relu(x2)
            out.append(x2)
        out = torch.cat(out, dim=1)
        out = self.reduce(out)
        out = self.tail(out + x0)
        out = self.add_mean(out)

        return out


