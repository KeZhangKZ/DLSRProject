import torch
from torch import nn
from model.common import *
from model import common
import torch.nn.functional as F
# from pytorch_wavelets import DWTForward, DWTInverse
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel


def make_model(args, parent=False):
    return WRAN(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
 

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

    def forward(self, x):
        out=x*self.ca(x)
        out=out*self.sa(out)
        # return out+residual
        return out



class MKAModule(nn.Module):
    def __init__(self, ratio=4, n_feats=64, alpha=0.1):
        super().__init__()
        self.k1 = nn.Sequential(nn.Conv2d(n_feats, n_feats // ratio, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k2 = nn.Sequential(nn.Conv2d(n_feats, n_feats // ratio, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k3 = nn.Sequential(nn.Conv2d(n_feats, n_feats // ratio, kernel_size=5, stride=1, padding=5//2),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k4 = nn.Sequential(nn.Conv2d(n_feats, n_feats // ratio, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(negative_slope=alpha),
                                 nn.Conv2d(n_feats // ratio, n_feats // ratio, kernel_size=5, stride=1, padding=5//2),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.mk = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.CBAM = CBAMBlock()

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
        cat_out = torch.cat([out1, out2, out3, out4], axis=1)
        inception_out = self.mk(cat_out)
        out = self.CBAM(inception_out)
        return x + out


def dwt2(x, mode, h0_col, h1_col, h0_row, h1_row):
    # Do 1 level of the transform
    ll, high = lowlevel.AFB2D.apply(
        x, h0_col, h1_col, h0_row, h1_row, lowlevel.mode_to_int(mode))
    return ll, high
    

def idwt2(yl, yh, mode, g0_col, g1_col, g0_row, g1_row):
    # 'Unpad' added dimensions
    ll = lowlevel.SFB2D.apply(
        yl, yh, g0_col, g1_col, g0_row, g1_row, lowlevel.mode_to_int(mode))
    return ll


class WRAN(nn.Module):
    def __init__(self, args, conv=default_conv, depth=8, alpha=0.1):
        super(WRAN, self).__init__()
        feat = args.n_feats
        self.scale = args.scale[0]

        # self.xfm = DWTForward(J=1, mode='zero', wave='haar')  # Accepts all wave types available to PyWavelets
        self.mode='zero'
        wave = pywt.Wavelet('haar')
        h0_col, h1_col = wave.dec_lo, wave.dec_hi
        h0_row, h1_row = h0_col, h1_col

        g0_col, g1_col = wave.rec_lo, wave.rec_hi
        g0_row, g1_row = g0_col, g1_col

        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])
        filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])

        # self.ifm = DWTInverse(mode='zero', wave='haar')
        
        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors * 4, feat, kernel_size=5, stride=1, padding=5//2), 
            nn.LeakyReLU(alpha)
        )

        self.MKAMs = nn.ModuleList([MKAModule() for _ in range(depth)])


        self.tail = nn.Sequential(
            nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=3//2),
            nn.LeakyReLU(alpha),
            conv(feat, args.n_colors * 4, 3)
        )


    def forward(self, x):
        b,c,h,w = x.shape

        # Upsmampling
        x_bic = F.interpolate(x, size = (x.size()[-2] * 2, x.size()[-1] * 2), mode='bicubic', align_corners=True)

        # wavelet transform
        cA, hvd = dwt2(x_bic, self.mode, self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        # cA, (cH, cV, cD) = pywt.dwt2(x_bic, 'haar')
        x_wav = torch.cat([cA, torch.reshape(hvd, (b, 3, 128, 128))], dim=1)

        x = self.head(x_wav)

        x_body = x
        for mkam in self.MKAMs:
            x_body = x + mkam(x_body)

        x_tail = self.tail(x_body)

        # reverse wavelet transform
        output = torch.split(x_tail, [1, 3], dim=1)
        out = idwt2(output[0], torch.reshape(output[1], (b, 1, 3, 128, 128)), self.mode, self.g0_col, self.g1_col, self.g0_row, self.g1_row)
        # out = (x_rwav + x_bic).reshape(1, s * self.scale, s * self.scale)

        # return out
        return out + x_bic


