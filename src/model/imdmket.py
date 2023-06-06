import torch
from torch import nn
from model.common import *
import torch.nn.functional as F
from collections import OrderedDict
from model.transformer import MLABlock


def make_model(args, parent=False):
    return IMDMKET(args)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.contrast = stdv_channels
        # self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        contrast = self.contrast(x)
        # max_result=self.maxpool(x)
        avg_result=self.avgpool(x) + contrast
        # max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(avg_out)
        # output=self.sigmoid(max_out+avg_out)
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

    def __init__(self, channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out=x*self.ca(x)
        out=out*self.sa(out)
        # return out+residual
        return out


class DKconv(nn.Module):
    def __init__(self, channels, kernal_size, neg_slope=0.1):
        super(DKconv, self).__init__()
        self.m = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernal_size, stride=1, padding=kernal_size//2),
            nn.LeakyReLU(neg_slope)
        )

    def forward(self, x):
        return self.m(x)


class MKAModule(nn.Module):
    def __init__(self, n_feats=64, alpha=0.1):
        super().__init__()
        self.k1 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k2 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.k3 = nn.Sequential(nn.Conv2d(n_feats, n_feats, kernel_size=5, stride=1, padding=5//2),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.reduce = nn.Sequential(nn.Conv2d(n_feats * 3, n_feats, kernel_size=1),
                                 nn.LeakyReLU(negative_slope=alpha))
        self.CBAM = CBAMBlock(channel=n_feats,reduction=n_feats//4)

    def forward(self, x):
        out1 = self.k1(x)
        out2 = self.k2(x)
        out3 = self.k3(x)
        cat_out = torch.cat([out1, out2, out3], dim=1)
        inception_out = self.reduce(cat_out)
        out = self.CBAM(inception_out)
        return x + out
    

class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()

        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = int(in_channels - self.distilled_channels)

        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.local4 = MKAModule(self.distilled_channels)

        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.local3 = MKAModule(self.distilled_channels)
        
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.local2 = MKAModule(self.distilled_channels)
        
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.local1 = MKAModule(self.distilled_channels)

        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(in_channels, in_channels, 1)
        self.cbam = CBAMBlock(self.distilled_channels * 4)

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
    

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
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


class IMDMKET(nn.Module):
    def __init__(self, args, num_modules=6):
        super(IMDMKET, self).__init__()
        feat = args.n_feats
        scale = args.scale[0]

        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, feat * 4, kernel_size=3, stride=1, padding=1), 
            nn.PReLU(),
            # nn.Conv2d(feat * 4, feat, kernel_size=3, stride=1, padding=1), 
            nn.Conv2d(feat * 4, feat, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(feat, feat, kernel_size=3, stride=1, padding=1), 
            nn.PReLU()
        )

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=feat)
        self.local_c1 = nn.Sequential(nn.Conv2d(feat * 2, feat, kernel_size=1), nn.PReLU())
        self.IMDB2 = IMDModule(in_channels=feat)
        self.local_c2 = nn.Sequential(nn.Conv2d(feat * 3, feat, kernel_size=1), nn.PReLU())
        self.IMDB3 = IMDModule(in_channels=feat)
        self.local_c3 = nn.Sequential(nn.Conv2d(feat * 4, feat, kernel_size=1), nn.PReLU())
        self.IMDB4 = IMDModule(in_channels=feat)
        self.local_c4 = nn.Sequential(nn.Conv2d(feat * 5, feat, kernel_size=1), nn.PReLU())
        self.IMDB5 = IMDModule(in_channels=feat)
        self.local_c5 = nn.Sequential(nn.Conv2d(feat * 6, feat, kernel_size=1), nn.PReLU())
        self.IMDB6 = IMDModule(in_channels=feat)
        self.c = conv_block(feat * num_modules, feat, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(feat, feat, kernel_size=3)

        
        self.LR_conv1 = conv_layer(feat, feat, kernel_size=3)
        self.LR_conv2 = conv_layer(32, feat, kernel_size=3)

        self.reduce = nn.Sequential(
            conv_block(feat, 32, kernel_size=1, act_type='lrelu'),
            conv_layer(32, 32, kernel_size=3))
        # Transformer
        self.attention = MLABlock(dim=32 * 3 * 3) 

        self.tail = nn.Sequential(
            Upsampler(conv_layer, scale, feat, act=False),
            conv_layer(feat, args.n_colors, 3)
        )


    def forward(self, input):
        out_fea = self.head(input)
        pre = [out_fea]
        out_B1 = self.IMDB1(out_fea)
        pre.append(out_B1)
        out_B2 = self.IMDB2(self.local_c1(torch.cat(pre, dim=1))) #8256
        pre.append(out_B2)
        out_B3 = self.IMDB3(self.local_c2(torch.cat(pre, dim=1))) #12352
        pre.append(out_B3)
        out_B4 = self.IMDB4(self.local_c3(torch.cat(pre, dim=1))) #16448
        pre.append(out_B4)
        out_B5 = self.IMDB5(self.local_c4(torch.cat(pre, dim=1))) #24576
        pre.append(out_B5)
        out_B6 = self.IMDB6(self.local_c5(torch.cat(pre, dim=1))) #28672

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))

        # b,c,h,w = out_B.shape
        out = self.attention(self.reduce(self.LR_conv1(out_B)))

        out_lr = self.LR_conv2(out) + out_fea

        # out_lr = self.LR_conv(out_B) + out_fea
        output = self.tail(out_lr)
        
        return output
