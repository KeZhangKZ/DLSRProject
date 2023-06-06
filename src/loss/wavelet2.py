from model import common
import torchvision.models.vgg as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                            hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)

    def forward(self, x):
        C = torch.floor_divide(x.shape[1], 4)
        filters = torch.cat([self.weight, ] * C, dim=0)
        y = F.conv_transpose2d(x, filters, groups=C, stride=2)
        return y


def loss_L1(input, output):
    return torch.mean(torch.abs(input-output))    


def loss_Textures(x, y, nc=1, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)


class LossNetwork(torch.nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '13': "relu3",
            '22': "relu4",
            '31': "relu5",
        }
        
    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        
        return output
    

class Wavelet2(nn.Module):
    def __init__(self):
        super(Wavelet2, self).__init__()
        self.dwt2 = DWT()
        self.idwt2 = IWT()
        self.lossnet = LossNetwork().float()

        for p in self.parameters():
            p.requires_grad = False        
        self.requires_grad = False

    def forward(self, sr, hr):

        hr_w = self.dwt2(hr)        
        sr_w = self.dwt2(sr)

        # loss_sr_v = self.lossnet(sr)
        # loss_hr_v = self.lossnet(hr)
        # p0 = loss_L1(sr, hr)*2
        # p1 = loss_L1(loss_sr_v['relu1'], loss_hr_v['relu1'])/2.6
        # p2 = loss_L1(loss_sr_v['relu2'], loss_hr_v['relu2'])/4.8
        # loss_per = p0+p1+p2 


        loss_AVH = F.mse_loss(sr_w[:,0:3,:,:], hr_w[:,0:3,:,:])
        loss_D = F.mse_loss(sr_w[:,3:,:,:], hr_w[:,3:,:,:])
        loss_textures_D = loss_Textures(sr_w[:,3:,:,:], hr_w[:,3:,:,:])
        loss_img = F.mse_loss(sr, hr)
        
        loss = loss_D.mul(1) + loss_AVH.mul(0.01) + loss_img.mul(0.1) + loss_textures_D.mul(1) #  + loss_img.mul(0.1) (1 * loss_per) + 
        # loss_sr.mul(0.99) + loss_lr.mul(0.01) + loss_img.mul(0.1) + loss_textures.mul(1) 
        return loss
