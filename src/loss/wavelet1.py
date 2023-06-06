from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 使用哈尔 haar 小波变换来实现二维离散小波
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height,
                     out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


# 二维离散小波
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


# 逆向二维离散小波
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def loss_MSE(x, y, size_average=False):
#   print(f"x.shape={x.shape}")
#   print(f"y.shape={y.shape}")
  z = x - y 
  z2 = z * z
#   print(f"z2.shape={z2.shape}")
#   print(f"z2.sum().shape={z2.sum().shape}")
#   print(f"z2.sum()={z2.sum()}")
#   print(f"size_average={size_average}")
#   print(f"x.size(0)={x.size(0)}")
  if size_average:
    return z2.mean()
  else:
    return z2.sum().div(x.size(0)*2)
    

def loss_Textures(x, y, nc=1, alpha=1.2, margin=0):
  xi = x.contiguous().view(x.size(0), -1, nc, x.size(2), x.size(3))
  yi = y.contiguous().view(y.size(0), -1, nc, y.size(2), y.size(3))
  
  xi2 = torch.sum(xi * xi, dim=2)
  yi2 = torch.sum(yi * yi, dim=2)
  
  out = nn.functional.relu(yi2.mul(alpha) - xi2 + margin)
  
  return torch.mean(out)


class Wavelet1(nn.Module):
    def __init__(self):
        super(Wavelet1, self).__init__()
        self.dwt2 = DWT()
        self.idwt2 = IWT()

        for p in self.parameters():
            p.requires_grad = False        
        self.requires_grad = False

    def forward(self, sr, hr):

        hr_w = self.dwt2(hr)        
        sr_w = self.dwt2(sr)

        
        loss_A = F.mse_loss(sr_w[:,0:1,:,:], hr_w[:,0:1,:,:])
        loss_VH = F.mse_loss(sr_w[:,1:3,:,:], hr_w[:,1:3,:,:])
        loss_D = F.mse_loss(sr_w[:,3:,:,:], hr_w[:,3:,:,:])
        loss_textures_VH = loss_Textures(sr_w[:,1:3,:,:], hr_w[:,1:3,:,:])
        loss_textures_D = loss_Textures(sr_w[:,3:,:,:], hr_w[:,3:,:,:])
        loss_img = F.mse_loss(sr, hr)
        
        loss = loss_D.mul(0.99) + loss_VH.mul(0.75) + loss_A.mul(0.01) + loss_img.mul(0.1) + loss_textures_VH.mul(0.75) + loss_textures_D.mul(1)

        return loss
