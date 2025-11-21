import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SSIM3D(nn.Module):
    def __init__(self, window_size=11, channel=1, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self.create_window_3d(window_size, channel)
    
    def gaussian_window(self, size, sigma=1.5):
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g
    
    def create_window_3d(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size)
        _2D_window = _1D_window.unsqueeze(0) * _1D_window.unsqueeze(1)
        _3D_window = _2D_window.unsqueeze(0) * _1D_window.unsqueeze(1).unsqueeze(1)
        window = _3D_window.unsqueeze(0).unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        channel = img1.size(1)
        
        if self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = self.create_window_3d(self.window_size, channel).to(img1.device).type(img1.dtype)
        
        return self._ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class PSNR3D(nn.Module):
    def __init__(self, max_val=1.0):
        super(PSNR3D, self).__init__()
        self.max_val = max_val
    
    def forward(self, img1, img2):
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return torch.tensor(float('inf'))
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse))
        return psnr


def compute_metrics(pred, target, mask=None):
    ssim_metric = SSIM3D(channel=pred.size(1))
    psnr_metric = PSNR3D()
    
    with torch.no_grad():
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
            ssim_val = ssim_metric(pred_masked, target_masked)
            psnr_val = psnr_metric(pred_masked, target_masked)
        else:
            ssim_val = ssim_metric(pred, target)
            psnr_val = psnr_metric(pred, target)
    
    return {
        'ssim': ssim_val.item(),
        'psnr': psnr_val.item()
    }
