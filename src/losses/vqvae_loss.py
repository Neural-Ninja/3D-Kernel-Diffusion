import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAELoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_ssim=0.0, lambda_perceptual=0.0):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
    
    def forward(self, recon, target, vq_loss):
        mse_loss = F.mse_loss(recon, target)
        
        ssim_loss = torch.tensor(0.0, device=recon.device)
        perceptual_loss = torch.tensor(0.0, device=recon.device)
        
        total_loss = self.lambda_mse * mse_loss + vq_loss
        
        return {
            'total': total_loss,
            'mse': mse_loss,
            'ssim': ssim_loss,
            'perceptual': perceptual_loss,
            'vq': vq_loss
        }
