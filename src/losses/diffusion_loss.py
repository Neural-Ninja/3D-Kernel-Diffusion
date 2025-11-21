import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='mse', weight_corrupted_regions=2.0, lambda_x0=0.0):
        super().__init__()
        self.loss_type = loss_type
        self.weight_corrupted_regions = weight_corrupted_regions
        self.lambda_x0 = lambda_x0
    
    def forward(self, noise_pred, noise_target, z_0_pred=None, z_0_target=None, mask=None):
        if self.loss_type == 'mse':
            if mask is not None:
                weight = 1.0 + (self.weight_corrupted_regions - 1.0) * mask
                noise_loss = (weight * (noise_pred - noise_target) ** 2).mean()
            else:
                noise_loss = F.mse_loss(noise_pred, noise_target)
        elif self.loss_type == 'l1':
            if mask is not None:
                weight = 1.0 + (self.weight_corrupted_regions - 1.0) * mask
                noise_loss = (weight * torch.abs(noise_pred - noise_target)).mean()
            else:
                noise_loss = F.l1_loss(noise_pred, noise_target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        total_loss = noise_loss
        
        x0_loss = torch.tensor(0.0, device=noise_loss.device)
        
        return {
            'total': total_loss,
            'noise': noise_loss,
            'x0': x0_loss
        }
