import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self, mask_weight=1.0, diffusion_weight=1.0, kernel_weight=0.1):
        super().__init__()
        self.mask_weight = mask_weight
        self.diffusion_weight = diffusion_weight
        self.kernel_weight = kernel_weight
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, predicted_noise, noise, predicted_mask, gt_mask, reconstructed=None, gt=None):
        diffusion_loss = self.mse(predicted_noise, noise)
        mask_loss = self.bce(predicted_mask, gt_mask)
        
        total_loss = self.diffusion_weight * diffusion_loss + self.mask_weight * mask_loss
        
        if reconstructed is not None and gt is not None:
            recon_loss = self.mse(reconstructed, gt)
            total_loss += recon_loss
        
        return total_loss, {
            'diffusion': diffusion_loss.item(),
            'mask': mask_loss.item(),
            'total': total_loss.item()
        }


class KernelGuidedLoss(nn.Module):
    def __init__(self, initial_sigma=1.0, axes=['axial', 'coronal', 'sagittal']):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor([initial_sigma] * len(axes))))
        self.axes = axes
        self.axis_map = {'axial': 0, 'coronal': 1, 'sagittal': 2}

    def _compute_axis_loss(self, x, axis, sigma):
        x_perm = x.permute(axis + 2, 0, 1, *[i for i in range(2, 5) if i != axis + 2])
        n_slices = x_perm.shape[0]
        
        if n_slices < 2:
            return torch.tensor(0.0, device=x.device)
        
        loss = 0.0
        for i in range(n_slices - 1):
            diff = x_perm[i] - x_perm[i + 1]
            squared_dist = torch.sum(diff ** 2, dim=[1, 2, 3])
            kernel_val = torch.exp(-squared_dist / (2 * sigma ** 2))
            loss += -torch.mean(kernel_val)
        
        return loss / (n_slices - 1)
    
    def forward(self, x):
        total_loss = 0.0
        
        for i, axis_name in enumerate(self.axes):
            axis = self.axis_map[axis_name]
            sigma = torch.exp(self.log_sigma[i])
            loss = self._compute_axis_loss(x, axis, sigma)
            total_loss += loss
        
        return total_loss / len(self.axes)


class MultiScaleKernelLoss(nn.Module):
    def __init__(self, scales=[1, 2, 4], sigma=1.0, axes=['axial', 'coronal', 'sagittal']):
        super().__init__()
        self.scales = scales
        self.kernel_losses = nn.ModuleList([
            KernelGuidedLoss(initial_sigma=sigma * scale, axes=axes)
            for scale in scales
        ])
    
    def forward(self, x):
        total_loss = 0.0
        
        for scale, kernel_loss in zip(self.scales, self.kernel_losses):
            x_scaled = F.avg_pool3d(x, kernel_size=scale, stride=scale) if scale > 1 else x
            loss = kernel_loss(x_scaled)
            total_loss += loss
        
        return total_loss / len(self.scales)