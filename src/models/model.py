import torch
import torch.nn as nn
from .corruption_detector import CorruptionDetector
from .swin_diffusion import SwinDiffusionUNet
from .diffusion_process import DiffusionProcess
from .encoder import PretrainedEncoder


class ReconstructionModel(nn.Module):
    def __init__(self, 
                 detector_channels=64,
                 detector_blocks=4,
                 diffusion_channels=64,
                 diffusion_depths=[2, 2, 2],
                 num_timesteps=1000,
                 beta_schedule='cosine',
                 use_pretrained=False,
                 pretrained_type='swin',
                 freeze_pretrained=True):
        super().__init__()
        
        self.use_pretrained = use_pretrained
        
        self.corruption_detector = CorruptionDetector(
            in_channels=1,
            base_channels=detector_channels,
            num_blocks=detector_blocks
        )
        
        self.diffusion_unet = SwinDiffusionUNet(
            in_channels=3,
            out_channels=1,
            base_channels=diffusion_channels,
            depths=diffusion_depths
        )
        
        self.diffusion = DiffusionProcess(
            num_timesteps=num_timesteps,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=beta_schedule
        )
        
        if use_pretrained:
            self.pretrained_encoder = PretrainedEncoder(
                encoder_type=pretrained_type,
                freeze=freeze_pretrained
            )
            
            encoder_dims = self.pretrained_encoder.feature_dims
            num_levels = len(diffusion_depths)
            self.feature_adapter = nn.ModuleList([
                nn.Conv3d(encoder_dims[i] if i < len(encoder_dims) else encoder_dims[-1], 
                         diffusion_channels * (2**i), 1)
                for i in range(num_levels)
            ])
        else:
            self.pretrained_encoder = None
    
    def forward(self, corrupted, gt=None, timesteps=None):
        predicted_mask = self.corruption_detector(corrupted)
        
        if self.training and gt is not None:
            if timesteps is None:
                timesteps = torch.randint(0, self.diffusion.num_timesteps, 
                                        (corrupted.shape[0],), device=corrupted.device)
            
            noise = torch.randn_like(gt)
            x_noisy = self.diffusion.q_sample(gt, timesteps, noise)
            
            predicted_noise = self.diffusion_unet(x_noisy, timesteps, corrupted, predicted_mask)
            
            return predicted_noise, noise, predicted_mask
        else:
            clean_volume = self.diffusion.p_sample_loop(
                self.diffusion_unet,
                corrupted.shape,
                corrupted,
                predicted_mask,
                corrupted.device
            )
            return clean_volume, predicted_mask
    
    def detect_corruption(self, corrupted):
        return self.corruption_detector(corrupted)
    
    def reconstruct(self, corrupted, mask=None):
        if mask is None:
            mask = self.corruption_detector(corrupted)
        
        reconstructed = self.diffusion.p_sample_loop(
            self.diffusion_unet,
            corrupted.shape,
            corrupted,
            mask,
            corrupted.device
        )
        return reconstructed


def create_model(config):
    return ReconstructionModel(
        detector_channels=config.get('detector_channels', 64),
        detector_blocks=config.get('detector_blocks', 4),
        diffusion_channels=config.get('diffusion_channels', 64),
        diffusion_depths=config.get('diffusion_depths', [2, 2, 2]),
        num_timesteps=config.get('num_timesteps', 1000),
        beta_schedule=config.get('beta_schedule', 'cosine'),
        use_pretrained=config.get('use_pretrained', False),
        pretrained_type=config.get('pretrained_type', 'swin'),
        freeze_pretrained=config.get('freeze_pretrained', True)
    )
