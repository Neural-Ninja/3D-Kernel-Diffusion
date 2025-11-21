import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from vae3d import VQVAE3D
from diffusion_unet3d import DiffusionUNet3D
from diffusion3d import GaussianDiffusion
from losses.diffusion_loss import DiffusionLoss


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        vae_in_channels=1,
        vae_out_channels=1,
        vae_n_hiddens=128,
        vae_downsample=(4, 4, 4),
        vae_num_groups=32,
        vae_embedding_dim=256,
        vae_n_codes=512,
        unet_model_channels=128,
        unet_channel_mult=(1, 2, 4),
        unet_attention_levels=[1, 2],
        unet_num_heads=4,
        unet_time_emb_dim=512,
        unet_context_dim=256,
        diffusion_timesteps=1000,
        diffusion_beta_schedule='cosine',
        vae_checkpoint=None,
        freeze_vae=True,
        device='cuda'
    ):
        super().__init__()
        self.device = torch.device(device if isinstance(device, str) else device)
        self.vae_embedding_dim = vae_embedding_dim
        self.freeze_vae = freeze_vae
        self.downsample_factor = vae_downsample

        self.vae = VQVAE3D(
            in_channels=vae_in_channels,
            out_channels=vae_out_channels,
            n_hiddens=vae_n_hiddens,
            downsample=vae_downsample,
            num_groups=vae_num_groups,
            embedding_dim=vae_embedding_dim,
            n_codes=vae_n_codes
        )

        if vae_checkpoint is not None:
            checkpoint = torch.load(vae_checkpoint, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.vae.load_state_dict(state_dict)
        
        self.vae.to(self.device)

        if freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        self.register_buffer('latent_scale', torch.tensor(1.0))
        self.register_buffer('skip_scales', torch.ones(3))
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, vae_in_channels, 64, 64, 64).to(self.device)
            z_dummy, _, skips_dummy = self.vae.encode(dummy_input)

            self._z_channels = z_dummy.shape[1]
            self._z_spatial = z_dummy.shape[2:]

            self._skip_channels = [s.shape[1] for s in skips_dummy]
            self._skip_orig_shapes = [s.shape[2:] for s in skips_dummy]

            total_raw_channels = self._z_channels + sum(self._skip_channels)

        self.unet = DiffusionUNet3D(
            in_channels=self._z_channels,
            out_channels=self._z_channels,
            model_channels=unet_model_channels,
            channel_mult=unet_channel_mult,
            attention_levels=unet_attention_levels,
            num_heads=unet_num_heads,
            time_emb_dim=unet_time_emb_dim,
            context_dim=unet_context_dim,
            vae_n_hiddens=vae_n_hiddens,
            vae_downsample=vae_downsample
        )

        self.unet.to(self.device)

        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_timesteps,
            beta_schedule=diffusion_beta_schedule,
            device=self.device
        )

        self.diffusion_loss_fn = DiffusionLoss(loss_type='mse', weight_corrupted_regions=2.0, lambda_x0=0.0)

    def downsample_mask(self, mask):
        D, H, W = self.downsample_factor
        downsampled = F.avg_pool3d(mask.float(), kernel_size=(D, H, W), stride=(D, H, W))
        return (downsampled > 0.5).float()

    def normalize_latents(self, z, skips):
        """Normalize latents to unit variance for stable diffusion training"""
        z_normalized = z / self.latent_scale
        skips_normalized = [s / self.skip_scales[i] for i, s in enumerate(skips)]
        return z_normalized, skips_normalized
    
    def denormalize_latents(self, z, skips):
        """Denormalize latents back to original scale"""
        z_denorm = z * self.latent_scale
        skips_denorm = [s * self.skip_scales[i] for i, s in enumerate(skips)]
        return z_denorm, skips_denorm
    
    def encode(self, x, normalize=True):
        if self.freeze_vae:
            with torch.no_grad():
                z, vq_output, skips = self.vae.encode(x)
        else:
            z, vq_output, skips = self.vae.encode(x)
        
        if normalize:
            z, skips = self.normalize_latents(z, skips)
        
        return z, vq_output, skips

    def decode(self, z, skips=None, denormalize=True):
        if denormalize and skips is not None:
            z, skips = self.denormalize_latents(z, skips)
        
        if self.freeze_vae:
            with torch.no_grad():
                return self.vae.decode(z, skips)
        else:
            return self.vae.decode(z, skips)

    def forward_diffusion(self, latent, t):
        noise = torch.randn_like(latent)
        z_t = self.diffusion.q_sample(latent, t, noise)
        return z_t, noise

    def predict_noise(self, latent_combined, t, context=None):
        return self.unet(latent_combined, t, context)

    def q_sample_multi(self, targets, t):
        noises = [torch.randn_like(tgt) for tgt in targets]
        noisy_targets = []
        for tgt, noise in zip(targets, noises):
            tgt_t = self.diffusion.q_sample(tgt, t, noise)
            noisy_targets.append(tgt_t)
        return noisy_targets, noises

    def post_process(self, x_pred, x_corr, m_mask):
        if x_pred.shape != x_corr.shape:
            x_pred = F.interpolate(x_pred, size=x_corr.shape[2:], mode='trilinear', align_corners=False)
        if m_mask.shape != x_pred.shape:
            m_mask = F.interpolate(m_mask, size=x_pred.shape[2:], mode='nearest')
        return m_mask * x_pred + (1 - m_mask) * x_corr

    @torch.no_grad()
    def reconstruct(self, x_corr, m_mask, use_ddim=True, ddim_steps=50):
        from diffusion_unet3d import downsample_mask_maxpool, denoise_to_x0
        
        z_corr, _, skips_corr = self.encode(x_corr, normalize=True)
        
        t_start = int(0.5 * self.diffusion.timesteps)
        
        z_noise = torch.randn_like(z_corr)
        skip_noises = [torch.randn_like(s) for s in skips_corr]
        
        sqrt_alpha = self.diffusion.sqrt_alphas_cumprod[t_start]
        sqrt_one_minus_alpha = self.diffusion.sqrt_one_minus_alphas_cumprod[t_start]
        
        z_t = sqrt_alpha * z_corr + sqrt_one_minus_alpha * z_noise
        skip_t_list = [sqrt_alpha * s + sqrt_one_minus_alpha * n for s, n in zip(skips_corr, skip_noises)]
        
        m_mask_z = downsample_mask_maxpool(m_mask, z_corr.shape[2:])
        m_mask_skips = [downsample_mask_maxpool(m_mask, s.shape[2:]) for s in skips_corr]
        
        z_t = z_corr * (1 - m_mask_z) + z_t * m_mask_z
        skip_t_list = [s_corr * (1 - m_s) + s_t * m_s for s_corr, s_t, m_s in zip(skips_corr, skip_t_list, m_mask_skips)]
        
        mask_value = m_mask.reshape(x_corr.shape[0], -1).mean(dim=1, keepdim=True)
        
        if use_ddim:
            timesteps = torch.linspace(t_start, 0, ddim_steps, device=z_corr.device).long()
            
            for i in range(len(timesteps) - 1):
                t_curr = int(timesteps[i].item())
                t_next = int(timesteps[i + 1].item())
                t_batch = torch.full((x_corr.shape[0],), t_curr, device=z_corr.device, dtype=torch.long)
                
                preds = self.unet(z_t, t_batch, mask_value, m_mask, return_dict=True, skip_target_shapes=[s.shape[2:] for s in skips_corr])
                
                eps_z_pred = preds['z']
                eps_skip_preds = preds['skips']
                
                alpha_bar_curr = self.diffusion.alphas_cumprod[t_curr]
                alpha_bar_next = self.diffusion.alphas_cumprod[t_next]
                
                sqrt_alpha_curr = torch.sqrt(alpha_bar_curr)
                sqrt_one_minus_alpha_curr = torch.sqrt(1 - alpha_bar_curr)
                
                z_0_pred = denoise_to_x0(z_t, eps_z_pred, sqrt_alpha_curr, sqrt_one_minus_alpha_curr)
                skip_0_preds = [denoise_to_x0(s_t, eps_s, sqrt_alpha_curr, sqrt_one_minus_alpha_curr) 
                               for s_t, eps_s in zip(skip_t_list, eps_skip_preds)]
                
                z_t = torch.sqrt(alpha_bar_next) * z_0_pred + torch.sqrt(1 - alpha_bar_next) * eps_z_pred
                skip_t_list = [torch.sqrt(alpha_bar_next) * s0 + torch.sqrt(1 - alpha_bar_next) * eps_s 
                              for s0, eps_s in zip(skip_0_preds, eps_skip_preds)]
        else:
            for t in reversed(range(t_start)):
                t_batch = torch.full((x_corr.shape[0],), t, device=z_corr.device, dtype=torch.long)
                
                preds = self.unet(z_t, t_batch, mask_value, m_mask, return_dict=True, skip_target_shapes=[s.shape[2:] for s in skips_corr])
                
                eps_z_pred = preds['z']
                eps_skip_preds = preds['skips']
                
                alpha_bar = self.diffusion.alphas_cumprod[t]
                sqrt_alpha = torch.sqrt(alpha_bar)
                sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar)
                
                z_t = denoise_to_x0(z_t, eps_z_pred, sqrt_alpha, sqrt_one_minus_alpha)
                skip_t_list = [denoise_to_x0(s_t, eps_s, sqrt_alpha, sqrt_one_minus_alpha) 
                              for s_t, eps_s in zip(skip_t_list, eps_skip_preds)]
        
        x_pred = self.decode(z_t, skip_t_list, denormalize=True)
        return x_pred

    def compute_loss(self, x_gt, x_corr, m_mask):
        from diffusion_unet3d import downsample_mask_maxpool, per_scale_mse_loss, denoise_to_x0
        
        batch_size = x_gt.shape[0]
        device = x_gt.device

        z_gt_raw, _, skips_gt_raw = self.encode(x_gt, normalize=False)
        
        with torch.no_grad():
            z_std = z_gt_raw.std().clamp(min=1e-6)
            self.latent_scale.copy_(0.99 * self.latent_scale + 0.01 * z_std)
            
            for i, s in enumerate(skips_gt_raw):
                s_std = s.std().clamp(min=1e-6)
                self.skip_scales[i] = 0.99 * self.skip_scales[i] + 0.01 * s_std
        
        z_gt, skips_gt = self.normalize_latents(z_gt_raw, skips_gt_raw)
        z_corr, _, skips_corr = self.encode(x_corr, normalize=True)
        
        m_mask_z = downsample_mask_maxpool(m_mask, z_gt.shape[2:])
        m_mask_skips = [downsample_mask_maxpool(m_mask, s.shape[2:]) for s in skips_gt]
        
        mask_value = m_mask.reshape(batch_size, -1).mean(dim=1, keepdim=True)
        
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=device).long()
        
        z_t, eps_z = self.forward_diffusion(z_gt, t)
        skip_t_list, eps_skip_list = self.q_sample_multi(skips_gt, t)
        
        preds = self.unet(z_t, t, mask_value, m_mask, return_dict=True, skip_target_shapes=[s.shape[2:] for s in skips_gt])
        
        eps_z_pred = preds['z']
        eps_skip_preds = preds['skips']
        
        L_z = per_scale_mse_loss(eps_z, eps_z_pred, m_mask_z)
        
        L_skips = [per_scale_mse_loss(eps_skip_list[i], eps_skip_preds[i], m_mask_skips[i]) 
                   for i in range(len(skips_gt))]
        
        w_z = 1.0
        w_skips = [0.5, 0.4, 0.25][:len(skips_gt)]
        
        L_diff = w_z * L_z + sum(w * L_s for w, L_s in zip(w_skips, L_skips))
        
        alpha_bar_t = self.diffusion.alphas_cumprod[t[0]]
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        
        z_hat = denoise_to_x0(z_t, eps_z_pred, sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t)
        skip_hat = [denoise_to_x0(skip_t_list[i], eps_skip_preds[i], sqrt_alpha_bar_t, sqrt_one_minus_alpha_bar_t) 
                    for i in range(len(skips_gt))]
        
        x_rec = self.decode(z_hat, skip_hat, denormalize=True)
        L_rec = F.l1_loss(x_rec, x_gt)
        
        lambda_rec = 0.1
        total_loss = L_diff + lambda_rec * L_rec
        
        return {
            'total_loss': total_loss,
            'diffusion_loss': L_diff,
            'x0_loss': torch.tensor(0.0, device=device),
            'vq_loss': torch.tensor(0.0, device=device),
            'mse_loss': L_z,
            'ssim_loss': torch.tensor(0.0, device=device),
            'perceptual_loss': L_rec
        }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LatentDiffusionModel(
        vae_in_channels=1,
        vae_out_channels=1,
        vae_n_hiddens=128,
        vae_downsample=(4, 4, 4),
        vae_num_groups=32,
        vae_embedding_dim=512,
        vae_n_codes=768,
        unet_model_channels=64,
        unet_channel_mult=(1, 2, 4),
        diffusion_timesteps=1000,
        diffusion_beta_schedule='cosine',
        device=device
    ).to(device)

    batch_size = 1
    D, H, W = 64, 64, 64
    x_gt = torch.randn(batch_size, 1, D, H, W).to(device)
    x_corr = x_gt.clone()
    m_mask = (torch.rand(batch_size, 1, D, H, W) > 0.8).float().to(device)

    z, vq_output, skips = model.encode(x_gt)
    print("z shape:", z.shape)
    for i, s in enumerate(skips):
        print(f"skip[{i}] shape:", s.shape)

    loss_dict = model.compute_loss(x_gt, x_corr, m_mask)
    print("Loss keys:", loss_dict.keys())

    recon = model.reconstruct(x_corr, m_mask, use_ddim=True, ddim_steps=5)
    print("reconstruct shape:", recon.shape)
    total_params = sum(p.numel() for p in model.parameters())
    vae_params = sum(p.numel() for p in model.vae.parameters())
    unet_params = sum(p.numel() for p in model.unet.parameters())
    print(f"Model parameters - Total: {total_params:,}, VAE: {vae_params:,}, U-Net: {unet_params:,}")
