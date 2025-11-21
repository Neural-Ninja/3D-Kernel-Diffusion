import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class GaussianDiffusion:
    def __init__(
        self,
        timesteps=1000,
        beta_schedule='cosine',
        beta_start=0.0001,
        beta_end=0.02,
        device='cuda'
    ):
        self.timesteps = timesteps
        self.device = device
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    
    def p_mean_variance(self, model, x_t, t, mask_value=None, clip_denoised=True):
        noise_pred = model(x_t, t, mask_value)
        
        x_start = self.predict_start_from_noise(x_t, t, noise_pred)
        
        if clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start, x_t, t
        )
        
        return model_mean, posterior_variance, posterior_log_variance, x_start
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, mask_value=None, clip_denoised=True):
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            model, x_t, t, mask_value, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        model_log_variance = torch.clamp(model_log_variance, min=-20, max=2)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, mask_value=None, return_intermediates=False):
        device = self.device
        batch_size = shape[0]
        
        x = torch.randn(shape, device=device)
        
        intermediates = [x] if return_intermediates else None
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='DDPM Sampling', total=self.timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, mask_value)
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model,
        shape,
        mask_value=None,
        ddim_timesteps=50,
        ddim_eta=0.0,
        return_intermediates=False
    ):
        device = self.device
        batch_size = shape[0]
        
        c = self.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        ddim_timestep_seq = np.clip(ddim_timestep_seq + 1, 0, self.timesteps - 1)
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        x = torch.randn(shape, device=device)
        
        intermediates = [x] if return_intermediates else None
        
        eps = 1e-8
        
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            
            noise_pred = model(x, t, mask_value)
            
            if torch.isnan(noise_pred).any():
                noise_pred = torch.nan_to_num(noise_pred, nan=0.0)
            
            alpha_cumprod_t = self._extract(self.alphas_cumprod, t, x.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, prev_t, x.shape)
            
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t + eps) * noise_pred) / torch.sqrt(alpha_cumprod_t + eps)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_t_prev - ddim_eta ** 2, min=0.0) + eps) * noise_pred
            
            noise = torch.randn_like(x) if ddim_eta > 0 else 0
            
            x = torch.sqrt(alpha_cumprod_t_prev + eps) * pred_x0 + dir_xt + ddim_eta * noise
            
            x = torch.clamp(x, -10.0, 10.0)
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def training_losses(self, model, x_start, t, mask_value=None, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        
        noise_pred = model(x_t, t, mask_value)
        
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
        loss = loss.mean(dim=[1, 2, 3, 4])
        
        return loss
    
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_schedule='cosine',
        device=device
    )
    
    batch_size = 2
    x_start = torch.randn(batch_size, 64, 16, 16, 16).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    print(f"x_start shape: {x_start.shape}")
    print(f"Timesteps: {t}")
    
    x_t = diffusion.q_sample(x_start, t)
    print(f"x_t shape: {x_t.shape}")
    
    class MockModel(nn.Module):
        def forward(self, x, t, mask_value=None):
            return torch.randn_like(x)
    
    model = MockModel().to(device)
    
    loss = diffusion.training_losses(model, x_start, t)
    print(f"Loss shape: {loss.shape}")
    print(f"Loss: {loss.mean().item():.4f}")
    
    print("\nDiffusion process initialized successfully!")
