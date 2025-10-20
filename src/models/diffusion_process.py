import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionProcess(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, schedule='linear'):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == 'cosine':
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('posterior_variance', betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, model, x, t, corrupted, mask):
        betas_t = self.betas[t].reshape(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].reshape(-1, 1, 1, 1, 1)
        
        model_output = model(x, t, corrupted, mask)
        
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t)
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, corrupted, mask, device, num_inference_steps=None):
        x = torch.randn(shape, device=device)
        
        if num_inference_steps is None:
            num_inference_steps = self.num_timesteps
        
        if num_inference_steps < self.num_timesteps:
            step_size = self.num_timesteps // num_inference_steps
            timesteps = list(range(0, self.num_timesteps, step_size))[:num_inference_steps]
            timesteps = sorted(timesteps, reverse=True)
        else:
            timesteps = list(reversed(range(self.num_timesteps)))
        
        for i in timesteps:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, corrupted, mask)
        
        return x
