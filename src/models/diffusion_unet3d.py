import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim, emb_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, t_emb):
        return self.mlp(t_emb)


class ResBlock3DWithTime(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
    
    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        t = self.time_mlp(t_emb)
        h = h + t[:, :, None, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    def __init__(self, channels, num_heads=4, patch_size=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.patch_size = patch_size
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        
        qkv = self.qkv(h)
        
        pd = (self.patch_size - D % self.patch_size) % self.patch_size
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        
        if pd > 0 or ph > 0 or pw > 0:
            qkv = F.pad(qkv, (0, pw, 0, ph, 0, pd))
        
        _, _, D_pad, H_pad, W_pad = qkv.shape
        
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, 
                         D_pad // self.patch_size, self.patch_size,
                         H_pad // self.patch_size, self.patch_size,
                         W_pad // self.patch_size, self.patch_size)
        qkv = qkv.permute(0, 4, 6, 8, 1, 2, 5, 7, 9, 3)
        qkv = qkv.reshape(-1, 3, self.num_heads, self.patch_size ** 3, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        
        num_patches_d = D_pad // self.patch_size
        num_patches_h = H_pad // self.patch_size
        num_patches_w = W_pad // self.patch_size
        
        out = out.reshape(B, num_patches_d, num_patches_h, num_patches_w, 
                         self.num_heads, self.patch_size, self.patch_size, self.patch_size, self.head_dim)
        out = out.permute(0, 4, 8, 1, 5, 2, 6, 3, 7)
        out = out.reshape(B, C, D_pad, H_pad, W_pad)
        
        if pd > 0 or ph > 0 or pw > 0:
            out = out[:, :, :D, :H, :W]
        
        out = self.proj(out)
        
        return x + out


class CrossAttention3D(nn.Module):
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.kv = nn.Linear(context_dim, channels * 2)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
    
    def forward(self, x, context):
        B, C, D, H, W = x.shape
        
        h = self.norm(x)
        
        q = self.q(h)
        q = q.reshape(B, self.num_heads, self.head_dim, D * H * W).permute(0, 1, 3, 2)
        
        kv = self.kv(context)
        kv = kv.reshape(B, 2, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        k, v = kv[0], kv[1]
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, D, H, W)
        
        out = self.proj(out)
        
        return x + out


class DownBlock3DWithAttn(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attn=False, num_heads=4):
        super().__init__()
        
        self.res_block1 = ResBlock3DWithTime(in_channels, out_channels, time_emb_dim)
        self.res_block2 = ResBlock3DWithTime(out_channels, out_channels, time_emb_dim)
        
        self.attn = SelfAttention3D(out_channels, num_heads) if use_attn else None
        self.downsample = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, t_emb):
        x = self.res_block1(x, t_emb)
        x = self.res_block2(x, t_emb)
        
        if self.attn is not None:
            x = self.attn(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock3DWithAttn(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, use_attn=False, num_heads=4):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        self.res_block1 = ResBlock3DWithTime(in_channels + out_channels, out_channels, time_emb_dim)
        self.res_block2 = ResBlock3DWithTime(out_channels, out_channels, time_emb_dim)
        
        self.attn = SelfAttention3D(out_channels, num_heads) if use_attn else None
    
    def forward(self, x, skip, t_emb):
        x = self.upsample(x)
        
        if x.shape[2:] != skip.shape[2:]:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            x = torch.nn.functional.pad(x, [diff_w//2, diff_w - diff_w//2,
                                            diff_h//2, diff_h - diff_h//2,
                                            diff_d//2, diff_d - diff_d//2])
        
        x = torch.cat([x, skip], dim=1)
        
        x = self.res_block1(x, t_emb)
        x = self.res_block2(x, t_emb)
        
        if self.attn is not None:
            x = self.attn(x)
        
        return x


class DiffusionUNet3D(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        model_channels=128,
        num_res_blocks=2,
        attention_levels=[1, 2],
        channel_mult=(1, 2, 4, 8),
        num_heads=4,
        time_emb_dim=512,
        context_dim=256,
        dropout=0.1,
        vae_n_hiddens=128,
        vae_downsample=(4, 4, 4)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.mask_embed = nn.Sequential(
            nn.Linear(1, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim)
        )
        
        self.mask_spatial_conv = nn.Conv3d(1, model_channels, kernel_size=3, padding=1)
        
        self.conv_in = nn.Conv3d(in_channels, model_channels, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        chs = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            use_attn = level in attention_levels
            
            self.down_blocks.append(
                DownBlock3DWithAttn(ch, out_ch, time_emb_dim, use_attn, num_heads)
            )
            ch = out_ch
            chs.append(ch)
        
        self.mid_block1 = ResBlock3DWithTime(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SelfAttention3D(ch, num_heads)
        self.mid_cross_attn = CrossAttention3D(ch, context_dim, num_heads)
        self.mid_block2 = ResBlock3DWithTime(ch, ch, time_emb_dim, dropout=dropout)
        
        self.up_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            skip_ch = chs.pop()
            use_attn = level in attention_levels
            
            self.up_blocks.append(
                UpBlock3DWithAttn(ch, out_ch, time_emb_dim, use_attn, num_heads)
            )
            ch = out_ch
        
        self.norm_out = nn.GroupNorm(8, model_channels)
        self.conv_out = nn.Conv3d(model_channels, out_channels, kernel_size=3, padding=1)
        
        self.act = nn.GELU()

        max_ds = max([int(math.log2(d)) for d in vae_downsample])
        skip_ch_list = [vae_n_hiddens * (2 ** i) for i in range(max_ds)]
        skip_ch_list = skip_ch_list + [skip_ch_list[-1]]
        self.skip_channels = skip_ch_list

        self.heads = nn.ModuleDict()
        def make_head(in_ch, out_ch, num_layers=2):
            layers = []
            hid = max(in_ch, out_ch)
            if num_layers >= 1:
                layers.append(nn.Conv3d(in_ch, hid, kernel_size=3, padding=1))
                layers.append(nn.GELU())
            if num_layers >= 2:
                layers.append(nn.Conv3d(hid, out_ch, kernel_size=3, padding=1))
            else:
                layers.append(nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1))
            return nn.Sequential(*layers)

        self.heads['z'] = make_head(model_channels, in_channels, num_layers=2)

        for i, sc in enumerate(self.skip_channels):
            self.heads[f'skip_{i}'] = make_head(model_channels, sc, num_layers=2)
    
    def forward(self, z_t: torch.Tensor, timesteps: torch.Tensor, mask_value: Optional[torch.Tensor]=None, m_mask_spatial: Optional[torch.Tensor]=None, return_dict: bool=False, skip_target_shapes: Optional[List[Tuple[int,int,int]]]=None):

        t_emb = self.time_embed(timesteps)
        
        if mask_value is not None:
            mask_emb = self.mask_embed(mask_value)
        else:
            mask_emb = torch.zeros(z_t.shape[0], 256, device=z_t.device)
        
        h = self.conv_in(z_t)
        
        if m_mask_spatial is not None:
            m_mask_down = F.interpolate(m_mask_spatial, size=h.shape[2:], mode='trilinear', align_corners=False)
            mask_feat = self.mask_spatial_conv(m_mask_down)
            h = h + mask_feat
        
        skips = []
        for down_block in self.down_blocks:
            h, skip = down_block(h, t_emb)
            skips.append(skip)
        
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_cross_attn(h, mask_emb)
        h = self.mid_block2(h, t_emb)
        
        for up_block in self.up_blocks:
            skip = skips.pop()
            h = up_block(h, skip, t_emb)

        shared = self.norm_out(h)
        shared = self.act(shared)

        legacy_out = self.conv_out(shared)

        if not return_dict:
            return legacy_out

        preds = {}

        z_pred = self.heads['z'](shared)
        if z_pred.shape[2:] != z_t.shape[2:]:
            z_pred = F.interpolate(z_pred, size=z_t.shape[2:], mode='trilinear', align_corners=False)
        preds['z'] = z_pred

        skip_preds = []
        for i in range(len(self.skip_channels)):
            key = f'skip_{i}'
            out = self.heads[key](shared)
            if skip_target_shapes is not None and i < len(skip_target_shapes):
                target_shape = skip_target_shapes[i]
                if out.shape[2:] != target_shape:
                    out = F.interpolate(out, size=target_shape, mode='trilinear', align_corners=False)
            skip_preds.append(out)

        preds['skips'] = skip_preds

        preds['legacy'] = legacy_out

        return preds


def downsample_mask_maxpool(mask: torch.Tensor, target_shape: Tuple[int,int,int]) -> torch.Tensor:
    if mask.dim() == 4:
        mask = mask.unsqueeze(1)
    mask_ds = F.adaptive_max_pool3d(mask, output_size=target_shape)
    return (mask_ds > 0.5).float()


def per_scale_mse_loss(eps_target: torch.Tensor, eps_pred: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    err = (eps_target - eps_pred) ** 2
    if mask is not None:
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        err = err * mask
        return err.sum() / (mask.sum().clamp_min(1.0) + 1e-8)
    else:
        return err.mean()


def denoise_to_x0(noisy: torch.Tensor, eps_pred: torch.Tensor, sqrt_alpha_bar_t: float, sqrt_one_minus_alpha_bar_t: float) -> torch.Tensor:
    return (noisy - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    unet = DiffusionUNet3D(
        in_channels=256,
        out_channels=256,
        model_channels=64,
        channel_mult=(1, 2, 4),
        attention_levels=[1, 2],
        num_heads=4,
        vae_n_hiddens=128,
        vae_downsample=(4, 4, 4)
    ).to(device)
    
    batch_size = 1
    D, H, W = 16, 16, 16
    z_t = torch.randn(batch_size, 256, D, H, W).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    mask_value = torch.rand(batch_size, 1).to(device)
    
    print(f"Input shape: {z_t.shape}")
    print(f"Timesteps: {timesteps}")
    
    legacy_out = unet(z_t, timesteps, mask_value, return_dict=False)
    print(f"Legacy output shape: {legacy_out.shape}")
    
    preds = unet(z_t, timesteps, mask_value, return_dict=True)
    print(f"\nMulti-head outputs:")
    print(f"  z pred shape: {preds['z'].shape}")
    print(f"  Number of skip heads: {len(preds['skips'])}")
    for i, skip_pred in enumerate(preds['skips']):
        print(f"  skip_{i} pred shape: {skip_pred.shape} (channels: {unet.skip_channels[i]})")
    
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal parameters: {total_params:,}")
