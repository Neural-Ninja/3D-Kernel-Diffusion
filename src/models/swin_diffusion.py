import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        x_flat = x.flatten(2).transpose(1, 2)
        
        shortcut = x_flat
        x_flat = self.norm1(x_flat)
        x_flat, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = shortcut + x_flat
        
        shortcut = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = self.mlp(x_flat)
        x_flat = shortcut + x_flat
        
        return x_flat.transpose(1, 2).reshape(B, C, D, H, W)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_mlp(time_emb)[:, :, None, None, None]
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.shortcut(x))


class SwinConvBlock(nn.Module):
    def __init__(self, channels, time_dim, num_heads=4):
        super().__init__()
        self.conv = ConvBlock(channels, channels, time_dim)
        self.swin = SwinBlock3D(channels, num_heads)
        self.fusion = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x, time_emb):
        conv_out = self.conv(x, time_emb)
        swin_out = self.swin(x)
        return self.fusion(conv_out + swin_out)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_swin=False):
        super().__init__()
        if use_swin:
            self.block = SwinConvBlock(in_channels, time_dim)
        else:
            self.block = ConvBlock(in_channels, in_channels, time_dim)
        self.downsample = nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x, time_emb):
        x = self.block(x, time_emb)
        skip = x
        x = self.downsample(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, use_swin=False):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.fuse = nn.Conv3d(out_channels * 2, out_channels, 1)
        if use_swin:
            self.block = SwinConvBlock(out_channels, time_dim)
        else:
            self.block = ConvBlock(out_channels, out_channels, time_dim)
    
    def forward(self, x, skip, time_emb):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.block(x, time_emb)
        return x


class SwinDiffusionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, 
                 depths=[2, 2, 2], time_dim=128):
        super().__init__()
        
        self.time_emb = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        channels = [base_channels * (2**i) for i in range(len(depths))]
        
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, (ch, depth) in enumerate(zip(channels, depths)):
            use_swin = (i >= len(depths) - 1)
            self.down_blocks.append(DownBlock(in_ch, ch, time_dim, use_swin))
            in_ch = ch
        
        self.bottleneck = nn.Sequential(
            SwinConvBlock(channels[-1], time_dim),
            SwinConvBlock(channels[-1], time_dim)
        )
        
        self.up_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            use_swin = (i >= len(depths) - 1)
            self.up_blocks.append(UpBlock(channels[i], channels[i-1], time_dim, use_swin))
        
        self.final_up = UpBlock(channels[0], base_channels, time_dim, False)
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 1)
        )
    
    def forward(self, x, timesteps, corrupted, mask):
        t_emb = self.time_emb(timesteps)
        
        x_in = torch.cat([x, corrupted, mask], dim=1)
        x = self.input_conv(x_in)
        
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t_emb)
            skips.append(skip)
        
        for block in self.bottleneck:
            x = block(x, t_emb)
        
        for up_block, skip in zip(self.up_blocks, reversed(skips[1:])):
            x = up_block(x, skip, t_emb)
        
        x = self.final_up(x, skips[0], t_emb)
        
        return self.output_conv(x)
