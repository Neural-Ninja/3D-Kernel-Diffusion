import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        device = x.device
        
        pe = torch.zeros(B, C, D, H, W, device=device)
        
        d_pos = torch.arange(D, device=device).float().unsqueeze(1)
        h_pos = torch.arange(H, device=device).float().unsqueeze(1)
        w_pos = torch.arange(W, device=device).float().unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, C, 2, device=device).float() * 
                            -(math.log(10000.0) / C))
        
        for i in range(0, C, 6):
            if i < C:
                pe[:, i, :, :, :] = torch.sin(d_pos * div_term[i//2 % len(div_term)]).view(D, 1, 1)
            if i+1 < C:
                pe[:, i+1, :, :, :] = torch.cos(d_pos * div_term[i//2 % len(div_term)]).view(D, 1, 1)
            if i+2 < C:
                pe[:, i+2, :, :, :] = torch.sin(h_pos * div_term[i//2 % len(div_term)]).view(1, H, 1)
            if i+3 < C:
                pe[:, i+3, :, :, :] = torch.cos(h_pos * div_term[i//2 % len(div_term)]).view(1, H, 1)
            if i+4 < C:
                pe[:, i+4, :, :, :] = torch.sin(w_pos * div_term[i//2 % len(div_term)]).view(1, 1, W)
            if i+5 < C:
                pe[:, i+5, :, :, :] = torch.cos(w_pos * div_term[i//2 % len(div_term)]).view(1, 1, W)
        
        return x + pe


class TransformerBlock3D(nn.Module):
    def __init__(self, channels, num_heads=8, mlp_ratio=4, max_tokens=4096):
        super().__init__()
        self.max_tokens = max_tokens
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * mlp_ratio),
            nn.GELU(),
            nn.Linear(channels * mlp_ratio, channels)
        )
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        num_tokens = D * H * W
        
        if num_tokens > self.max_tokens:
            pool_size = int(math.ceil((num_tokens / self.max_tokens) ** (1/3)))
            x_pooled = F.avg_pool3d(x, kernel_size=pool_size, stride=pool_size)
            B, C, D_p, H_p, W_p = x_pooled.shape
            x_flat = x_pooled.flatten(2).transpose(1, 2)
        else:
            x_flat = x.flatten(2).transpose(1, 2)
            D_p, H_p, W_p = D, H, W
        
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        x_norm = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm)
        x_flat = x_flat + mlp_out
        
        x_out = x_flat.transpose(1, 2).reshape(B, C, D_p, H_p, W_p)
        
        if num_tokens > self.max_tokens:
            x_out = F.interpolate(x_out, size=(D, H, W), mode='trilinear', align_corners=False)
        
        return x_out


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.conv(x)


class HybridBlock(nn.Module):
    def __init__(self, channels, num_heads=8, use_checkpoint=False):
        super().__init__()
        self.conv = ConvBlock3D(channels, channels)
        self.transformer = TransformerBlock3D(channels, num_heads)
        self.fusion = nn.Conv3d(channels * 2, channels, 1)
        self.use_checkpoint = use_checkpoint
        
    def forward(self, x):
        if self.use_checkpoint and self.training:
            conv_out = torch.utils.checkpoint.checkpoint(self.conv, x, use_reentrant=False)
            trans_out = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
        else:
            conv_out = self.conv(x)
            trans_out = self.transformer(x)
        fused = self.fusion(torch.cat([conv_out, trans_out], dim=1))
        return fused + x


class CorruptionDetector(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_blocks=4, use_checkpoint=False):
        super().__init__()
        
        self.input_proj = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.pos_encoding = PositionalEncoding3D(base_channels)
        
        self.encoder_blocks = nn.ModuleList([
            HybridBlock(base_channels * (2**i), num_heads=8, use_checkpoint=use_checkpoint)
            for i in range(num_blocks)
        ])
        
        self.downsample = nn.ModuleList([
            nn.Conv3d(base_channels * (2**i), base_channels * (2**(i+1)), 3, stride=2, padding=1)
            for i in range(num_blocks-1)
        ])
        
        channels_list = [base_channels * (2**i) for i in range(num_blocks)]
        
        self.upsample = nn.ModuleList([
            nn.ConvTranspose3d(channels_list[num_blocks-1-i], channels_list[num_blocks-2-i], 2, stride=2)
            for i in range(num_blocks-1)
        ])
        
        self.skip_fuse = nn.ModuleList([
            nn.Conv3d(channels_list[num_blocks-2-i] * 2, channels_list[num_blocks-2-i], 1)
            for i in range(num_blocks-1)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            HybridBlock(channels_list[num_blocks-2-i], num_heads=8, use_checkpoint=use_checkpoint)
            for i in range(num_blocks-1)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv3d(base_channels, 1, 1)
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        skips = []
        for i, (block, down) in enumerate(zip(self.encoder_blocks[:-1], self.downsample)):
            x = block(x)
            skips.append(x)
            x = down(x)
        
        x = self.encoder_blocks[-1](x)
        
        for i, (up, fuse, block) in enumerate(zip(self.upsample, self.skip_fuse, self.decoder_blocks)):
            x = up(x)
            skip = skips[-(i+1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            skip_fused = fuse(torch.cat([x, skip], dim=1))
            x = block(skip_fused)
        
        return self.output_proj(x)
