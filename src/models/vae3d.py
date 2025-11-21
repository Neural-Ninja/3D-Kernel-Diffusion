import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

def shift_dim(x, src_dim=-1, dest_dim=-1):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim
    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims
    dims = list(range(n_dims))
    del dims[src_dim]
    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    return x

def silu(x):
    return x * torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return silu(x)

def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels, num_groups=num_groups)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.norm2 = Normalize(out_channels, num_groups=num_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Codebook(nn.Module):
    def __init__(self, n_codes, embedding_dim, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.register_buffer('embeddings', torch.randn(n_codes, embedding_dim))
        self.register_buffer('N', torch.zeros(n_codes))
        self.register_buffer('z_avg', self.embeddings.data.clone())
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
    def _tile(self, x):
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x
    def _init_embeddings(self, z):
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)
        _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes]
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))
    def forward(self, z):
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
            - 2 * flat_inputs @ self.embeddings.t() \
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])
        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)
        commitment_loss = F.mse_loss(z, embeddings.detach())
        commitment_loss = torch.clamp(commitment_loss, max=10.0) * 0.25
        if self.training:
            n_total = encode_onehot.sum(dim=0).to(self.N.dtype)
            encode_sum = flat_inputs.t() @ encode_onehot
            with torch.no_grad():
                self.N.mul_(0.99)
                self.N.add_(n_total, alpha=0.01)
                self.z_avg.mul_(0.99)
                self.z_avg.add_(encode_sum.t(), alpha=0.01)
                denom = self.N.unsqueeze(1).clamp(min=1e-6)
                encode_normalized = self.z_avg / denom
                valid = (self.N > 0)
                if valid.any():
                    valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
                    self.embeddings[valid_idx] = encode_normalized[valid_idx]
                self.embeddings.data.clamp_(-10.0, 10.0)
                if not self.no_random_restart:
                    y = self._tile(flat_inputs)
                    _k_rand = y[torch.randperm(y.shape[0])][:self.n_codes].to(self.embeddings.dtype).to(self.embeddings.device)
                    usage = (self.N.view(self.n_codes, 1) >= self.restart_thres)
                    usage_mask = usage.float().to(self.embeddings.dtype)
                    self.embeddings.mul_(usage_mask).add_(_k_rand * (1 - usage_mask))
        embeddings_st = (embeddings - z).detach() + z
        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))
        num_codes_used = (avg_probs > 0).sum()
        code_usage_ratio = num_codes_used.float() / self.n_codes
        return dict(embeddings=embeddings_st, encodings=encoding_indices,
                    commitment_loss=commitment_loss, perplexity=perplexity,
                    num_codes_used=num_codes_used, code_usage_ratio=code_usage_ratio)
    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings

class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=1, num_groups=32, embedding_dim=8):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(image_channel, n_hiddens, kernel_size=3, stride=1, padding=1)
        channels = [n_hiddens * 2 ** i for i in range(max_ds)]
        channels = channels + [channels[-1]]
        in_channels = channels[0]
        for i in range(max_ds + 1):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i-1]
            out_channels = channels[i]
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            if in_channels != out_channels:
                block.res1 = ResBlock3D(in_channels, out_channels, num_groups=num_groups)
            else:
                block.res1 = ResBlock3D(in_channels, out_channels, num_groups=num_groups)
            block.res2 = ResBlock3D(out_channels, out_channels, num_groups=num_groups)
            if i != max_ds:
                block.down = nn.Conv3d(out_channels, out_channels, kernel_size=(4, 4, 4), 
                                     stride=stride, padding=1)
            else:
                block.down = nn.Identity()
            self.conv_blocks.append(block)
            n_times_downsample -= 1
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlock3D(out_channels, out_channels, num_groups=num_groups)
        self.mid_block.res2 = ResBlock3D(out_channels, out_channels, num_groups=num_groups)
        self.final_block = nn.Sequential(
            Normalize(out_channels, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, self.embedding_dim, 3, 1, 1)
        )
        self.out_channels = out_channels
    def forward(self, x):
        h = self.conv_first(x)
        skips = []
        for block in self.conv_blocks:
            h = block.res1(h)
            h = block.res2(h)
            skips.append(h)
            h = block.down(h)
        h = self.mid_block.res1(h)
        h = self.mid_block.res2(h)
        h = self.final_block(h)
        return h, skips

class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, num_groups=32, embedding_dim=8):
        super().__init__()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        encoder_channels = [n_hiddens * 2 ** i for i in range(max_us)]
        encoder_channels = encoder_channels + [encoder_channels[-1]]
        channels = list(reversed(encoder_channels))
        self.embedding_dim = embedding_dim
        self.conv_first = nn.Conv3d(self.embedding_dim, channels[0], 3, 1, 1)
        self.mid_block = nn.Module()
        self.mid_block.res1 = ResBlock3D(channels[0], channels[0], num_groups=num_groups)
        self.mid_block.res2 = ResBlock3D(channels[0], channels[0], num_groups=num_groups)
        self.conv_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        in_channels = channels[0]
        for i in range(len(channels)):
            block = nn.Module()
            if i != 0:
                in_channels = channels[i-1]
            out_channels = channels[i]
            if in_channels != out_channels:
                block.res1 = ResBlock3D(in_channels, out_channels, num_groups=num_groups)
            else:
                block.res1 = ResBlock3D(in_channels, out_channels, num_groups=num_groups)
            block.res2 = ResBlock3D(out_channels, out_channels, num_groups=num_groups)
            if i != len(channels)-1:
                block.up = nn.ConvTranspose3d(out_channels, out_channels, 4, stride=2, padding=1)
            else:
                block.up = nn.Identity()
            self.conv_blocks.append(block)
            expected_in = block.res1.in_channels
            skip_ch = encoder_channels[::-1][i]
            concat_ch = expected_in + skip_ch
            self.skip_convs.append(nn.Conv3d(concat_ch, expected_in, kernel_size=1))
            n_times_upsample -= 1
        self.final_block = nn.Sequential(
            Normalize(out_channels, num_groups=num_groups),
            SiLU(),
            nn.Conv3d(out_channels, image_channel, 3, 1, 1)
        )
    def forward(self, x, skips=None):
        h = self.conv_first(x)
        h = self.mid_block.res1(h)
        h = self.mid_block.res2(h)
        if skips is not None:
            skips = skips[::-1]
        for i, block in enumerate(self.conv_blocks):
            if skips is not None and i < len(skips):
                skip = skips[i]
                if h.shape[2:] != skip.shape[2:]:
                    skip = F.interpolate(skip, size=h.shape[2:], mode='trilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = self.skip_convs[i](h)
            h = block.res1(h)
            h = block.res2(h)
            h = block.up(h)
        h = self.final_block(h)
        return h

class VQVAE3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_hiddens=128, downsample=(2, 2, 2),
                 num_groups=32, embedding_dim=256, n_codes=512, no_random_restart=False, restart_thres=1.0):
        super().__init__()
        self.encoder = Encoder(n_hiddens, downsample, in_channels, num_groups, embedding_dim)
        self.decoder = Decoder(n_hiddens, downsample, out_channels, num_groups, embedding_dim)
        self.codebook = Codebook(n_codes, embedding_dim, no_random_restart, restart_thres)
        self.pre_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)
        self.post_vq_conv = nn.Conv3d(embedding_dim, embedding_dim, 1, 1)
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
    def encode(self, x):
        h, skips = self.encoder(x)
        h = self.pre_vq_conv(h)
        vq_output = self.codebook(h)
        return vq_output['embeddings'], vq_output, skips
    def decode(self, z, skips=None):
        h = self.post_vq_conv(z)
        return self.decoder(h, skips)
    def forward(self, x):
        z, vq_output, skips = self.encode(x)
        x_recon = self.decode(z, skips)
        return x_recon, vq_output

VAE3D = VQVAE3D

def vqvae_loss(recon, target, vq_output, lambda_vq=1.0):
    recon_loss = F.l1_loss(recon, target)
    commitment_loss = vq_output['commitment_loss']
    total_loss = recon_loss + lambda_vq * commitment_loss
    return total_loss, recon_loss, commitment_loss

def vae_loss(recon, target, vq_output, lambda_vq=1.0):
    return vqvae_loss(recon, target, vq_output, lambda_vq)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae = VQVAE3D(
        in_channels=1,
        out_channels=1,
        n_hiddens=128,
        downsample=(4, 4, 4),
        num_groups=32,
        embedding_dim=256,
        n_codes=512
    ).to(device)
    batch_size = 2
    D, H, W = 64, 64, 64
    x = torch.randn(batch_size, 1, D, H, W).to(device)
    print(f"Input shape: {x.shape}")
    recon, vq_output = vqvae(x)
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Commitment loss: {vq_output['commitment_loss'].item():.4f}")
    print(f"Perplexity: {vq_output['perplexity'].item():.4f}")
    total_loss, recon_loss, commitment_loss = vqvae_loss(recon, x, vq_output)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Recon loss: {recon_loss.item():.4f}")
    total_params = sum(p.numel() for p in vqvae.parameters())
    print(f"Total parameters: {total_params:,}")
