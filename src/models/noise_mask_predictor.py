import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(spatial))
        return x * attention


class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1, 1)
        return x * attention


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attn = ChannelAttention3D(out_channels)
            self.spatial_attn = SpatialAttention3D()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_attention:
            out = self.channel_attn(out)
            out = self.spatial_attn(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm3d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        out = self.fusion(out)
        
        return out


class NoiseDetectionHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.variance_branch = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.edge_branch = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.freq_branch = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv3d(96, 64, kernel_size=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
    
    def compute_local_variance(self, x):
        kernel_size = 5
        padding = kernel_size // 2
        
        mean = F.avg_pool3d(x, kernel_size=kernel_size, stride=1, padding=padding)
        sq_mean = F.avg_pool3d(x ** 2, kernel_size=kernel_size, stride=1, padding=padding)
        variance = sq_mean - mean ** 2
        
        return variance
    
    def compute_edge_strength(self, x):
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 1, 3, 3)
        sobel_y = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 1, 3, 3)
        sobel_z = torch.tensor([[[-1, -2, -1]], [[0, 0, 0]], [[1, 2, 1]]], 
                               dtype=x.dtype, device=x.device).view(1, 1, 3, 1, 1)
        
        sobel_x = sobel_x.repeat(x.size(1), 1, 1, 1, 1)
        sobel_y = sobel_y.repeat(x.size(1), 1, 1, 1, 1)
        sobel_z = sobel_z.repeat(x.size(1), 1, 1, 1, 1)
        
        grad_x = F.conv3d(x, sobel_x, padding=(0, 1, 1), groups=x.size(1))
        grad_y = F.conv3d(x, sobel_y, padding=(0, 1, 1), groups=x.size(1))
        grad_z = F.conv3d(x, sobel_z, padding=(1, 0, 0), groups=x.size(1))
        
        edge_strength = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-8)
        
        return edge_strength
    
    def forward(self, x):
        variance = self.compute_local_variance(x)
        variance_feat = self.variance_branch(variance)
        
        edges = self.compute_edge_strength(x)
        edge_feat = self.edge_branch(edges)
        
        x_freq = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_freq_mag = torch.abs(x_freq)
        freq_feat = self.freq_branch(x_freq_mag)
        
        combined = torch.cat([variance_feat, edge_feat, freq_feat], dim=1)
        out = self.fusion(combined)
        
        return out


class NoiseMaskPredictor(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_levels=4):
        super().__init__()
        
        self.num_levels = num_levels
        
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        ch = base_channels
        for i in range(num_levels):
            use_attn = i >= 1
            
            self.encoder_blocks.append(
                nn.Sequential(
                    ResidualBlock3D(ch, ch * 2, stride=1, use_attention=use_attn),
                    ResidualBlock3D(ch * 2, ch * 2, stride=1, use_attention=use_attn)
                )
            )
            
            if i < num_levels - 1:
                self.downsample_blocks.append(
                    nn.Conv3d(ch * 2, ch * 2, kernel_size=3, stride=2, padding=1)
                )
            
            ch = ch * 2
        
        self.bottleneck = nn.Sequential(
            MultiScaleFeatureExtractor(ch, ch),
            ResidualBlock3D(ch, ch, use_attention=True),
            NoiseDetectionHead(ch)
        )
        
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(num_levels - 1):
            self.upsample_blocks.append(
                nn.ConvTranspose3d(ch, ch // 2, kernel_size=4, stride=2, padding=1)
            )
            
            self.decoder_blocks.append(
                nn.Sequential(
                    ResidualBlock3D(ch, ch // 2, stride=1, use_attention=True),
                    ResidualBlock3D(ch // 2, ch // 2, stride=1, use_attention=True)
                )
            )
            
            ch = ch // 2
        
        self.output_head = nn.Sequential(
            nn.Conv3d(ch, ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.aux_output = nn.Sequential(
            nn.Conv3d(ch * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x_corrupted):
        x = self.conv_in(x_corrupted)
        
        skips = []
        for i in range(self.num_levels):
            x = self.encoder_blocks[i](x)
            
            if i < self.num_levels - 1:
                skips.append(x)
                x = self.downsample_blocks[i](x)
        
        x = self.bottleneck(x)
        
        aux_features = x
        
        for i in range(self.num_levels - 1):
            x = self.upsample_blocks[i](x)
            
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                diff_d = skip.shape[2] - x.shape[2]
                diff_h = skip.shape[3] - x.shape[3]
                diff_w = skip.shape[4] - x.shape[4]
                x = F.pad(x, [diff_w//2, diff_w - diff_w//2,
                             diff_h//2, diff_h - diff_h//2,
                             diff_d//2, diff_d - diff_d//2])
            
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_blocks[i](x)
        
        noise_mask = self.output_head(x)
        
        aux_mask = self.aux_output(aux_features)
        
        return noise_mask, aux_mask
    
    def predict(self, x_corrupted, threshold=0.5):
        noise_mask, _ = self.forward(x_corrupted)
        binary_mask = (noise_mask > threshold).float()
        return binary_mask


def create_noise_mask_predictor(config=None):
    if config is None:
        config = {
            'in_channels': 1,
            'base_channels': 32,
            'num_levels': 4
        }
    
    model = NoiseMaskPredictor(
        in_channels=config.get('in_channels', 1),
        base_channels=config.get('base_channels', 32),
        num_levels=config.get('num_levels', 4)
    )
    
    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_noise_mask_predictor().to(device)
    
    batch_size = 2
    D, H, W = 64, 128, 128
    x_corrupted = torch.randn(batch_size, 1, D, H, W).to(device)
    
    print(f"Input shape: {x_corrupted.shape}")
    
    noise_mask, aux_mask = model(x_corrupted)
    
    print(f"Output noise mask shape: {noise_mask.shape}")
    print(f"Auxiliary mask shape: {aux_mask.shape}")
    print(f"Noise mask range: [{noise_mask.min():.4f}, {noise_mask.max():.4f}]")
    
    binary_mask = model.predict(x_corrupted, threshold=0.5)
    print(f"Binary mask unique values: {binary_mask.unique()}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
