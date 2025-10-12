import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinTransformer3D(nn.Module):
    def __init__(self, in_channels=1, embed_dim=96, depths=[2, 2, 6, 2]):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4)
        
        self.layers = nn.ModuleList()
        for i, depth in enumerate(depths):
            dim = embed_dim * (2 ** i)
            layer = nn.Sequential(
                *[nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ) for _ in range(depth)]
            )
            self.layers.append(layer)
            
            if i < len(depths) - 1:
                self.layers.append(nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2))
        
        self.feature_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
    
    def forward(self, x):
        features = []
        x = self.patch_embed(x)
        
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv3d):
                x = layer(x)
            else:
                B, C, D, H, W = x.shape
                x_flat = x.flatten(2).transpose(1, 2)
                for block in layer:
                    x_flat = block(x_flat)
                x = x_flat.transpose(1, 2).reshape(B, C, D, H, W)
                features.append(x)
        
        return features


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
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
        
        x = x_flat.transpose(1, 2).reshape(B, C, D, H, W)
        return x


class SwinUNETR(nn.Module):
    def __init__(self, img_size=(128, 128, 128), in_channels=1, feature_size=48):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, feature_size, kernel_size=4, stride=4)
        
        self.encoder1 = nn.Sequential(
            SwinTransformerBlock3D(feature_size, num_heads=3),
            SwinTransformerBlock3D(feature_size, num_heads=3)
        )
        self.down1 = nn.Conv3d(feature_size, feature_size * 2, kernel_size=2, stride=2)
        
        self.encoder2 = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 2, num_heads=6),
            SwinTransformerBlock3D(feature_size * 2, num_heads=6)
        )
        self.down2 = nn.Conv3d(feature_size * 2, feature_size * 4, kernel_size=2, stride=2)
        
        self.encoder3 = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 4, num_heads=12),
            SwinTransformerBlock3D(feature_size * 4, num_heads=12)
        )
        self.down3 = nn.Conv3d(feature_size * 4, feature_size * 8, kernel_size=2, stride=2)
        
        self.encoder4 = nn.Sequential(
            SwinTransformerBlock3D(feature_size * 8, num_heads=24),
            SwinTransformerBlock3D(feature_size * 8, num_heads=24)
        )
        self.down4 = nn.Conv3d(feature_size * 8, feature_size * 16, kernel_size=2, stride=2)
        
    def forward(self, x):
        x0 = self.patch_embed(x)
        
        x1 = self.encoder1(x0)
        x1_down = self.down1(x1)
        
        x2 = self.encoder2(x1_down)
        x2_down = self.down2(x2)
        
        x3 = self.encoder3(x2_down)
        x3_down = self.down3(x3)
        
        x4 = self.encoder4(x3_down)
        x4_down = self.down4(x4)
        
        return [x0, x1_down, x2_down, x3_down, x4_down]


class PretrainedEncoder(nn.Module):
    def __init__(self, encoder_type='swin', freeze=False, pretrained_path=None):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'swin':
            self.encoder = SwinTransformer3D(in_channels=1, embed_dim=96, depths=[2, 2, 6, 2])
            self.feature_dims = [96, 192, 384, 768]
            default_path = 'src/pretrained-encoders/swin_transformer.pth'
        elif encoder_type == 'swin_unetr':
            self.encoder = SwinUNETR(in_channels=1, feature_size=48)
            self.feature_dims = [48, 96, 192, 384, 768]
            default_path = 'src/pretrained-encoders/swin_unetr.pth'
        else:
            raise ValueError(f"Unknown encoder: {encoder_type}")
        
        if pretrained_path:
            weight_path = pretrained_path
        else:
            weight_path = default_path
            
        try:
            checkpoint = torch.load(weight_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.encoder.load_state_dict(state_dict, strict=False)
        except:
            pass
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.encoder(x)
