import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        pos_h = torch.arange(H, device=x.device).float().view(H, 1)
        pos_w = torch.arange(W, device=x.device).float().view(1, W)
        
        div_term = torch.exp(torch.arange(0, C, 2, device=x.device).float() * (-math.log(10000.0) / C))
        
        pe = torch.zeros(C, H, W, device=x.device)
        
        pe[0::2] = torch.sin(pos_h * div_term.view(-1, 1, 1))
        pe[1::2] = torch.cos(pos_h * div_term.view(-1, 1, 1))
        
        return x + pe.unsqueeze(0)


class SwinEncoder2D(nn.Module):
    def __init__(self, img_size=(256, 256), in_channels=1, feature_size=48, use_pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.feature_size = feature_size
        
        self.patch_embed = nn.Conv2d(in_channels, feature_size, kernel_size=4, stride=4)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(feature_size, feature_size*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, feature_size*2),
            nn.GELU(),
            nn.Conv2d(feature_size*2, feature_size*4, 3, stride=2, padding=1),
            nn.GroupNorm(8, feature_size*4),
            nn.GELU(),
            nn.Conv2d(feature_size*4, feature_size*8, 3, stride=2, padding=1),
            nn.GroupNorm(8, feature_size*8),
            nn.GELU(),
            nn.Conv2d(feature_size*8, feature_size*16, 3, stride=1, padding=1),
            nn.GroupNorm(8, feature_size*16),
            nn.GELU()
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.pos_encoding = PositionalEncoding2D(feature_size * 16)
        
        if use_pretrained:
            self._load_pretrained_weights()
    
    def _load_pretrained_weights(self):
        from pathlib import Path
        
        current_dir = Path(__file__).parent
        
        mri_weight_path = current_dir.parent / "pretrained-weights" / "swin_mri_pretrained.pt"
        old_weight_path = current_dir.parent / "pretrained-weights" / "model_swinvit.pt"
        
        weight_path = mri_weight_path if mri_weight_path.exists() else old_weight_path
        
        if weight_path.exists():
            try:
                checkpoint = torch.load(str(weight_path), map_location='cpu')
                state_dict = checkpoint.get('state_dict', checkpoint)
                
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('module.'):
                        k = k[7:]
                    
                    if len(v.shape) == 5:
                        print(f"Skipping 3D weight: {k} {list(v.shape)}")
                        continue
                    
                    if 'patch_embed' in k or 'layers' in k or 'encoder' in k:
                        new_state_dict[k] = v
                
                if new_state_dict:
                    self.load_state_dict(new_state_dict, strict=False)
                    source = checkpoint.get('source', 'unknown')
                    print(f"Pretrained weights loaded successfully from: {source}")
                else:
                    print("No compatible 2D weights found in checkpoint.")
            except Exception as e:
                print(f"Failed to load pretrained weights: {e}")
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.adaptive_pool(x)
        x = self.pos_encoding(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=256, heads=4, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=512, batch_first=True)
            for _ in range(layers)
        ])
    
    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        f1_flat = f1.flatten(2).permute(0, 2, 1)
        f2_flat = f2.flatten(2).permute(0, 2, 1)
        fused = torch.cat([f1_flat, f2_flat], dim=1)
        for layer in self.layers:
            fused = layer(fused)
        return fused.mean(1)


class AxisEmbedding(nn.Module):
    def __init__(self, embedding_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
    
    def forward(self, axis_indices):
        return self.embedding(axis_indices)


class MultiTaskPredictionHead(nn.Module):
    def __init__(self, input_dim, use_regression=True, axis_embed_dim=16):
        super().__init__()
        self.use_regression = use_regression
        
        total_dim = input_dim * 4 + axis_embed_dim
        
        self.binary_classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
        
        if use_regression:
            self.regressor = nn.Sequential(
                nn.Linear(total_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
    
    def forward(self, f1_mean, f2_mean, f_fused, axis_embed):
        f_diff = torch.abs(f1_mean - f2_mean)
        z = torch.cat([f1_mean, f2_mean, f_diff, f_fused, axis_embed], dim=1)
        
        binary_logits = self.binary_classifier(z)
        
        if self.use_regression:
            gap_value = self.regressor(z)
            return binary_logits, gap_value
        else:
            return binary_logits, None


class SlicePairPredictor(nn.Module):
    def __init__(self, in_channels=1, feature_size=48, img_size=(256, 256), 
                 use_pretrained=True, cross_attn_heads=4, cross_attn_layers=2,
                 use_regression=True, axis_embed_dim=16):
        super().__init__()
        self.img_size = img_size
        self.feature_size = feature_size
        
        self.encoder = SwinEncoder2D(
            img_size=img_size,
            in_channels=in_channels,
            feature_size=feature_size,
            use_pretrained=use_pretrained
        )
        
        encoder_out_dim = feature_size * 16
        
        self.cross_attention = CrossAttentionFusion(
            dim=encoder_out_dim,
            heads=cross_attn_heads,
            layers=cross_attn_layers
        )
        
        self.axis_embedding = AxisEmbedding(embedding_dim=axis_embed_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.prediction_head = MultiTaskPredictionHead(
            input_dim=encoder_out_dim,
            use_regression=use_regression,
            axis_embed_dim=axis_embed_dim
        )
    
    def forward(self, slice1, slice2, axis):
        f1 = self.encoder(slice1)
        f2 = self.encoder(slice2)
        
        f1_mean = self.global_pool(f1).view(f1.size(0), -1)
        f2_mean = self.global_pool(f2).view(f2.size(0), -1)
        
        f_fused = self.cross_attention(f1, f2)
        
        axis_embed = self.axis_embedding(axis)
        
        binary_logits, gap_value = self.prediction_head(f1_mean, f2_mean, f_fused, axis_embed)
        
        return binary_logits, gap_value
    
    def predict_gap_count(self, slice1, slice2, axis):
        binary_logits, gap_value = self.forward(slice1, slice2, axis)
        has_gap = torch.argmax(binary_logits, dim=1)
        if gap_value is not None:
            gap_count = torch.clamp(gap_value.squeeze(), min=0).round()
        else:
            gap_count = has_gap.float()
        return has_gap, gap_count


class PatchBasedSlicePairPredictor(nn.Module):
    def __init__(self, base_predictor, patch_size=256, overlap=64):
        super().__init__()
        self.base_predictor = base_predictor
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
    
    def extract_patches(self, slice_img):
        batch_size, channels, height, width = slice_img.shape
        patch_size = self.patch_size
        stride = self.stride
        
        patch_list = []
        patch_positions = []
        
        for height_idx in range(0, max(1, height - patch_size + 1), stride):
            for width_idx in range(0, max(1, width - patch_size + 1), stride):
                patch = slice_img[:, :, 
                              height_idx:min(height_idx + patch_size, height), 
                              width_idx:min(width_idx + patch_size, width)]
                if patch.shape[2] < patch_size or patch.shape[3] < patch_size:
                    pad_height = patch_size - patch.shape[2]
                    pad_width = patch_size - patch.shape[3]
                    patch = F.pad(patch, (0, pad_width, 0, pad_height))
                patch_list.append(patch)
                patch_positions.append((height_idx, width_idx))
        
        return patch_list, patch_positions
    
    def merge_predictions(self, predictions):
        logits_list = [pred[0] for pred in predictions]
        gap_values_list = [pred[1] for pred in predictions]
        
        logits_avg = torch.stack(logits_list).mean(0)
        
        if gap_values_list[0] is not None:
            gap_values_avg = torch.stack(gap_values_list).mean(0)
        else:
            gap_values_avg = None
        
        return logits_avg, gap_values_avg
    
    def forward(self, slice1, slice2, axis):
        batch_size, channels, height, width = slice1.shape
        
        if height == self.patch_size and width == self.patch_size:
            return self.base_predictor(slice1, slice2, axis)
        
        patch_list1, patch_positions = self.extract_patches(slice1)
        patch_list2, _ = self.extract_patches(slice2)
        
        predictions = []
        for patch1, patch2 in zip(patch_list1, patch_list2):
            with torch.no_grad():
                logits, gap_value = self.base_predictor(patch1, patch2, axis)
                predictions.append((logits, gap_value))
        
        logits_avg, gap_values_avg = self.merge_predictions(predictions)
        
        return logits_avg, gap_values_avg
    


def create_slice_pair_predictor(config=None):
    if config is None:
        config = {
            'in_channels': 1,
            'feature_size': 48,
            'img_size': (256, 256),
            'use_pretrained': True,
            'cross_attn_heads': 4,
            'cross_attn_layers': 2,
            'use_regression': True,
            'axis_embed_dim': 16
        }
    
    predictor = SlicePairPredictor(
        in_channels=config.get('in_channels', 1),
        feature_size=config.get('feature_size', 48),
        img_size=tuple(config.get('img_size', (256, 256))),
        use_pretrained=config.get('use_pretrained', True),
        cross_attn_heads=config.get('cross_attn_heads', 4),
        cross_attn_layers=config.get('cross_attn_layers', 2),
        use_regression=config.get('use_regression', True),
        axis_embed_dim=config.get('axis_embed_dim', 16)
    )
    return predictor


def create_patch_based_predictor(base_predictor=None, patch_size=256, overlap=64):
    if base_predictor is None:
        base_predictor = create_slice_pair_predictor({'img_size': (patch_size, patch_size)})
    return PatchBasedSlicePairPredictor(base_predictor, patch_size, overlap)


