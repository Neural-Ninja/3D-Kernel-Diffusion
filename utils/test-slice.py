import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.slice_predictor import create_slice_pair_predictor
from utils.corruption_dataloader import get_slice_pair_test_dataloader


class SlicePairTester:
    def __init__(self, config, checkpoint_path):
        self.config = config
        
        if torch.cuda.is_available() and config.get('device') == 'cuda':
            self.gpu_ids = config.get('gpu_ids', None)
            if self.gpu_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
                self.device = torch.device('cuda:0')
                print(f"Using GPUs: {self.gpu_ids}")
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        model_config = config['model']
        
        self.model = create_slice_pair_predictor({
            'in_channels': model_config.get('in_channels', 1),
            'feature_size': model_config.get('feature_size', 48),
            'img_size': tuple(model_config.get('img_size', (256, 256))),
            'use_pretrained': False,
            'cross_attn_heads': model_config.get('cross_attn_heads', 4),
            'cross_attn_layers': model_config.get('cross_attn_layers', 2),
            'use_regression': model_config.get('use_regression', True),
            'axis_embed_dim': model_config.get('axis_embed_dim', 16)
        }).to(self.device)
        
        self.load_checkpoint(checkpoint_path)
        
        self.results_dir = Path(config['testing']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.lambda_reg = model_config.get('lambda_reg', 1.0)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        print(f"Loaded model weights from {checkpoint_path}")
    
    def compute_metrics(self, binary_logits, gap_value, has_gap, gap_count):
        binary_loss = F.cross_entropy(binary_logits, has_gap)
        
        if gap_value is not None:
            mask = (has_gap > 0).float()
            if mask.sum() > 0:
                reg_loss = F.smooth_l1_loss(gap_value.squeeze() * mask, gap_count * mask) / (mask.sum() + 1e-8)
                mae = F.l1_loss(gap_value.squeeze() * mask, gap_count * mask) / (mask.sum() + 1e-8)
            else:
                reg_loss = torch.tensor(0.0)
                mae = torch.tensor(0.0)
        else:
            reg_loss = torch.tensor(0.0)
            mae = torch.tensor(0.0)
        
        predicted_has_gap = torch.argmax(binary_logits, dim=1)
        accuracy = (predicted_has_gap == has_gap).float().mean()
        
        return {
            'binary_loss': binary_loss.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'accuracy': accuracy.item(),
            'mae': mae.item() if isinstance(mae, torch.Tensor) else mae
        }
    
    def test(self, test_loader):
        self.model.eval()
        
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                slice1 = batch['slice1'].to(self.device)
                slice2 = batch['slice2'].to(self.device)
                axis = batch['axis'].to(self.device)
                has_gap = batch['has_gap'].to(self.device)
                gap_count = batch['gap_count'].to(self.device)
                
                binary_logits, gap_value = self.model(slice1, slice2, axis)
                
                metrics = self.compute_metrics(binary_logits, gap_value, has_gap, gap_count)
                all_metrics.append(metrics)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
        
        print("\n" + "="*50)
        print("Test Results:")
        print("="*50)
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.6f}")
        print("="*50)
        
        results_file = self.results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Test Slice Pair Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint (overrides config)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = args.checkpoint if args.checkpoint else config['testing']['checkpoint']
    
    tester = SlicePairTester(config, checkpoint_path)
    
    test_loader = get_slice_pair_test_dataloader(
        config['data'],
        batch_size=config['testing']['batch_size']
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    tester.test(test_loader)


if __name__ == '__main__':
    main()
