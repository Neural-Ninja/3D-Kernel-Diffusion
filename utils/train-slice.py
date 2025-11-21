import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
import os
from tqdm import tqdm
from pathlib import Path
import json
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.slice_predictor import create_slice_pair_predictor
from utils.corruption_dataloader import get_slice_pair_dataloaders


class SlicePairTrainer:
    def __init__(self, config):
        self.config = config
        
        if torch.cuda.is_available() and config.get('device') == 'cuda':
            self.gpu_ids = config.get('gpu_ids', None)
            if self.gpu_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
                self.device = torch.device('cuda:0')
                print(f"Using GPUs: {self.gpu_ids}")
            else:
                self.device = torch.device('cuda')
                print(f"Using all available GPUs")
        else:
            self.device = torch.device('cpu')
            self.gpu_ids = None
            print("Using CPU")
        
        self.use_multi_gpu = self.gpu_ids and len(self.gpu_ids) > 1
        if self.use_multi_gpu:
            print(f"Multi-GPU training enabled with {len(self.gpu_ids)} GPUs")
        
        model_config = config['model']
        
        self.model = create_slice_pair_predictor({
            'in_channels': model_config.get('in_channels', 1),
            'feature_size': model_config.get('feature_size', 48),
            'img_size': tuple(model_config.get('img_size', (256, 256))),
            'use_pretrained': model_config.get('use_pretrained', True),
            'cross_attn_heads': model_config.get('cross_attn_heads', 4),
            'cross_attn_layers': model_config.get('cross_attn_layers', 2),
            'use_regression': model_config.get('use_regression', True),
            'axis_embed_dim': model_config.get('axis_embed_dim', 16)
        }).to(self.device)
        
        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        self.lambda_reg = model_config.get('lambda_reg', 1.0)
        
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.save_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.train_history = []
        self.val_history = []
    
    def compute_loss(self, binary_logits, gap_value, has_gap, gap_count):
        binary_loss = F.cross_entropy(binary_logits, has_gap)
        
        if gap_value is not None:
            mask = (has_gap > 0).float()
            if mask.sum() > 0:
                reg_loss = F.smooth_l1_loss(gap_value.squeeze() * mask, gap_count * mask) / (mask.sum() + 1e-8)
            else:
                reg_loss = torch.tensor(0.0, device=binary_loss.device)
            total_loss = binary_loss + self.lambda_reg * reg_loss
        else:
            reg_loss = torch.tensor(0.0)
            total_loss = binary_loss
        
        predicted_has_gap = torch.argmax(binary_logits, dim=1)
        accuracy = (predicted_has_gap == has_gap).float().mean()
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'binary_loss': binary_loss.item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'accuracy': accuracy.item(),
            'n_gaps': mask.sum().item() if gap_value is not None else 0
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        total_n_gaps = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            slice1 = batch['slice1'].to(self.device)
            slice2 = batch['slice2'].to(self.device)
            axis = batch['axis'].to(self.device)
            has_gap = batch['has_gap'].to(self.device)
            gap_count = batch['gap_count'].to(self.device)
            
            self.optimizer.zero_grad()
            
            binary_logits, gap_value = self.model(slice1, slice2, axis)
            
            loss, loss_dict = self.compute_loss(binary_logits, gap_value, has_gap, gap_count)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            total_n_gaps += loss_dict['n_gaps']
            
            for key, value in loss_dict.items():
                if key == 'n_gaps':
                    continue
                if key not in epoch_metrics:
                    epoch_metrics[key] = 0.0
                epoch_metrics[key] += value
            
            pbar.set_postfix({
                'total': f"{loss.item():.4f}",
                'bin': f"{loss_dict['binary_loss']:.4f}",
                'reg': f"{loss_dict['reg_loss']:.4f}",
                'acc': f"{loss_dict['accuracy']:.2%}"
            })
        
        avg_loss = epoch_loss / len(train_loader)
        avg_metrics = {k: v / len(train_loader) for k, v in epoch_metrics.items()}
        if total_n_gaps > 0:
            avg_metrics['reg_loss'] = epoch_metrics['reg_loss'] / total_n_gaps * len(train_loader)
        
        return avg_loss, avg_metrics
    
    def validate(self, val_loader):
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        total_n_gaps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                slice1 = batch['slice1'].to(self.device)
                slice2 = batch['slice2'].to(self.device)
                axis = batch['axis'].to(self.device)
                has_gap = batch['has_gap'].to(self.device)
                gap_count = batch['gap_count'].to(self.device)
                
                binary_logits, gap_value = self.model(slice1, slice2, axis)
                
                loss, loss_dict = self.compute_loss(binary_logits, gap_value, has_gap, gap_count)
                
                epoch_loss += loss.item()
                total_n_gaps += loss_dict['n_gaps']
                
                for key, value in loss_dict.items():
                    if key == 'n_gaps':
                        continue
                    if key not in epoch_metrics:
                        epoch_metrics[key] = 0.0
                    epoch_metrics[key] += value
        
        avg_loss = epoch_loss / len(val_loader)
        avg_metrics = {k: v / len(val_loader) for k, v in epoch_metrics.items()}
        if total_n_gaps > 0:
            # avg_metrics['reg_loss'] = epoch_metrics['reg_loss'] / total_n_gaps * len(val_loader)
            avg_metrics['reg_loss'] = epoch_metrics['reg_loss'] / total_n_gaps
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, is_best=False):
        model_state = self.model.module.state_dict() if self.use_multi_gpu else self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with loss: {self.best_loss:.6f}")
        
        latest_path = self.save_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.use_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader):
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                **train_metrics
            })
            
            self.val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                **val_metrics
            })
            
            print(f"\nEpoch {self.current_epoch}/{num_epochs}")
            print(f"Train - Total: {train_loss:.4f}, Binary: {train_metrics['binary_loss']:.4f}, Reg: {train_metrics['reg_loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
            print(f"Val   - Total: {val_loss:.4f}, Binary: {val_metrics['binary_loss']:.4f}, Reg: {val_metrics['reg_loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(is_best=is_best)
            
            with open(self.log_dir / 'training_history.json', 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history
                }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Slice Pair Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = SlicePairTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    train_loader, val_loader = get_slice_pair_dataloaders(
        config['data'],
        batch_size=config['training']['batch_size']
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
