import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
import argparse
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.noise_mask_predictor import create_noise_mask_predictor


class NoiseMaskDataset(Dataset):
    def __init__(self, data_dir, split='train', max_samples=None):
        self.data_dir = Path(data_dir)
        self.split = split
        
        self.samples = []
        
        corrupted_dir = self.data_dir / 'corrupted'
        mask_dir = self.data_dir / 'noise_masks'
        
        if not corrupted_dir.exists() or not mask_dir.exists():
            raise ValueError(f"Data directories not found: {corrupted_dir}, {mask_dir}")
        
        corrupted_files = sorted(list(corrupted_dir.glob('*.nii.gz')))
        
        for corr_file in corrupted_files:
            mask_file = mask_dir / corr_file.name
            
            if mask_file.exists():
                self.samples.append({
                    'corrupted': corr_file,
                    'mask': mask_file
                })
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def load_nifti(self, filepath):
        """Load and normalize NIfTI file"""
        img = nib.load(str(filepath))
        data = img.get_fdata().astype(np.float32)
        
        if data.max() > data.min():
            data = (data - data.min()) / (data.max() - data.min())
        
        return data
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        x_corrupted = self.load_nifti(sample['corrupted'])
        
        noise_mask = self.load_nifti(sample['mask'])
        
        x_corrupted = torch.from_numpy(x_corrupted).unsqueeze(0)
        noise_mask = torch.from_numpy(noise_mask).unsqueeze(0)
        
        return {
            'x_corrupted': x_corrupted,
            'noise_mask': noise_mask,
            'filename': sample['corrupted'].name
        }


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, lambda_dice=1.0, lambda_bce=1.0, lambda_focal=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.focal_loss = FocalLoss()
        
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.lambda_focal = lambda_focal
    
    def forward(self, pred, target, aux_pred=None):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        
        main_loss = (self.lambda_dice * dice + 
                     self.lambda_bce * bce + 
                     self.lambda_focal * focal)
        
        aux_loss = 0.0
        if aux_pred is not None:
            target_down = F.interpolate(target, size=aux_pred.shape[2:], 
                                       mode='trilinear', align_corners=False)
            aux_dice = self.dice_loss(aux_pred, target_down)
            aux_bce = self.bce_loss(aux_pred, target_down)
            aux_loss = 0.5 * (aux_dice + aux_bce)
        
        total_loss = main_loss + aux_loss
        
        return total_loss, {
            'dice': dice.item(),
            'bce': bce.item(),
            'focal': focal.item(),
            'aux': aux_loss if isinstance(aux_loss, float) else aux_loss.item()
        }


class NoiseMaskTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.model = create_noise_mask_predictor(config.get('model', {}))
        self.model = self.model.to(self.device)
        
        self.criterion = CombinedLoss(
            lambda_dice=config.get('lambda_dice', 1.0),
            lambda_bce=config.get('lambda_bce', 1.0),
            lambda_focal=config.get('lambda_focal', 0.5)
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer'].get('weight_decay', 1e-4)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['optimizer'].get('lr_min', 1e-6)
        )
        
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.train_history = []
        self.val_history = []
    
    def compute_metrics(self, pred, target, threshold=0.5):
        pred_binary = (pred > threshold).float()
        target_binary = target
        
        intersection = (pred_binary * target_binary).sum()
        dice = (2.0 * intersection + 1e-7) / (pred_binary.sum() + target_binary.sum() + 1e-7)
        
        union = pred_binary.sum() + target_binary.sum() - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item()
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {'dice': 0.0, 'bce': 0.0, 'focal': 0.0, 'aux': 0.0}
        epoch_eval_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            x_corrupted = batch['x_corrupted'].to(self.device)
            noise_mask_gt = batch['noise_mask'].to(self.device)
            
            noise_mask_pred, aux_pred = self.model(x_corrupted)
            
            loss, loss_dict = self.criterion(noise_mask_pred, noise_mask_gt, aux_pred)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key]
            
            with torch.no_grad():
                eval_metrics = self.compute_metrics(noise_mask_pred, noise_mask_gt)
                for key in epoch_eval_metrics:
                    epoch_eval_metrics[key] += eval_metrics[key]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{eval_metrics['dice']:.4f}"
            })
        
        n_batches = len(train_loader)
        avg_loss = epoch_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        avg_eval_metrics = {k: v / n_batches for k, v in epoch_eval_metrics.items()}
        
        return avg_loss, avg_metrics, avg_eval_metrics
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_metrics = {'dice': 0.0, 'bce': 0.0, 'focal': 0.0, 'aux': 0.0}
        epoch_eval_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        for batch in tqdm(val_loader, desc='Validation'):
            x_corrupted = batch['x_corrupted'].to(self.device)
            noise_mask_gt = batch['noise_mask'].to(self.device)
            
            noise_mask_pred, aux_pred = self.model(x_corrupted)
            
            loss, loss_dict = self.criterion(noise_mask_pred, noise_mask_gt, aux_pred)
            
            epoch_loss += loss.item()
            for key in epoch_metrics:
                epoch_metrics[key] += loss_dict[key]
            
            eval_metrics = self.compute_metrics(noise_mask_pred, noise_mask_gt)
            for key in epoch_eval_metrics:
                epoch_eval_metrics[key] += eval_metrics[key]
        
        n_batches = len(val_loader)
        avg_loss = epoch_loss / n_batches
        avg_metrics = {k: v / n_batches for k, v in epoch_metrics.items()}
        avg_eval_metrics = {k: v / n_batches for k, v in epoch_eval_metrics.items()}
        
        return avg_loss, avg_metrics, avg_eval_metrics
    
    def save_checkpoint(self, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.config
        }
        
        torch.save(checkpoint, self.save_dir / 'latest_model.pth')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f"  Saved best model (Val Loss: {self.best_loss:.4f})")
        
        if (self.current_epoch + 1) % self.config['training'].get('checkpoint_interval', 10) == 0:
            torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth')
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        num_epochs = self.config['training']['epochs']
        
        print("\n" + "="*80)
        print("TRAINING NOISE MASK PREDICTOR")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("="*80 + "\n")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_loss, train_metrics, train_eval = self.train_epoch(train_loader)
            
            val_loss, val_metrics, val_eval = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.train_history.append({
                'epoch': epoch,
                'loss': train_loss,
                **train_metrics,
                **train_eval
            })
            
            self.val_history.append({
                'epoch': epoch,
                'loss': val_loss,
                **val_metrics,
                **val_eval
            })
            
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_eval['dice']:.4f}, IoU: {train_eval['iou']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_eval['dice']:.4f}, IoU: {val_eval['iou']:.4f}")
            if is_best:
                print(f"  *** NEW BEST MODEL ***")
            
            self.save_checkpoint(is_best=is_best)
            
            with open(self.save_dir / 'training_history.json', 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history,
                    'best_loss': self.best_loss
                }, f, indent=2)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print(f"Best Validation Loss: {self.best_loss:.4f}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Noise Mask Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_dataset = NoiseMaskDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        max_samples=config['data'].get('max_samples')
    )
    
    val_dataset = NoiseMaskDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        max_samples=config['data'].get('max_samples')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    trainer = NoiseMaskTrainer(config)
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.current_epoch = checkpoint['epoch']
        trainer.best_loss = checkpoint['best_loss']
        trainer.train_history = checkpoint['train_history']
        trainer.val_history = checkpoint['val_history']
        print(f"Resumed from epoch {trainer.current_epoch}")
    
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
