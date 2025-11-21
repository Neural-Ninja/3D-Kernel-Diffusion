import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import yaml
import argparse
import json
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.vae3d import VQVAE3D
from utils.optimized_dataloader import get_vqvae_dataloaders


def gradient_loss(pred, target):
    def gradient_3d(x):
        dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return dx, dy, dz
    
    pred_dx, pred_dy, pred_dz = gradient_3d(pred)
    target_dx, target_dy, target_dz = gradient_3d(target)
    
    loss_dx = F.l1_loss(pred_dx, target_dx)
    loss_dy = F.l1_loss(pred_dy, target_dy)
    loss_dz = F.l1_loss(pred_dz, target_dz)
    
    return (loss_dx + loss_dy + loss_dz) / 3.0


class VQVAETrainer:
    def __init__(self, config):
        self.config = config
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            available_gpus = list(range(torch.cuda.device_count()))
            print(f"Available GPUs: {available_gpus}")
            print(f"Using GPUs: {available_gpus}")
            self.gpu_ids = available_gpus
        else:
            self.device = torch.device('cpu')
            self.gpu_ids = []
            print("No GPU available, using CPU")
        
        self.use_multi_gpu = len(self.gpu_ids) > 1
        
        model_config = config['model']
        self.model = VQVAE3D(
            in_channels=model_config.get('in_channels', 1),
            out_channels=model_config.get('out_channels', 1),
            n_hiddens=model_config.get('n_hiddens', 128),
            downsample=tuple(model_config.get('downsample', [4, 4, 4])),
            num_groups=model_config.get('num_groups', 32),
            embedding_dim=model_config.get('embedding_dim', 256),
            n_codes=model_config.get('n_codes', 512),
            no_random_restart=model_config.get('no_random_restart', False),
            restart_thres=model_config.get('restart_thres', 1.0)
        )
        
        self.model = self.model.to(self.device)
        
        if torch.cuda.is_available():
            print(f"Memory allocated after model.to(device): {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
            print(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        
        if self.use_multi_gpu:
            print(f"Wrapping model with DataParallel across {len(self.gpu_ids)} GPUs")
            self.model = nn.DataParallel(self.model)
            print(f"Model will use GPUs: {list(range(len(self.gpu_ids)))}")
            if torch.cuda.is_available():
                print(f"Memory after DataParallel: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr'],
            weight_decay=config['optimizer']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['optimizer'].get('lr_min', 1e-6)
        )
        
        self.use_amp = config['training'].get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = self.save_dir / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.train_history = []
        self.val_history = []
        
        self.accumulation_steps = config['training'].get('accumulation_steps', 1)
    
    def get_model(self):
        return self.model.module if self.use_multi_gpu else self.model
    
    def train_epoch(self, train_loader):
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'grad_loss': 0.0,
            'vq_loss': 0.0
        }
        
        epoch_perplexity = 0.0
        epoch_code_usage = 0.0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {self.current_epoch}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            x_gt = batch['x_gt'].to(self.device)
            
            # Clear cache before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if self.use_amp:
                with autocast('cuda'):
                    recon, vq_output = self.get_model()(x_gt)
                    recon_loss = nn.functional.l1_loss(recon, x_gt)
                    grad_loss = gradient_loss(recon, x_gt)
                    commitment_loss = vq_output['commitment_loss']
                    
                    # Check for NaN/Inf BEFORE combining losses
                    if torch.isnan(commitment_loss) or torch.isinf(commitment_loss):
                        print(f"\nWARNING: NaN/Inf commitment_loss at batch {batch_idx}")
                        print(f"  commitment_loss: {commitment_loss.item()}")
                        print(f"  Skipping this batch")
                        self.optimizer.zero_grad(set_to_none=True)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        del x_gt, recon, vq_output, recon_loss, grad_loss, commitment_loss
                        continue
                    
                    total_loss = recon_loss + 0.1 * grad_loss + commitment_loss
                    loss = total_loss / self.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWARNING: NaN/Inf loss at batch {batch_idx}")
                    print(f"  recon_loss: {recon_loss.item() if not torch.isnan(recon_loss) else 'NaN'}")
                    print(f"  grad_loss: {grad_loss.item() if not torch.isnan(grad_loss) else 'NaN'}")
                    print(f"  commitment_loss: {commitment_loss.item() if not torch.isnan(commitment_loss) else 'NaN'}")
                    print(f"  x_gt range: [{x_gt.min().item():.4f}, {x_gt.max().item():.4f}]")
                    print(f"  recon range: [{recon.min().item():.4f}, {recon.max().item():.4f}]")
                    self.optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    del x_gt, recon, commitment_loss, recon_loss, grad_loss, total_loss, loss
                    continue
                
                self.scaler.scale(loss).backward()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    params = self.model.module.parameters() if self.use_multi_gpu else self.model.parameters()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"WARNING: NaN/Inf gradient norm detected, skipping step")
                        self.optimizer.zero_grad()
                        self.scaler.update()
                        continue
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                recon, vq_output = self.get_model()(x_gt)
                recon_loss = nn.functional.l1_loss(recon, x_gt)
                grad_loss = gradient_loss(recon, x_gt)
                commitment_loss = vq_output['commitment_loss']
                total_loss = recon_loss + 0.1 * grad_loss + commitment_loss
                loss = total_loss / self.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nWARNING: NaN/Inf loss at batch {batch_idx} (no AMP)")
                    print(f"  recon_loss: {recon_loss.item() if not torch.isnan(recon_loss) else 'NaN'}")
                    print(f"  commitment_loss: {commitment_loss.item() if not torch.isnan(commitment_loss) else 'NaN'}")
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    del x_gt, recon, commitment_loss, recon_loss, total_loss, loss
                    continue
                
                loss.backward()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    params = self.model.module.parameters() if self.use_multi_gpu else self.model.parameters()
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        print(f"WARNING: NaN/Inf gradient norm detected, skipping step")
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['grad_loss'] += grad_loss.item()
            epoch_losses['vq_loss'] += commitment_loss.item()
            epoch_perplexity += vq_output['perplexity'].item()
            epoch_code_usage += vq_output['code_usage_ratio'].item()
            
            pbar.set_postfix({
                'total': f"{total_loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'grad': f"{grad_loss.item():.4f}",
                'vq': f"{commitment_loss.item():.4f}",
                'ppl': f"{vq_output['perplexity'].item():.1f}",
                'codes': f"{vq_output['num_codes_used'].item():.0f}/{self.config['model']['n_codes']}"
            })
            
            del recon, commitment_loss, recon_loss, total_loss, x_gt
            if 'loss' in locals():
                del loss
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        avg_losses['perplexity'] = epoch_perplexity / len(train_loader)
        avg_losses['code_usage_ratio'] = epoch_code_usage / len(train_loader)
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'recon_loss': 0.0,
            'grad_loss': 0.0,
            'vq_loss': 0.0
        }
        
        epoch_perplexity = 0.0
        epoch_code_usage = 0.0
        
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            x_gt = batch['x_gt'].to(self.device)
            
            recon, vq_output = self.get_model()(x_gt)
            recon_loss = nn.functional.l1_loss(recon, x_gt)
            grad_loss = gradient_loss(recon, x_gt)
            commitment_loss = vq_output['commitment_loss']
            total_loss = recon_loss + 0.1 * grad_loss + commitment_loss
            
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['grad_loss'] += grad_loss.item()
            epoch_losses['vq_loss'] += commitment_loss.item()
            epoch_perplexity += vq_output['perplexity'].item()
            epoch_code_usage += vq_output['code_usage_ratio'].item()
            
            del x_gt, recon, commitment_loss, recon_loss, total_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_losses = {k: v / len(val_loader) for k, v in epoch_losses.items()}
        avg_losses['perplexity'] = epoch_perplexity / len(val_loader)
        avg_losses['code_usage_ratio'] = epoch_code_usage / len(val_loader)
        
        return avg_losses
    
    @torch.no_grad()
    def reconstruct_full_volume(self, data_dir, epoch):
        self.model.eval()
        
        gt_dir = Path(data_dir) / 'gt'
        val_files = sorted([f for f in gt_dir.glob('*.nii.gz')])
        
        if len(val_files) == 0:
            print("No validation volumes found")
            return
        
        vol_file = val_files[0]
        print(f"Reconstructing full volume: {vol_file.name}")
        
        img = nib.load(str(vol_file))
        volume = img.get_fdata().astype(np.float32)
        affine = img.affine
        
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            normalized = (volume - vol_min) / (vol_max - vol_min)
            normalized = normalized * 2 - 1
        else:
            normalized = np.full_like(volume, -1.0)
        
        D, H, W = normalized.shape
        patch_size = 64
        stride = 32
        
        reconstructed = np.zeros_like(normalized)
        count_map = np.zeros_like(normalized)
        
        for d in range(0, D, stride):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    d_end = min(d + patch_size, D)
                    h_end = min(h + patch_size, H)
                    w_end = min(w + patch_size, W)
                    
                    patch = normalized[d:d_end, h:h_end, w:w_end]
                    
                    pad_d = patch_size - (d_end - d)
                    pad_h = patch_size - (h_end - h)
                    pad_w = patch_size - (w_end - w)
                    
                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        patch = np.pad(patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
                    
                    patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(self.device)
                    
                    recon_patch, _ = self.get_model()(patch_tensor)
                    recon_patch = recon_patch.cpu().numpy()[0, 0]
                    
                    recon_patch = recon_patch[:d_end-d, :h_end-h, :w_end-w]
                    
                    reconstructed[d:d_end, h:h_end, w:w_end] += recon_patch
                    count_map[d:d_end, h:h_end, w:w_end] += 1
                    
                    del patch_tensor, recon_patch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        reconstructed = reconstructed / np.maximum(count_map, 1)
        
        if vol_max > vol_min:
            reconstructed = (reconstructed + 1) / 2
            reconstructed = reconstructed * (vol_max - vol_min) + vol_min
        
        save_path = self.save_dir / f'epoch_{epoch}_reconstructed.nii.gz'
        recon_img = nib.Nifti1Image(reconstructed.astype(np.float32), affine)
        nib.save(recon_img, str(save_path))
        print(f"Saved full volume reconstruction to {save_path}")
    
    def save_checkpoint(self, is_best=False):
        if self.use_multi_gpu:
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
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
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        latest_path = self.save_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        print(f"  Saved: latest_model.pth")
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved: best_model.pth (Val Loss: {self.best_loss:.4f})")
        
        checkpoint_interval = self.config['training'].get('checkpoint_interval', 10)
        if (self.current_epoch + 1) % checkpoint_interval == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"  Saved: checkpoint_epoch_{self.current_epoch}.pth")
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        state_dict = checkpoint['model_state_dict']
        if self.use_multi_gpu:
            if not list(state_dict.keys())[0].startswith('module.'):
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        else:
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
    
    def train(self, train_loader, val_loader):
        num_epochs = self.config['training']['epochs']
        
        print("\n" + "="*80)
        print("STAGE 1: VQ-VAE TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Multi-GPU: {self.use_multi_gpu}")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['optimizer']['lr']}")
        print(f"Save Directory: {self.save_dir}")
        print("="*80 + "\n")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate(val_loader)
            
            self.reconstruct_full_volume(self.config['data']['data_dir'], epoch)
            
            self.scheduler.step()
            
            self.train_history.append({
                'epoch': epoch,
                **train_losses
            })
            
            self.val_history.append({
                'epoch': epoch,
                **val_losses
            })
            
            is_best = val_losses['total_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_losses['total_loss']
                print(f"\nEpoch {epoch}/{num_epochs} - NEW BEST MODEL!")
                print(f"  Train Loss: {train_losses['total_loss']:.4f} | Perplexity: {train_losses['perplexity']:.1f} | Code Usage: {train_losses['code_usage_ratio']*100:.1f}%")
                print(f"  Val Loss: {val_losses['total_loss']:.4f} | Perplexity: {val_losses['perplexity']:.1f} | Code Usage: {val_losses['code_usage_ratio']*100:.1f}% (Best: {self.best_loss:.4f})")
            else:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Train - Loss: {train_losses['total_loss']:.4f} | PPL: {train_losses['perplexity']:.1f} | Codes: {train_losses['code_usage_ratio']*100:.1f}%")
                print(f"  Val   - Loss: {val_losses['total_loss']:.4f} | PPL: {val_losses['perplexity']:.1f} | Codes: {val_losses['code_usage_ratio']*100:.1f}% (Best: {self.best_loss:.4f})")
            
            self.save_checkpoint(is_best=is_best)
            
            with open(self.log_dir / 'training_history.json', 'w') as f:
                json.dump({
                    'train': self.train_history,
                    'val': self.val_history,
                    'best_loss': self.best_loss,
                    'current_epoch': epoch,
                    'config': self.config
                }, f, indent=2)
        
        print("\n" + "="*80)
        print("STAGE 1 TRAINING COMPLETE")
        print("="*80)
        print(f"Best Validation Loss: {self.best_loss:.6f}")
        print(f"Model saved in: {self.save_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train VQ-VAE Model (Stage 1)')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = VQVAETrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    train_loader, val_loader = get_vqvae_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        train_split=config['data'].get('train_split', 0.7),
        val_split=config['data'].get('val_split', 0.15),
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        patch_size=tuple(config['data']['patch_size']) if 'patch_size' in config['data'] else (64, 64, 64),
        stride=tuple(config['data']['stride']) if 'stride' in config['data'] else None
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
