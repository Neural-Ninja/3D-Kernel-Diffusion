import torch
import torch.nn as nn
from pathlib import Path
import yaml
import argparse
import json
import os
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import nibabel as nib
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.latent_diffusion import LatentDiffusionModel
from utils.optimized_dataloader import get_optimized_dataloaders


class LatentDiffusionTrainer:
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
        self.model = LatentDiffusionModel(
            vae_in_channels=model_config.get('vae_in_channels', 1),
            vae_out_channels=model_config.get('vae_out_channels', 1),
            vae_n_hiddens=model_config.get('vae_n_hiddens', 128),
            vae_downsample=tuple(model_config.get('vae_downsample', [4, 4, 4])),
            vae_num_groups=model_config.get('vae_num_groups', 32),
            vae_embedding_dim=model_config.get('vae_embedding_dim', 256),
            vae_n_codes=model_config.get('vae_n_codes', 512),
            unet_model_channels=model_config.get('unet_model_channels', 128),
            unet_channel_mult=tuple(model_config.get('unet_channel_mult', [1, 2, 4])),
            unet_attention_levels=model_config.get('unet_attention_levels', [1, 2]),
            unet_num_heads=model_config.get('unet_num_heads', 4),
            unet_time_emb_dim=model_config.get('unet_time_emb_dim', 512),
            unet_context_dim=model_config.get('unet_context_dim', 256),
            diffusion_timesteps=model_config.get('diffusion_timesteps', 1000),
            diffusion_beta_schedule=model_config.get('diffusion_beta_schedule', 'cosine'),
            vae_checkpoint=model_config.get('vae_checkpoint', None),
            freeze_vae=model_config.get('freeze_vae', True),
            device=self.device
        )
        
        self.model = self.model.to(self.device)
        
        if self.use_multi_gpu:
            print(f"Wrapping model with DataParallel across {len(self.gpu_ids)} GPUs")
            self.model = nn.DataParallel(self.model)
            print(f"Model will use GPUs: {list(range(len(self.gpu_ids)))}")
        else:
            if torch.cuda.is_available():
                print(f"Single GPU mode on device: {self.device}")
            else:
                print("CPU mode")
        
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
        self.scaler = GradScaler('cuda') if self.use_amp else None
        if self.use_amp:
            print("Mixed precision training (AMP) enabled")
        else:
            print("Mixed precision training (AMP) disabled")
        
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
            'diffusion_loss': 0.0,
            'x0_loss': 0.0,
            'vq_loss': 0.0,
            'mse_loss': 0.0,
            'ssim_loss': 0.0,
            'perceptual_loss': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}', leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            x_gt = batch['x_gt'].to(self.device)
            x_corr = batch['x_corr'].to(self.device)
            m_mask = batch['m_mask'].to(self.device)
            
            if self.use_amp:
                with autocast('cuda'):
                    loss_dict = self.get_model().compute_loss(x_gt, x_corr, m_mask)
                    loss = loss_dict['total_loss'] / self.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n{'='*60}")
                    print(f"WARNING: NaN/Inf loss at batch {batch_idx}")
                    print(f"Loss components:")
                    for k, v in loss_dict.items():
                        print(f"  {k}: {v.item() if not torch.isnan(v) else 'NaN'}")
                    print(f"{'='*60}\n")
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
                    
                    if grad_norm > 10.0:
                        print(f"WARNING: Large gradient norm: {grad_norm:.2f}")
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                loss_dict = self.get_model().compute_loss(x_gt, x_corr, m_mask)
                loss = loss_dict['total_loss'] / self.accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n{'='*60}")
                    print(f"WARNING: NaN/Inf loss at batch {batch_idx}")
                    print(f"Loss components:")
                    for k, v in loss_dict.items():
                        print(f"  {k}: {v.item() if not torch.isnan(v) else 'NaN'}")
                    print(f"{'='*60}\n")
                    self.optimizer.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
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
                    
                    if grad_norm > 10.0:
                        print(f"WARNING: Large gradient norm: {grad_norm:.2f}")
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            loss_values = {k: v.item() for k, v in loss_dict.items() if not torch.isnan(v)}
            
            for key in epoch_losses.keys():
                if key in loss_values:
                    epoch_losses[key] += loss_values[key]
            
            pbar.set_postfix({
                'total': f"{loss_values.get('total_loss', 0):.4f}",
                'diff': f"{loss_values.get('diffusion_loss', 0):.4f}",
                'mse': f"{loss_values.get('mse_loss', 0):.4f}",
                'perc': f"{loss_values.get('perceptual_loss', 0):.4f}"
            })
            
            del loss_dict, x_gt, x_corr, m_mask
            if 'loss' in locals():
                del loss
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_losses = {k: v / len(train_loader) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'diffusion_loss': 0.0,
            'x0_loss': 0.0,
            'vq_loss': 0.0,
            'mse_loss': 0.0,
            'ssim_loss': 0.0,
            'perceptual_loss': 0.0
        }
        
        for batch in tqdm(val_loader, desc='Validation', leave=False):
            x_gt = batch['x_gt'].to(self.device)
            x_corr = batch['x_corr'].to(self.device)
            m_mask = batch['m_mask'].to(self.device)
            
            loss_dict = self.get_model().compute_loss(x_gt, x_corr, m_mask)

            for k, v in loss_dict.items():
                try:
                    val = v.item() if hasattr(v, 'item') else float(v)
                except Exception:
                    continue

                if k in epoch_losses:
                    epoch_losses[k] += val
                else:
                    epoch_losses[k] = val
        
        avg_losses = {k: v / len(val_loader) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def reconstruct_full_volume(self, data_dir, epoch):
        self.model.eval()

        gt_dir = Path(data_dir) / 'gt'
        corrupted_dir = Path(data_dir) / 'corrupted'
        noise_mask_dir = Path(data_dir) / 'noise-mask'
        val_files = sorted([f for f in gt_dir.glob('*.nii.gz')])

        if len(val_files) == 0:
            print("No validation volumes found")
            return

        vol_file = val_files[0]
        print(f"Reconstructing full volume: {vol_file.name}")

        gt_img = nib.load(str(vol_file))
        gt_volume = gt_img.get_fdata().astype(np.float32)
        affine = gt_img.affine

        corrupted_file = corrupted_dir / vol_file.name
        corrupted_img = nib.load(str(corrupted_file))
        corrupted_volume = corrupted_img.get_fdata().astype(np.float32)

        noise_mask_file = noise_mask_dir / vol_file.name
        if noise_mask_file.exists():
            noise_mask_volume = nib.load(str(noise_mask_file)).get_fdata().astype(np.float32)
        else:
            noise_mask_volume = np.zeros_like(corrupted_volume, dtype=np.float32)

        vol_min = gt_volume.min()
        vol_max = gt_volume.max()
        if vol_max > vol_min:
            gt_normalized = (gt_volume - vol_min) / (vol_max - vol_min)
            gt_normalized = gt_normalized * 2 - 1
            corrupted_normalized = (corrupted_volume - vol_min) / (vol_max - vol_min)
            corrupted_normalized = corrupted_normalized * 2 - 1
        else:
            gt_normalized = np.full_like(gt_volume, -1.0)
            corrupted_normalized = np.full_like(corrupted_volume, -1.0)

        mask = (noise_mask_volume > 0).astype(np.float32)

        D, H, W = gt_normalized.shape
        patch_size = 64
        stride = 32

        reconstructed = np.zeros_like(gt_normalized, dtype=np.float32)
        count_map = np.zeros_like(gt_normalized, dtype=np.float32)

        for d in range(0, D, stride):
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    d_end = min(d + patch_size, D)
                    h_end = min(h + patch_size, H)
                    w_end = min(w + patch_size, W)

                    corr_patch = corrupted_normalized[d:d_end, h:h_end, w:w_end]
                    mask_patch = mask[d:d_end, h:h_end, w:w_end]

                    pad_d = patch_size - (d_end - d)
                    pad_h = patch_size - (h_end - h)
                    pad_w = patch_size - (w_end - w)

                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        corr_patch = np.pad(corr_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=-1)
                        mask_patch = np.pad(mask_patch, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

                    corr_tensor = torch.from_numpy(corr_patch).unsqueeze(0).unsqueeze(0).to(self.device)
                    mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0).unsqueeze(0).to(self.device)

                    recon_tensor = self.get_model().reconstruct(corr_tensor, mask_tensor, use_ddim=True, ddim_steps=50)
                    recon_patch = recon_tensor.cpu().numpy()[0, 0]

                    recon_patch = recon_patch[:d_end - d, :h_end - h, :w_end - w]

                    reconstructed[d:d_end, h:h_end, w:w_end] += recon_patch
                    count_map[d:d_end, h:h_end, w:w_end] += 1.0

                    del corr_tensor, mask_tensor, recon_tensor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        count_map = np.maximum(count_map, 1.0)
        reconstructed = reconstructed / count_map

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
        print("STAGE 2: LATENT DIFFUSION TRAINING")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Multi-GPU: {self.use_multi_gpu}")
        if self.use_multi_gpu:
            print(f"GPU IDs: {self.gpu_ids}")
            print(f"DataParallel Device IDs: {self.model.device_ids}")
        print(f"Mixed Precision (AMP): {self.use_amp}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {self.config['training']['batch_size']}")
        print(f"Learning Rate: {self.config['optimizer']['lr']}")
        print(f"Checkpoint Interval: {self.config['training'].get('checkpoint_interval', 10)}")
        print(f"Save Directory: {self.save_dir}")
        print(f"Training Samples: {len(train_loader.dataset)}")
        print(f"Validation Samples: {len(val_loader.dataset)}")
        print(f"Training Batches: {len(train_loader)}")
        print(f"Validation Batches: {len(val_loader)}")
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
                print(f"\n{'='*80}")
                print(f"Epoch {epoch}/{num_epochs} - NEW BEST MODEL!")
                print(f"  Train Loss: {train_losses['total_loss']:.4f}")
                print(f"  Val Loss: {val_losses['total_loss']:.4f} (Best: {self.best_loss:.4f})")
                print(f"  Diff: {val_losses['diffusion_loss']:.4f}")
                print(f"{'='*80}\n")
            else:
                print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_losses['total_loss']:.4f}, Val Loss: {val_losses['total_loss']:.4f} (Best: {self.best_loss:.4f})")
            
            self.save_checkpoint(is_best=is_best)
            
            history_data = {
                'train': self.train_history,
                'val': self.val_history,
                'best_loss': self.best_loss,
                'best_epoch': self.train_history[self.val_history.index(min(self.val_history, key=lambda x: x['total_loss']))]['epoch'] if self.val_history else 0,
                'current_epoch': epoch,
                'config': self.config
            }
            
            with open(self.log_dir / 'training_history.json', 'w') as f:
                json.dump(history_data, f, indent=2)
        
        print("\n" + "="*80)
        print("STAGE 2 TRAINING COMPLETE")
        print("="*80)
        print(f"Total Epochs: {num_epochs}")
        print(f"Best Validation Loss: {self.best_loss:.6f}")
        best_epoch_idx = self.val_history.index(min(self.val_history, key=lambda x: x['total_loss']))
        print(f"Best Epoch: {self.val_history[best_epoch_idx]['epoch']}")
        print(f"All files saved in: {self.save_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = LatentDiffusionTrainer(config)
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    train_loader, val_loader = get_optimized_dataloaders(
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
