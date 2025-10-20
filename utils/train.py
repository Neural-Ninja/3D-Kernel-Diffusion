import torch
import yaml
from pathlib import Path
import sys
from torch.amp import autocast, GradScaler
import gc
from tqdm import tqdm
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.regularizers.loss import ReconstructionLoss
from utils.dataloader import get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp, clear_cache_freq, accumulation_steps, epoch):
    model.train()
    total_loss = 0
    metrics = {'diffusion': 0, 'mask': 0}
    
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch} [Train]')
    for batch_idx, (corrupted, gt, gt_mask) in pbar:
        corrupted = corrupted.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        gt_mask = gt_mask.to(device, non_blocking=True)
        
        if use_amp:
            with autocast('cuda'):
                predicted_noise, noise, predicted_mask = model(corrupted, gt)
                loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
            loss = loss / accumulation_steps
            scaler.scale(loss).backward()
        else:
            predicted_noise, noise, predicted_mask = model(corrupted, gt)
            loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
            loss = loss / accumulation_steps
            loss.backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        total_loss += loss.item() * accumulation_steps
        metrics['diffusion'] += loss_dict['diffusion']
        metrics['mask'] += loss_dict['mask']
        
        pbar.set_postfix({
            'loss': f"{loss.item() * accumulation_steps:.4f}",
            'diff': f"{loss_dict['diffusion']:.4f}",
            'mask': f"{loss_dict['mask']:.4f}"
        })
        
        del corrupted, gt, gt_mask, predicted_noise, noise, predicted_mask, loss
        
        if (batch_idx + 1) % clear_cache_freq == 0:
            torch.cuda.empty_cache()
    
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


def validate(model, dataloader, criterion, device, use_amp, clear_cache_freq, epoch, max_val_batches=None):
    model.eval()
    total_loss = 0
    metrics = {'diffusion': 0, 'mask': 0}
    
    num_batches = min(len(dataloader), max_val_batches) if max_val_batches else len(dataloader)
    pbar = tqdm(enumerate(dataloader), total=num_batches, desc=f'Epoch {epoch} [Val]')
    with torch.no_grad():
        for batch_idx, (corrupted, gt, gt_mask) in pbar:
            if max_val_batches and batch_idx >= max_val_batches:
                break
            corrupted = corrupted.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            gt_mask = gt_mask.to(device, non_blocking=True)
            
            if use_amp:
                with autocast('cuda'):
                    predicted_noise, noise, predicted_mask = model(corrupted, gt)
                    loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
            else:
                predicted_noise, noise, predicted_mask = model(corrupted, gt)
                loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
            
            total_loss += loss.item()
            metrics['diffusion'] += loss_dict['diffusion']
            metrics['mask'] += loss_dict['mask']
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'diff': f"{loss_dict['diffusion']:.4f}",
                'mask': f"{loss_dict['mask']:.4f}"
            })
            
            del corrupted, gt, gt_mask, predicted_noise, noise, predicted_mask, loss
            
            if (batch_idx + 1) % clear_cache_freq == 0:
                torch.cuda.empty_cache()
    
    n = batch_idx + 1
    return total_loss / n, {k: v / n for k, v in metrics.items()}


def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    gpu_ids = config['training'].get('gpu_ids', [0])
    use_data_parallel = config['training'].get('use_data_parallel', False)
    use_amp = config['training'].get('use_amp', False)
    clear_cache_freq = config['training'].get('clear_cache_freq', 5)
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    if torch.cuda.is_available():
        if use_data_parallel and len(gpu_ids) > 1:
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"Using DataParallel on GPUs: {gpu_ids}")
        else:
            device = torch.device(f'cuda:{gpu_ids[0]}')
            print(f"Using single GPU: {gpu_ids[0]}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
    
    print(f"\n{'='*60}")
    print(f"Mixed Precision: {use_amp}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Gradient Accumulation Steps: {accumulation_steps}")
    print(f"Effective Batch Size: {config['training']['batch_size'] * accumulation_steps}")
    print(f"Gradient Checkpointing: {config['training'].get('gradient_checkpointing', False)}")
    print(f"{'='*60}\n")
    
    train_loader, val_loader = get_dataloaders(config)
    
    use_checkpoint = config['training'].get('gradient_checkpointing', False)
    model = create_model(config['model'], use_checkpoint=use_checkpoint)
    
    if use_data_parallel and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    
    model = model.to(device)
    
    criterion = ReconstructionLoss(
        mask_weight=config['training']['loss_weights']['mask'],
        diffusion_weight=config['training']['loss_weights']['diffusion']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['optimizer']['learning_rate'],
        weight_decay=config['optimizer']['weight_decay'],
        betas=tuple(config['optimizer']['betas'])
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['scheduler']['min_lr']
    )
    
    scaler = GradScaler('cuda') if use_amp else None
    
    torch.cuda.empty_cache()
    gc.collect()
    
    best_val_loss = float('inf')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    max_val_batches = config['training'].get('max_val_batches', None)
    if max_val_batches:
        print(f"Validation limited to {max_val_batches} batches per epoch\n")
    
    print(f"Starting training for {config['training']['num_epochs']} epochs...\n")
    
    for epoch in range(config['training']['num_epochs']):
        train_start = time.time()
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, clear_cache_freq, accumulation_steps, epoch+1)
        train_time = time.time() - train_start
        
        val_start = time.time()
        val_loss, val_metrics = validate(model, val_loader, criterion, device, use_amp, clear_cache_freq, epoch+1, max_val_batches)
        val_time = time.time() - val_start
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train - Diffusion: {train_metrics['diffusion']:.4f}, Mask: {train_metrics['mask']:.4f}")
        print(f"  Val   - Diffusion: {val_metrics['diffusion']:.4f}, Mask: {val_metrics['mask']:.4f}")
        print(f"  Time - Train: {train_time/60:.1f}m | Val: {val_time/60:.1f}m | Total: {(train_time+val_time)/60:.1f}m")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            torch.cuda.reset_peak_memory_stats()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_state = model.module.state_dict() if use_data_parallel and len(gpu_ids) > 1 else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        
        if (epoch + 1) % 50 == 0:
            model_state = model.module.state_dict() if use_data_parallel and len(gpu_ids) > 1 else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"  ✓ Checkpoint saved at epoch {epoch+1}")
        
        print(f"{'-'*60}\n")


if __name__ == '__main__':
    train('config/model_config.yaml')
