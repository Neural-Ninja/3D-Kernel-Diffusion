import torch
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from src.regularizers.loss import ReconstructionLoss
from utils.dataloader import get_dataloaders


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics = {'diffusion': 0, 'mask': 0}
    
    for corrupted, gt, gt_mask in dataloader:
        corrupted = corrupted.to(device)
        gt = gt.to(device)
        gt_mask = gt_mask.to(device)
        
        optimizer.zero_grad()
        
        predicted_noise, noise, predicted_mask = model(corrupted, gt)
        
        loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        metrics['diffusion'] += loss_dict['diffusion']
        metrics['mask'] += loss_dict['mask']
    
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'diffusion': 0, 'mask': 0}
    
    with torch.no_grad():
        for corrupted, gt, gt_mask in dataloader:
            corrupted = corrupted.to(device)
            gt = gt.to(device)
            gt_mask = gt_mask.to(device)
            
            predicted_noise, noise, predicted_mask = model(corrupted, gt)
            
            loss, loss_dict = criterion(predicted_noise, noise, predicted_mask, gt_mask)
            
            total_loss += loss.item()
            metrics['diffusion'] += loss_dict['diffusion']
            metrics['mask'] += loss_dict['mask']
    
    n = len(dataloader)
    return total_loss / n, {k: v / n for k, v in metrics.items()}


def train(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader = get_dataloaders(config)
    
    model = create_model(config['model']).to(device)
    
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
    
    best_val_loss = float('inf')
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Metrics: {train_metrics}")
            print(f"  Val Metrics: {val_metrics}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / 'best_model.pth')
        
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    train('config/model_config.yaml')
