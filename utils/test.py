import torch
import nibabel as nib
import yaml
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model
from utils.dataloader import MRIDataset
from torch.utils.data import DataLoader


def compute_metrics(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    pred_mean = np.mean(pred)
    gt_mean = np.mean(gt)
    pred_std = np.std(pred)
    gt_std = np.std(gt)
    
    covariance = np.mean((pred - pred_mean) * (gt - gt_mean))
    ssim = (2 * pred_mean * gt_mean + 1e-8) * (2 * covariance + 1e-8) / \
           ((pred_mean**2 + gt_mean**2 + 1e-8) * (pred_std**2 + gt_std**2 + 1e-8))
    
    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim
    }


def test(config_path, checkpoint_path, output_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = MRIDataset(
        config['data']['preprocessed_dir'],
        split='val',
        val_split=config['data']['val_split']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = create_model(config['model']).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_metrics = []
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, (corrupted, gt, gt_mask) in enumerate(test_loader):
            corrupted = corrupted.to(device)
            gt = gt.to(device)
            
            clean_volume, predicted_mask = model(corrupted)
            
            clean_np = clean_volume.cpu().numpy()[0, 0]
            gt_np = gt.cpu().numpy()[0, 0]
            
            metrics = compute_metrics(clean_np, gt_np)
            all_metrics.append(metrics)
            
            if output_dir:
                clean_nii = nib.Nifti1Image(clean_np, affine=np.eye(4))
                nib.save(clean_nii, output_path / f'reconstructed_{idx:04d}.nii.gz')
                
                mask_np = predicted_mask.cpu().numpy()[0, 0]
                mask_nii = nib.Nifti1Image(mask_np, affine=np.eye(4))
                nib.save(mask_nii, output_path / f'mask_{idx:04d}.nii.gz')
    
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    print("Test Results:")
    print(f"  MSE: {avg_metrics['mse']:.6f}")
    print(f"  PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {avg_metrics['ssim']:.4f}")
    
    return avg_metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/model_config.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', default=None)
    args = parser.parse_args()
    
    test(args.config, args.checkpoint, args.output_dir)
