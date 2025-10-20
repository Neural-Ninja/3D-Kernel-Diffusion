import torch
import nibabel as nib
import yaml
from pathlib import Path
import sys
import numpy as np
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model


def compute_metrics(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    pred_mean = np.mean(pred)
    gt_mean = np.mean(gt)
    pred_std = np.std(pred)
    gt_std = np.std(gt)
    covariance = np.mean((pred - pred_mean) * (gt - gt_mean))
    
    ssim = (2 * pred_mean * gt_mean + 1e-8) * (2 * covariance + 1e-8) / \
           ((pred_mean**2 + gt_mean**2 + 1e-8) * (pred_std**2 + gt_std**2 + 1e-8))
    
    return {'mse': mse, 'psnr': psnr, 'ssim': ssim}


def reconstruct_from_file(corrupted_path, gt_path=None, config_path='config/model_config.yaml'):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    checkpoint_path = config['inference']['checkpoint']
    num_steps = config['inference']['num_steps']
    output_dir = Path(config['inference']['output_dir'])
    save_mask = config['inference']['save_mask']
    save_corrupted = config['inference']['save_corrupted']
    
    model = create_model(config['model']).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    corrupted_nii = nib.load(corrupted_path)
    corrupted = corrupted_nii.get_fdata()
    affine = corrupted_nii.affine
    
    print(f"Corrupted shape: {corrupted.shape}")
    
    corrupted_tensor = torch.from_numpy(np.ascontiguousarray(corrupted, dtype=np.float32))
    corrupted_tensor = corrupted_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_mask = model.corruption_detector(corrupted_tensor)
        
        reconstructed = model.diffusion.p_sample_loop(
            model.diffusion_unet,
            corrupted_tensor.shape,
            corrupted_tensor,
            predicted_mask,
            device,
            num_inference_steps=num_steps
        )
    
    reconstructed_np = reconstructed.cpu().numpy()[0, 0]
    mask_probs = torch.sigmoid(predicted_mask).cpu().numpy()[0, 0]
    mask_np = (mask_probs > 0.5).astype(np.float32)
    
    print(f"Reconstructed shape: {reconstructed_np.shape}")
    print(f"Reconstructed range: [{reconstructed_np.min():.3f}, {reconstructed_np.max():.3f}]")
    print(f"Mask unique values: {np.unique(mask_np)}")
    
    if gt_path:
        gt_nii = nib.load(gt_path)
        gt_np = gt_nii.get_fdata()
        metrics = compute_metrics(reconstructed_np, gt_np)
        print(f"\nMSE: {metrics['mse']:.6f}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_name = Path(corrupted_path).stem.replace('.nii', '')
    
    recon_path = output_dir / f"{input_name}_reconstructed.nii.gz"
    reconstructed_nii = nib.Nifti1Image(reconstructed_np, affine)
    nib.save(reconstructed_nii, recon_path)
    
    if save_mask:
        mask_path = output_dir / f"{input_name}_mask.nii.gz"
        mask_nii = nib.Nifti1Image(mask_np, affine)
        nib.save(mask_nii, mask_path)
    
    if save_corrupted:
        corrupted_path_out = output_dir / f"{input_name}_corrupted.nii.gz"
        corrupted_nii_out = nib.Nifti1Image(corrupted, affine)
        nib.save(corrupted_nii_out, corrupted_path_out)
    
    if gt_path:
        gt_save_path = output_dir / f"{input_name}_gt.nii.gz"
        gt_save_nii = nib.Nifti1Image(gt_np, affine)
        nib.save(gt_save_nii, gt_save_path)
    
    return reconstructed_np, mask_np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--gt', default=None)
    parser.add_argument('--config', default='config/model_config.yaml')
    args = parser.parse_args()
    
    reconstruct_from_file(args.input, args.gt, args.config)
