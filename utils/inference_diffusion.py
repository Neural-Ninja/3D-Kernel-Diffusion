import torch
from pathlib import Path
import yaml
import argparse
import nibabel as nib
import os
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.latent_diffusion import LatentDiffusionModel
from src.losses.metrics import compute_metrics


class LatentDiffusionInference:
    def __init__(self, config, checkpoint_path):
        self.config = config
        
        if torch.cuda.is_available() and config.get('device') == 'cuda':
            self.gpu_ids = config.get('gpu_ids', None)
            if self.gpu_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
                self.device = torch.device('cuda:0')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        model_config = config['model']
        self.model = LatentDiffusionModel(
            vae_in_channels=model_config.get('vae_in_channels', 1),
            vae_out_channels=model_config.get('vae_out_channels', 1),
            vae_n_hiddens=model_config.get('vae_n_hiddens', 128),
            vae_downsample=tuple(model_config.get('vae_downsample', [4, 4, 4])),
            vae_num_groups=model_config.get('vae_num_groups', 32),
            vae_embedding_dim=model_config.get('vae_embedding_dim', 256),
            vae_n_codes=model_config.get('vae_n_codes', 512),
            vae_checkpoint=model_config.get('vae_checkpoint', None),
            freeze_vae=model_config.get('freeze_vae', True),
            unet_model_channels=model_config.get('unet_model_channels', 128),
            unet_channel_mult=tuple(model_config.get('unet_channel_mult', [1, 2, 4])),
            unet_attention_levels=model_config.get('unet_attention_levels', [1, 2]),
            unet_num_heads=model_config.get('unet_num_heads', 4),
            unet_time_emb_dim=model_config.get('unet_time_emb_dim', 512),
            unet_context_dim=model_config.get('unet_context_dim', 256),
            diffusion_timesteps=model_config.get('diffusion_timesteps', 1000),
            diffusion_beta_schedule=model_config.get('diffusion_beta_schedule', 'cosine'),
            device=self.device
        )
        
        self.load_checkpoint(checkpoint_path)
        
        self.use_ddim = config['inference'].get('use_ddim', True)
        self.ddim_steps = config['inference'].get('ddim_steps', 50)
        self.output_dir = Path(config['inference']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.inference_use_patches = config['inference'].get('use_patches', config['data'].get('use_patches', False))
        self.inference_patch_size = tuple(config['inference'].get('patch_size', config['data'].get('patch_size', [64, 64, 64])))
        self.inference_stride = tuple(config['inference'].get('stride', config['data'].get('stride', self.inference_patch_size)))
        
        print(f"Inference setup complete:")
        print(f"  Device: {self.device}")
        print(f"  DDIM: {self.use_ddim} (steps={self.ddim_steps})")
        print(f"  Mode: {'Patch-based' if self.inference_use_patches else 'Full volume'}")
        if self.inference_use_patches:
            print(f"  Patch size: {self.inference_patch_size}")
            print(f"  Stride: {self.inference_stride}")
            overlap_pct = [(1 - s/p) * 100 for s, p in zip(self.inference_stride, self.inference_patch_size)]
            print(f"  Overlap: {overlap_pct[0]:.0f}% (D), {overlap_pct[1]:.0f}% (H), {overlap_pct[2]:.0f}% (W)")
        print(f"  Output dir: {self.output_dir}")
        print()
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Model moved to {self.device}")
    
    def load_nifti(self, filepath):
        img = nib.load(str(filepath))
        data = img.get_fdata()
        affine = img.affine
        return data, affine
    
    def save_nifti(self, data, affine, filepath):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, str(filepath))
    
    def normalize_volume(self, volume):
        vol_min = np.min(volume)
        vol_max = np.max(volume)
        
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
            volume = volume * 2 - 1
        else:
            volume = np.full_like(volume, -1.0)
        
        return volume, vol_min, vol_max
    
    def denormalize_volume(self, volume, vol_min, vol_max):
        volume = (volume + 1) / 2
        volume = volume * (vol_max - vol_min) + vol_min
        return volume
    
    def _get_slice_shape(self, volume_shape, axis):
        shape_list = list(volume_shape)
        shape_list.pop(axis)
        return tuple(shape_list)
    
    def _interpolate_slice(self, volume, axis, position):
        if position == 0:
            if axis == 0:
                return volume[0, :, :]
            elif axis == 1:
                return volume[:, 0, :]
            else:
                return volume[:, :, 0]
        elif position >= volume.shape[axis]:
            if axis == 0:
                return volume[-1, :, :]
            elif axis == 1:
                return volume[:, -1, :]
            else:
                return volume[:, :, -1]
        else:
            if axis == 0:
                before = volume[position-1, :, :] if position > 0 else volume[0, :, :]
                after = volume[position, :, :] if position < volume.shape[axis] else volume[-1, :, :]
            elif axis == 1:
                before = volume[:, position-1, :] if position > 0 else volume[:, 0, :]
                after = volume[:, position, :] if position < volume.shape[axis] else volume[:, -1, :]
            else:
                before = volume[:, :, position-1] if position > 0 else volume[:, :, 0]
                after = volume[:, :, position] if position < volume.shape[axis] else volume[:, :, -1]
            
            return (before + after) / 2.0
    
    def reconstruct_corrupted_volume(self, corrupted_volume, target_shape, missing_slices_dict):
        if not missing_slices_dict:
            return corrupted_volume
        
        reconstructed = corrupted_volume.copy()
        
        for axis, indices in missing_slices_dict.items():
            if not indices:
                continue
            
            sorted_indices = sorted(indices)
            for idx in sorted_indices:
                idx = int(idx)
                fill_slice = self._interpolate_slice(reconstructed, axis, idx)
                reconstructed = np.insert(reconstructed, idx, fill_slice, axis=axis)
        
        return reconstructed
    
    def create_mask_from_missing_slices(self, volume_shape, missing_slices_dict):
        mask = np.zeros(volume_shape, dtype=np.float32)
        
        if missing_slices_dict:
            for axis, indices in missing_slices_dict.items():
                for idx in indices:
                    idx = int(idx)
                    if axis == 0:
                        mask[idx, :, :] = 1.0
                    elif axis == 1:
                        mask[:, idx, :] = 1.0
                    else:
                        mask[:, :, idx] = 1.0
        
        return mask
    
    def reconstruct_noise_mask(self, noise_mask, gt_shape, missing_slices_dict):
        if not missing_slices_dict:
            return noise_mask
        
        reconstructed = noise_mask.copy()
        for axis, indices in missing_slices_dict.items():
            sorted_indices = sorted(indices)
            if sorted_indices:
                for idx in sorted_indices:
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed.shape, axis))
                    reconstructed = np.insert(reconstructed, int(idx), fill_slice, axis=axis)
        
        return reconstructed
    
    def create_unified_mask(self, corrupted_volume, damage_mask, noise_mask=None):
        background_mask = (corrupted_volume == 0)
        brain_mask = (corrupted_volume != 0)
        
        if noise_mask is not None:
            combined_damage = np.logical_or(damage_mask > 0, noise_mask > 0).astype(np.float32)
        else:
            combined_damage = damage_mask
        
        unified_mask = np.zeros_like(corrupted_volume, dtype=np.float32)
        unified_mask[background_mask] = 0.0
        unified_mask[brain_mask & (combined_damage == 0)] = 0.5
        unified_mask[brain_mask & (combined_damage > 0)] = 1.0
        
        return unified_mask
    
    def extract_patches(self, volume, patch_size, stride):
        D, H, W = volume.shape
        pd, ph, pw = patch_size
        sd, sh, sw = stride
        
        patches = []
        positions = []
        
        d_positions = list(range(0, D - pd + 1, sd))
        h_positions = list(range(0, H - ph + 1, sh))
        w_positions = list(range(0, W - pw + 1, sw))
        
        if d_positions[-1] + pd < D:
            d_positions.append(D - pd)
        if h_positions[-1] + ph < H:
            h_positions.append(H - ph)
        if w_positions[-1] + pw < W:
            w_positions.append(W - pw)
        
        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    patch = volume[d:d+pd, h:h+ph, w:w+pw]
                    patches.append(patch)
                    positions.append((d, h, w))
        
        return patches, positions
    
    def reconstruct_from_patches(self, patches, positions, volume_shape, patch_size, stride):
        D, H, W = volume_shape
        pd, ph, pw = patch_size
        
        reconstructed = np.zeros(volume_shape, dtype=np.float32)
        weight_map = np.zeros(volume_shape, dtype=np.float32)
        
        for patch, (d, h, w) in zip(patches, positions):
            reconstructed[d:d+pd, h:h+ph, w:w+pw] += patch
            weight_map[d:d+pd, h:h+ph, w:w+pw] += 1.0
        
        weight_map[weight_map == 0] = 1.0
        reconstructed = reconstructed / weight_map
        
        return reconstructed
    
    @torch.no_grad()
    def reconstruct_volume(self, corrupted_volume, missing_slices_dict, noise_mask_volume=None, gt_volume=None, affine=None):
        if gt_volume is not None:
            target_shape = gt_volume.shape
        else:
            target_shape = list(corrupted_volume.shape)
            for axis, indices in missing_slices_dict.items():
                target_shape[axis] += len(indices)
            target_shape = tuple(target_shape)
        
        corrupted_volume = self.reconstruct_corrupted_volume(
            corrupted_volume, target_shape, missing_slices_dict
        )
        
        if noise_mask_volume is not None:
            noise_mask_volume = self.reconstruct_noise_mask(noise_mask_volume, target_shape, missing_slices_dict)
        
        corrupted_norm, corr_min, corr_max = self.normalize_volume(corrupted_volume)
        
        if noise_mask_volume is not None:
            mask = (noise_mask_volume > 0).astype(np.float32)
        else:
            mask = np.zeros_like(corrupted_volume, dtype=np.float32)
        
        if self.inference_use_patches:
            corr_patches, positions = self.extract_patches(corrupted_norm, self.inference_patch_size, self.inference_stride)
            mask_patches, _ = self.extract_patches(mask, self.inference_patch_size, self.inference_stride)
            
            reconstructed_patches = []
            
            for corr_patch, mask_patch in tqdm(zip(corr_patches, mask_patches), total=len(corr_patches), desc="Reconstructing patches"):
                x_corr = torch.from_numpy(corr_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                m_mask = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                
                x_pred = self.model.reconstruct(
                    x_corr, m_mask,
                    use_ddim=self.use_ddim,
                    ddim_steps=self.ddim_steps
                )
                
                x_pred_np = x_pred.squeeze().cpu().numpy()
                reconstructed_patches.append(x_pred_np)
            
            reconstructed_norm = self.reconstruct_from_patches(
                reconstructed_patches, positions, corrupted_volume.shape, self.inference_patch_size, self.inference_stride
            )
        else:
            x_corr = torch.from_numpy(corrupted_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            m_mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            
            x_pred = self.model.reconstruct(
                x_corr, m_mask,
                use_ddim=self.use_ddim,
                ddim_steps=self.ddim_steps
            )
            
            reconstructed_norm = x_pred.squeeze().cpu().numpy()
        
        reconstructed = self.denormalize_volume(reconstructed_norm, corr_min, corr_max)
        
        metrics = {}
        if gt_volume is not None:
            gt_tensor = torch.from_numpy(gt_volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            recon_tensor = torch.from_numpy(reconstructed.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            
            torch_metrics = compute_metrics(recon_tensor, gt_tensor)
            metrics['ssim'] = torch_metrics['ssim']
            metrics['psnr'] = torch_metrics['psnr']
            
            mse = np.mean((reconstructed - gt_volume) ** 2)
            metrics['mse'] = float(mse)
            mae = np.mean(np.abs(reconstructed - gt_volume))
            metrics['mae'] = float(mae)
            
            print(f"\n{'='*60}")
            print(f"RECONSTRUCTION METRICS:")
            print(f"{'='*60}")
            print(f"SSIM:  {metrics['ssim']:.4f}")
            print(f"PSNR:  {metrics['psnr']:.2f} dB")
            print(f"MSE:   {metrics['mse']:.6f}")
            print(f"MAE:   {metrics['mae']:.6f}")
            print(f"{'='*60}\n")
        
        results = {
            'reconstructed': reconstructed,
            'corrupted': corrupted_volume,
            'mask': mask,
            'affine': affine,
            'metrics': metrics
        }
        
        if gt_volume is not None:
            results['ground_truth'] = gt_volume
        
        return results
    
    def process_single_volume(self, corrupted_path, missing_slices_dict, noise_mask_path=None, gt_path=None, output_name=None):
        if output_name is None:
            output_name = Path(corrupted_path).stem.replace('.nii', '')
        
        corrupted_volume, affine = self.load_nifti(corrupted_path)
        noise_mask_volume = None
        if noise_mask_path and Path(noise_mask_path).exists():
            noise_mask_volume, _ = self.load_nifti(noise_mask_path)
        gt_volume = None
        if gt_path:
            gt_volume, _ = self.load_nifti(gt_path)
        
        results = self.reconstruct_volume(corrupted_volume, missing_slices_dict, noise_mask_volume, gt_volume, affine)
        
        volume_output_dir = self.output_dir / output_name
        volume_output_dir.mkdir(exist_ok=True)
        
        self.save_nifti(
            results['reconstructed'],
            results['affine'],
            volume_output_dir / 'reconstructed.nii.gz'
        )
        
        self.save_nifti(
            results['corrupted'],
            results['affine'],
            volume_output_dir / 'corrupted.nii.gz'
        )
        
        self.save_nifti(
            results['mask'],
            results['affine'],
            volume_output_dir / 'mask.nii.gz'
        )
        
        if 'ground_truth' in results:
            self.save_nifti(
                results['ground_truth'],
                results['affine'],
                volume_output_dir / 'ground_truth.nii.gz'
            )
        
        if results['metrics']:
            with open(volume_output_dir / 'metrics.json', 'w') as f:
                json.dump(results['metrics'], f, indent=2)
        
        return results
    
    def process_dataset(self, data_dir):
        data_path = Path(data_dir)
        corrupted_dir = data_path / 'corrupted'
        metadata_dir = data_path / 'metadata'
        gt_dir = data_path / 'gt'
        noise_mask_dir = data_path / 'noise-mask'
        
        corrupted_files = sorted(list(corrupted_dir.glob('*.nii.gz')))
        
        if not corrupted_files:
            return
        
        all_metrics = []
        
        for corrupted_path in corrupted_files:
            filename = corrupted_path.name
            metadata_path = metadata_dir / filename.replace('.nii.gz', '.json')
            gt_path = gt_dir / filename
            
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            missing_slices_dict = {}
            if 'slice_removal' in metadata:
                for axis_key, axis_data in metadata['slice_removal'].items():
                    axis = int(axis_key.split('_')[1])
                    missing_slices_dict[axis] = axis_data['indices']
            
            noise_mask_path = noise_mask_dir / filename
            noise_mask_arg = noise_mask_path if noise_mask_path.exists() else None
            gt_path_arg = gt_path if gt_path.exists() else None
            
            try:
                results = self.process_single_volume(
                    corrupted_path,
                    missing_slices_dict,
                    noise_mask_arg,
                    gt_path_arg,
                    output_name=filename.replace('.nii.gz', '')
                )
                
                if results['metrics']:
                    results['metrics']['filename'] = filename
                    all_metrics.append(results['metrics'])
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if all_metrics:
            summary_path = self.output_dir / 'summary_metrics.json'
            with open(summary_path, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            avg_metrics = {}
            for key in all_metrics[0].keys():
                if key != 'filename':
                    values = [m[key] for m in all_metrics]
                    avg_metrics[f'avg_{key}'] = sum(values) / len(values)
            
            with open(self.output_dir / 'average_metrics.json', 'w') as f:
                json.dump(avg_metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Inference with Latent Diffusion Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input volume or dataset directory')
    parser.add_argument('--missing_slices', type=str, default=None, help='Missing slices as JSON string or path to metadata.json file')
    parser.add_argument('--gt', type=str, default=None, help='Path to ground truth (optional)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = args.checkpoint if args.checkpoint else config['inference']['checkpoint']
    
    inference = LatentDiffusionInference(config, checkpoint_path)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        if not args.missing_slices:
            print("Error: --missing_slices required for single volume")
            print("Example: --missing_slices '{\"0\": [7, 66], \"1\": [80, 84], \"2\": [64, 102]}'")
            print("Or: --missing_slices /path/to/metadata.json")
            return
        
        if Path(args.missing_slices).exists():
            with open(args.missing_slices, 'r') as f:
                metadata = json.load(f)
            slice_removal = metadata.get('slice_removal', {})
            missing_slices_dict = {}
            for axis_key, axis_data in slice_removal.items():
                if isinstance(axis_key, str) and 'axis_' in axis_key:
                    axis = int(axis_key.split('_')[1])
                else:
                    axis = int(axis_key)
                if isinstance(axis_data, dict):
                    missing_slices_dict[axis] = axis_data.get('indices', [])
                else:
                    missing_slices_dict[axis] = axis_data
        else:
            missing_slices_dict = json.loads(args.missing_slices)
            missing_slices_dict = {int(k): v for k, v in missing_slices_dict.items()}
        
        inference.process_single_volume(
            args.input, missing_slices_dict, args.gt
        )
    
    elif input_path.is_dir():
        inference.process_dataset(args.input)


if __name__ == '__main__':
    main()
