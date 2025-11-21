import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
import yaml
import argparse
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.slice_predictor import create_slice_pair_predictor
from src.models.noise_mask_predictor import create_noise_mask_predictor
from src.models.latent_diffusion import LatentDiffusionModel


class CompleteMRIReconstructor:
    def __init__(self, slice_config, noise_config, diffusion_config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("Loading models...")
        
        print("  - Loading Slice Gap Predictor...")
        self.slice_predictor = create_slice_pair_predictor(slice_config['model'])
        slice_checkpoint = torch.load(slice_config['inference']['checkpoint'], 
                                     map_location=self.device)
        self.slice_predictor.load_state_dict(slice_checkpoint['model_state_dict'])
        self.slice_predictor = self.slice_predictor.to(self.device)
        self.slice_predictor.eval()
        
        print("  - Loading Noise Mask Predictor...")
        self.noise_predictor = create_noise_mask_predictor(noise_config['model'])
        noise_checkpoint = torch.load(noise_config['inference']['checkpoint'], 
                                     map_location=self.device)
        self.noise_predictor.load_state_dict(noise_checkpoint['model_state_dict'])
        self.noise_predictor = self.noise_predictor.to(self.device)
        self.noise_predictor.eval()
        
        print("  - Loading Latent Diffusion Model...")
        model_config = diffusion_config['model']
        self.diffusion_model = LatentDiffusionModel(
            vae_in_channels=model_config.get('vae_in_channels', 2),
            vae_out_channels=model_config.get('vae_out_channels', 1),
            vae_latent_channels=model_config.get('vae_latent_channels', 64),
            vae_base_channels=model_config.get('vae_base_channels', 64),
            unet_model_channels=model_config.get('unet_model_channels', 128),
            unet_channel_mult=tuple(model_config.get('unet_channel_mult', [1, 2, 4])),
            unet_attention_levels=model_config.get('unet_attention_levels', [1, 2]),
            unet_num_heads=model_config.get('unet_num_heads', 4),
            diffusion_timesteps=model_config.get('diffusion_timesteps', 1000),
            diffusion_beta_schedule=model_config.get('diffusion_beta_schedule', 'cosine'),
            device=self.device
        )
        
        diffusion_checkpoint = torch.load(diffusion_config['inference']['checkpoint'], 
                                         map_location=self.device)
        
        state_dict = diffusion_checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.diffusion_model.load_state_dict(state_dict)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.diffusion_model.eval()
        
        self.use_ddim = diffusion_config['inference'].get('use_ddim', True)
        self.ddim_steps = diffusion_config['inference'].get('ddim_steps', 100)
        self.noise_threshold = noise_config['inference'].get('threshold', 0.5)
        
        print("All models loaded successfully!\n")
    
    @torch.no_grad()
    def predict_slice_gaps(self, volume):
        print("Predicting missing slices...")
        
        B, C, D, H, W = volume.shape
        slice_mask = torch.zeros_like(volume)
        
        for axis in range(3):
            print(f"  - Checking axis {axis} ({'depth' if axis==0 else 'height' if axis==1 else 'width'})...")
            
            if axis == 0:
                for i in range(D - 1):
                    slice1 = volume[:, :, i, :, :]
                    slice2 = volume[:, :, i+1, :, :]
                    
                    if slice1.shape[-2:] != (256, 256):
                        slice1_resized = F.interpolate(slice1, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                        slice2_resized = F.interpolate(slice2, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                    else:
                        slice1_resized = slice1
                        slice2_resized = slice2
                    
                    axis_tensor = torch.tensor([axis], device=self.device)
                    
                    binary_logits, gap_value = self.slice_predictor(
                        slice1_resized, slice2_resized, axis_tensor
                    )
                    
                    has_gap = torch.argmax(binary_logits, dim=1).item()
                    
                    if has_gap and gap_value is not None:
                        n_missing = int(torch.clamp(gap_value, min=0).round().item())
                        for j in range(1, min(n_missing + 1, D - i)):
                            slice_mask[:, :, i+j, :, :] = 1.0
            
            elif axis == 1:
                for i in range(H - 1):
                    slice1 = volume[:, :, :, i, :]
                    slice2 = volume[:, :, :, i+1, :]
                    
                    slice1 = slice1.permute(0, 1, 3, 2)
                    slice2 = slice2.permute(0, 1, 3, 2)
                    
                    if slice1.shape[-2:] != (256, 256):
                        slice1_resized = F.interpolate(slice1, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                        slice2_resized = F.interpolate(slice2, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                    else:
                        slice1_resized = slice1
                        slice2_resized = slice2
                    
                    axis_tensor = torch.tensor([axis], device=self.device)
                    
                    binary_logits, gap_value = self.slice_predictor(
                        slice1_resized, slice2_resized, axis_tensor
                    )
                    
                    has_gap = torch.argmax(binary_logits, dim=1).item()
                    
                    if has_gap and gap_value is not None:
                        n_missing = int(torch.clamp(gap_value, min=0).round().item())
                        for j in range(1, min(n_missing + 1, H - i)):
                            slice_mask[:, :, :, i+j, :] = 1.0
            
            else:
                for i in range(W - 1):
                    slice1 = volume[:, :, :, :, i]
                    slice2 = volume[:, :, :, :, i+1]
                    
                    if slice1.shape[-2:] != (256, 256):
                        slice1_resized = F.interpolate(slice1, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                        slice2_resized = F.interpolate(slice2, size=(256, 256), 
                                                       mode='bilinear', align_corners=False)
                    else:
                        slice1_resized = slice1
                        slice2_resized = slice2
                    
                    axis_tensor = torch.tensor([axis], device=self.device)
                    
                    binary_logits, gap_value = self.slice_predictor(
                        slice1_resized, slice2_resized, axis_tensor
                    )
                    
                    has_gap = torch.argmax(binary_logits, dim=1).item()
                    
                    if has_gap and gap_value is not None:
                        n_missing = int(torch.clamp(gap_value, min=0).round().item())
                        for j in range(1, min(n_missing + 1, W - i)):
                            slice_mask[:, :, :, :, i+j] = 1.0
        
        n_missing_slices = slice_mask.sum().item()
        print(f"  Found {int(n_missing_slices)} missing slice locations\n")
        
        return slice_mask
    
    @torch.no_grad()
    def predict_noise_mask(self, volume):
        print("Predicting noise mask...")
        
        noise_mask, _ = self.noise_predictor(volume)
        noise_mask_binary = (noise_mask > self.noise_threshold).float()
        
        n_corrupted_voxels = noise_mask_binary.sum().item()
        total_voxels = noise_mask_binary.numel()
        corruption_ratio = n_corrupted_voxels / total_voxels * 100
        
        print(f"  Detected {int(n_corrupted_voxels)} corrupted voxels ({corruption_ratio:.2f}%)\n")
        
        return noise_mask_binary
    
    @torch.no_grad()
    def reconstruct(self, volume):
        print("="*80)
        print("STARTING RECONSTRUCTION PIPELINE")
        print("="*80 + "\n")
        
        slice_mask = self.predict_slice_gaps(volume)
        
        noise_mask = self.predict_noise_mask(volume)
        
        print("Combining masks...")
        combined_mask = torch.clamp(slice_mask + noise_mask, 0, 1)
        
        total_corruption = combined_mask.sum().item() / combined_mask.numel() * 100
        print(f"  Total corruption: {total_corruption:.2f}%\n")
        
        print("Running diffusion model reconstruction...")
        print(f"  Using {'DDIM' if self.use_ddim else 'DDPM'} sampling")
        print(f"  Steps: {self.ddim_steps if self.use_ddim else 1000}\n")
        
        reconstructed = self.diffusion_model.reconstruct(
            volume, 
            combined_mask,
            use_ddim=self.use_ddim,
            ddim_steps=self.ddim_steps
        )
        
        print("="*80)
        print("RECONSTRUCTION COMPLETE")
        print("="*80 + "\n")
        
        masks = {
            'slice_mask': slice_mask,
            'noise_mask': noise_mask,
            'combined_mask': combined_mask
        }
        
        return reconstructed, masks
    
    def load_nifti(self, filepath):
        img = nib.load(str(filepath))
        data = img.get_fdata().astype(np.float32)
        affine = img.affine
        return data, affine
    
    def save_nifti(self, data, affine, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        img = nib.Nifti1Image(data, affine)
        nib.save(img, str(filepath))
        print(f"Saved: {filepath}")
    
    def process_file(self, input_path, output_dir):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing: {input_path.name}")
        print("-" * 80)
        
        volume, affine = self.load_nifti(input_path)
        
        if volume.max() > volume.min():
            volume = (volume - volume.min()) / (volume.max() - volume.min())
        
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)
        
        print(f"Input shape: {volume_tensor.shape}\n")
        
        reconstructed, masks = self.reconstruct(volume_tensor)
        
        reconstructed_np = reconstructed.cpu().squeeze().numpy()
        
        base_name = input_path.stem.replace('.nii', '')
        
        self.save_nifti(reconstructed_np, affine, 
                       output_dir / f"{base_name}_reconstructed.nii.gz")
        
        self.save_nifti(masks['slice_mask'].cpu().squeeze().numpy(), affine,
                       output_dir / f"{base_name}_slice_mask.nii.gz")
        
        self.save_nifti(masks['noise_mask'].cpu().squeeze().numpy(), affine,
                       output_dir / f"{base_name}_noise_mask.nii.gz")
        
        self.save_nifti(masks['combined_mask'].cpu().squeeze().numpy(), affine,
                       output_dir / f"{base_name}_combined_mask.nii.gz")
        
        metadata = {
            'input_file': str(input_path),
            'input_shape': list(volume.shape),
            'slice_corruption': float(masks['slice_mask'].sum().item() / masks['slice_mask'].numel() * 100),
            'noise_corruption': float(masks['noise_mask'].sum().item() / masks['noise_mask'].numel() * 100),
            'total_corruption': float(masks['combined_mask'].sum().item() / masks['combined_mask'].numel() * 100),
            'reconstruction_method': 'DDIM' if self.use_ddim else 'DDPM',
            'ddim_steps': self.ddim_steps if self.use_ddim else 1000
        }
        
        with open(output_dir / f"{base_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAll outputs saved to: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description='Complete MRI Reconstruction Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Path to corrupted MRI file')
    parser.add_argument('--slice_config', type=str, default='config/slice.yaml',
                       help='Path to slice predictor config')
    parser.add_argument('--noise_config', type=str, default='config/noise_predictor.yaml',
                       help='Path to noise predictor config')
    parser.add_argument('--diffusion_config', type=str, default='config/diffusion.yaml',
                       help='Path to diffusion model config')
    parser.add_argument('--output_dir', type=str, default='outputs/complete_reconstruction',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    with open(args.slice_config, 'r') as f:
        slice_config = yaml.safe_load(f)
    
    with open(args.noise_config, 'r') as f:
        noise_config = yaml.safe_load(f)
    
    with open(args.diffusion_config, 'r') as f:
        diffusion_config = yaml.safe_load(f)
    
    reconstructor = CompleteMRIReconstructor(
        slice_config, noise_config, diffusion_config, device=args.device
    )
    
    reconstructor.process_file(args.input, args.output_dir)


if __name__ == '__main__':
    main()
