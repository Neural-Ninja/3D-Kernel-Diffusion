import os
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.slice_predictor import create_slice_pair_predictor, create_patch_based_predictor


class SlicePairInference:
    def __init__(self, config, checkpoint_path):
        self.config = config
        
        if torch.cuda.is_available() and config.get('device') == 'cuda':
            self.gpu_ids = config.get('gpu_ids', None)
            if self.gpu_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, self.gpu_ids))
                self.device = torch.device('cuda:0')
                print(f"Using GPUs: {self.gpu_ids}")
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        model_config = config['model']
        
        self.model = create_slice_pair_predictor({
            'in_channels': model_config.get('in_channels', 1),
            'feature_size': model_config.get('feature_size', 48),
            'img_size': tuple(model_config.get('img_size', (256, 256))),
            'use_pretrained': False,
            'cross_attn_heads': model_config.get('cross_attn_heads', 4),
            'cross_attn_layers': model_config.get('cross_attn_layers', 2),
            'use_regression': model_config.get('use_regression', True),
            'axis_embed_dim': model_config.get('axis_embed_dim', 16)
        }).to(self.device)
        
        self.load_checkpoint(checkpoint_path)
        
        self.output_dir = Path(config['inference']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded model weights from {checkpoint_path}")
    
    def load_volume(self, volume_path):
        img = nib.load(str(volume_path))
        volume = img.get_fdata()
        affine = img.affine
        return volume, affine
    
    def normalize_slice(self, slice_data):
        if slice_data.max() > 0:
            slice_data = slice_data / slice_data.max()
        return slice_data
    
    def extract_consecutive_pairs(self, volume, axis):
        if axis == 0:
            n_slices = volume.shape[0]
            pairs = [(volume[i, :, :], volume[i+1, :, :]) for i in range(n_slices - 1)]
        elif axis == 1:
            n_slices = volume.shape[1]
            pairs = [(volume[:, i, :], volume[:, i+1, :]) for i in range(n_slices - 1)]
        else:
            n_slices = volume.shape[2]
            pairs = [(volume[:, :, i], volume[:, :, i+1]) for i in range(n_slices - 1)]
        
        return pairs
    
    def predict_gaps_for_axis(self, volume, axis):
        pairs = self.extract_consecutive_pairs(volume, axis)
        gap_predictions = []
        
        axis_tensor = torch.tensor([axis], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            for slice1, slice2 in pairs:
                slice1_norm = self.normalize_slice(slice1)
                slice2_norm = self.normalize_slice(slice2)
                
                slice1_tensor = torch.from_numpy(slice1_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                slice2_tensor = torch.from_numpy(slice2_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
                
                binary_logits, gap_value = self.model(slice1_tensor, slice2_tensor, axis_tensor)
                
                has_gap = torch.argmax(binary_logits, dim=1).item()
                if gap_value is not None and has_gap > 0:
                    predicted_spacing = max(0, gap_value.item())
                else:
                    predicted_spacing = 0
                
                gap_predictions.append({
                    'has_gap': has_gap,
                    'spacing': predicted_spacing
                })
        
        return gap_predictions
    
    def reconstruct_volume_from_predictions(self, volume, predictions_dict):
        reconstructed = volume.copy()
        reconstruction_mask = np.ones_like(volume)
        
        for axis, predictions in predictions_dict.items():
            if axis == 0:
                new_slices = []
                new_masks = []
                for i in range(volume.shape[0]):
                    new_slices.append(reconstructed[i, :, :])
                    new_masks.append(reconstruction_mask[i, :, :])
                    
                    if i < len(predictions):
                        n_gaps = int(round(predictions[i]['spacing']))
                        for _ in range(n_gaps):
                            new_slices.append(np.zeros_like(reconstructed[i, :, :]))
                            new_masks.append(np.zeros_like(reconstruction_mask[i, :, :]))
                
                if len(new_slices) > volume.shape[0]:
                    reconstructed = np.stack(new_slices, axis=0)
                    reconstruction_mask = np.stack(new_masks, axis=0)
            
            elif axis == 1:
                new_slices = []
                new_masks = []
                for i in range(reconstructed.shape[1]):
                    new_slices.append(reconstructed[:, i, :])
                    new_masks.append(reconstruction_mask[:, i, :])
                    
                    if i < len(predictions):
                        n_gaps = int(round(predictions[i]['spacing']))
                        for _ in range(n_gaps):
                            new_slices.append(np.zeros((reconstructed.shape[0], reconstructed.shape[2])))
                            new_masks.append(np.zeros((reconstruction_mask.shape[0], reconstruction_mask.shape[2])))
                
                if len(new_slices) > volume.shape[1]:
                    reconstructed = np.stack(new_slices, axis=1)
                    reconstruction_mask = np.stack(new_masks, axis=1)
            
            else:
                new_slices = []
                new_masks = []
                for i in range(reconstructed.shape[2]):
                    new_slices.append(reconstructed[:, :, i])
                    new_masks.append(reconstruction_mask[:, :, i])
                    
                    if i < len(predictions):
                        n_gaps = int(round(predictions[i]['spacing']))
                        for _ in range(n_gaps):
                            new_slices.append(np.zeros((reconstructed.shape[0], reconstructed.shape[1])))
                            new_masks.append(np.zeros((reconstruction_mask.shape[0], reconstruction_mask.shape[1])))
                
                if len(new_slices) > volume.shape[2]:
                    reconstructed = np.stack(new_slices, axis=2)
                    reconstruction_mask = np.stack(new_masks, axis=2)
        
        return reconstructed, reconstruction_mask
    
    def save_outputs(self, volume_name, reconstructed_volume, reconstruction_mask, affine):
        volume_output_dir = self.output_dir / volume_name
        volume_output_dir.mkdir(exist_ok=True)
        
        img_recon = nib.Nifti1Image(reconstructed_volume.astype(np.float32), affine)
        nib.save(img_recon, volume_output_dir / 'reconstructed_volume.nii.gz')
        
        img_mask = nib.Nifti1Image(reconstruction_mask.astype(np.float32), affine)
        nib.save(img_mask, volume_output_dir / 'reconstruction_mask.nii.gz')
        
        print(f"Saved reconstruction to {volume_output_dir}")
    
    def process_volume(self, volume_path):
        volume_name = Path(volume_path).stem.replace('.nii', '')
        print(f"\nProcessing: {volume_name}")
        
        volume, affine = self.load_volume(volume_path)
        print(f"  Volume shape: {volume.shape}")
        
        predictions_dict = {}
        
        for axis in range(3):
            axis_name = ['depth', 'height', 'width'][axis]
            print(f"  Predicting gaps along {axis_name} axis...")
            predictions = self.predict_gaps_for_axis(volume, axis)
            predictions_dict[axis] = predictions
            
            total_gaps = sum([p['spacing'] for p in predictions])
            print(f"    Total predicted gaps: {total_gaps:.1f}")
        
        print(f"  Reconstructing volume...")
        reconstructed_volume, reconstruction_mask = self.reconstruct_volume_from_predictions(
            volume, predictions_dict
        )
        
        print(f"  Reconstructed volume shape: {reconstructed_volume.shape}")
        
        self.save_outputs(
            volume_name,
            reconstructed_volume,
            reconstruction_mask,
            affine
        )
    
    def process_directory(self, input_dir):
        input_path = Path(input_dir)
        
        volume_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
        
        if not volume_files:
            print(f"No NIfTI files found in {input_dir}")
            return
        
        print(f"Found {len(volume_files)} volumes to process")
        
        for volume_file in tqdm(volume_files, desc="Processing volumes"):
            try:
                self.process_volume(volume_file)
            except Exception as e:
                print(f"Error processing {volume_file.name}: {str(e)}")
                continue


def main():
    parser = argparse.ArgumentParser(description='Inference with Slice Pair Predictor')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Path to input volume or directory')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = config['inference']['checkpoint']
    
    inference = SlicePairInference(config, checkpoint_path)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        inference.process_volume(input_path)
    elif input_path.is_dir():
        inference.process_directory(input_path)
    else:
        print(f"Error: {args.input} is not a valid file or directory")


if __name__ == '__main__':
    main()
