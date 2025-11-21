import warnings
warnings.filterwarnings("ignore")
import yaml
import json
import random
import numpy as np
import nibabel as nib
from pathlib import Path
import scipy.ndimage as ndimage
import scipy.fftpack as fftpack
from typing import Tuple, Dict, List, Optional

class MRICorruption:
    def __init__(self, volume: np.ndarray, config_path: str):
        self.raw_volume = volume.astype(np.float32)
        
        vol_min = np.min(self.raw_volume)
        vol_max = np.max(self.raw_volume)

        if vol_max > vol_min:
            self.raw_volume = (self.raw_volume - vol_min) / (vol_max - vol_min)
        else:
            print("Warning: Volume has constant intensity")
        
        self.original_min = vol_min
        self.original_max = vol_max
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.corruption_severity = config.get('severity', 'moderate')
        self.slice_removal_prob = config.get('slice_removal_prob', 0.8)
        self.noise_prob = config.get('noise_prob', 0.7)
        self.frequency_corruption_prob = config.get('frequency_corruption_prob', 0.0)
        self.motion_artifact_prob = config.get('motion_artifact_prob', 0.0)
        self.bias_field_prob = config.get('bias_field_prob', 0.0)
        self.contiguous_slices = config.get('contiguous_slices', True)
        self.affine = np.array(config.get('affine', np.eye(4)))
        
        seed = config.get('seed', None)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if 'severity_configs' not in config:
            raise ValueError("severity_configs key not found in config YAML")
        
        self.severity_configs = config['severity_configs']

        self._set_severity_params()

    def _set_severity_params(self):
        config = self.severity_configs[self.corruption_severity]
        self.slice_removal_range = tuple(config['slice_removal_range'])
        self.noise_std_range = tuple(config['noise_std_range'])
        self.n_noise_regions_range = tuple(config['n_noise_regions'])
        self.motion_severity = config['motion_severity']
        self.bias_field_strength = config['bias_field_strength']
        self.freq_corruption_intensity = config['freq_corruption_intensity']

    def slices_removal(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        corrupted_volume = volume.copy()
        corruption_map = np.zeros_like(volume, dtype=np.float32)
        shape = volume.shape
        brain_mask = volume > 0

        slices_to_remove = {0: [], 1: [], 2: []}

        n_axes = random.choice([2, 2, 3])
        axes = random.sample([0, 1, 2], n_axes)
        n_slices_total = random.randint(*self.slice_removal_range)
        base_per_axis = n_slices_total // len(axes)
        remainder = n_slices_total % len(axes)
        slices_per_axis = [base_per_axis] * len(axes)
        for i in range(remainder):
            slices_per_axis[i] += 1

        for i, axis in enumerate(axes):
            n_slices = min(slices_per_axis[i], shape[axis] - 4)
            max_idx = shape[axis]
            if n_slices <= 0:
                continue
            available = list(range(2, max_idx - 2))
            selected = []
            while len(selected) < n_slices and available:
                idx = random.choice(available)
                selected.append(idx)
                available.remove(idx)
            slices_to_remove[axis].extend(sorted(selected))

        for axis, indices in slices_to_remove.items():
            for idx in indices:
                if axis == 0:
                    mask = brain_mask[idx, :, :]
                    corrupted_volume[idx, :, :] = np.where(mask, 0, corrupted_volume[idx, :, :])
                    corruption_map[idx, :, :] = np.where(mask, 1.0, corruption_map[idx, :, :])
                elif axis == 1:
                    mask = brain_mask[:, idx, :]
                    corrupted_volume[:, idx, :] = np.where(mask, 0, corrupted_volume[:, idx, :])
                    corruption_map[:, idx, :] = np.where(mask, 1.0, corruption_map[:, idx, :])
                else:
                    mask = brain_mask[:, :, idx]
                    corrupted_volume[:, :, idx] = np.where(mask, 0, corrupted_volume[:, :, idx])
                    corruption_map[:, :, idx] = np.where(mask, 1.0, corruption_map[:, :, idx])

        removed_info = {f'axis_{k}': {'indices': v, 'n_slices': len(v), 'axis_name': ['depth','height','width'][k]} for k,v in slices_to_remove.items() if v}
        return corrupted_volume, corruption_map, removed_info

    

    def add_noise(self, volume: np.ndarray, corruption_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List]:
        corrupted = volume.copy()
        shape = volume.shape
        noise_info = []
        brain_mask = volume > 0
        n_brain_voxels = np.sum(brain_mask)
        corruption_map_noise = np.zeros_like(volume, dtype=np.float32)
        
        brain_coords = np.argwhere(brain_mask)
        if len(brain_coords) == 0:
            return corrupted, corruption_map, noise_info
            
        brain_min = brain_coords.min(axis=0)
        brain_max = brain_coords.max(axis=0)
        brain_extent = brain_max - brain_min + 1
        
        n_noise_patches = random.randint(3, 8)
        
        for patch_idx in range(n_noise_patches):
            patch_center_size = [
                random.randint(brain_extent[0]//6, brain_extent[0]//3),
                random.randint(brain_extent[1]//6, brain_extent[1]//3),
                random.randint(brain_extent[2]//6, brain_extent[2]//3)
            ]
            
            center = [
                random.randint(brain_min[0] + patch_center_size[0], brain_max[0] - patch_center_size[0]),
                random.randint(brain_min[1] + patch_center_size[1], brain_max[1] - patch_center_size[1]),
                random.randint(brain_min[2] + patch_center_size[2], brain_max[2] - patch_center_size[2])
            ]
            
            patch_mask = np.zeros_like(volume, dtype=bool)
            
            for coord in brain_coords:
                dist = np.sqrt(
                    ((coord[0] - center[0]) / patch_center_size[0])**2 +
                    ((coord[1] - center[1]) / patch_center_size[1])**2 +
                    ((coord[2] - center[2]) / patch_center_size[2])**2
                )
                
                threshold = random.uniform(0.8, 1.2)
                if dist < threshold:
                    patch_mask[coord[0], coord[1], coord[2]] = True
            
            patch_mask = patch_mask & brain_mask
            
            if not np.any(patch_mask):
                continue
            
            noise_type = random.choice(['rician', 'spike', 'gaussian'])
            
            if noise_type == 'rician':
                base_noise_std = random.uniform(*self.noise_std_range)
                noise_real = np.random.normal(0, base_noise_std, size=shape)
                noise_imag = np.random.normal(0, base_noise_std, size=shape)
                noisy_magnitude = np.sqrt((corrupted + noise_real)**2 + noise_imag**2)
                
                corrupted = np.where(patch_mask, noisy_magnitude, corrupted)
                corruption_map_noise = np.where(patch_mask, 1.0, corruption_map_noise)
                
                noise_info.append({
                    'type': 'rician_patch',
                    'patch_id': patch_idx,
                    'std': float(base_noise_std),
                    'n_voxels': int(np.sum(patch_mask))
                })
                
            elif noise_type == 'spike':
                patch_coords = np.argwhere(patch_mask)
                n_spikes = random.randint(20, 100)
                n_actual_spikes = min(n_spikes, len(patch_coords))
                
                if n_actual_spikes > 0:
                    spike_indices = np.random.choice(len(patch_coords), n_actual_spikes, replace=False)
                    spike_positions = []
                    
                    for idx in spike_indices:
                        x, y, z = patch_coords[idx]
                        spike_value = corrupted[x, y, z] * random.uniform(2.0, 5.0)
                        corrupted[x, y, z] = spike_value
                        corruption_map_noise[x, y, z] = 1.0
                        
                        if len(spike_positions) < 5:
                            spike_positions.append([int(x), int(y), int(z)])
                    
                    noise_info.append({
                        'type': 'spike_patch',
                        'patch_id': patch_idx,
                        'n_spikes': n_actual_spikes,
                        'sample_positions': spike_positions
                    })
                    
            else:
                noise_std = random.uniform(*self.noise_std_range) * 1.5
                noise = np.random.normal(0, noise_std, size=shape)
                
                corrupted = np.where(patch_mask, corrupted + noise, corrupted)
                corruption_map_noise = np.where(patch_mask, 1.0, corruption_map_noise)
                
                noise_info.append({
                    'type': 'gaussian_patch',
                    'patch_id': patch_idx,
                    'std': float(noise_std),
                    'n_voxels': int(np.sum(patch_mask))
                })
        
        corruption_map = np.maximum(corruption_map, corruption_map_noise)
        return corrupted, corruption_map, noise_info
    


    def k_space_corruption(self, volume: np.ndarray, corruption_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        corrupted = volume.copy()
        corruption_types = []
        applied_corruptions = {}
        
        primary_type = random.choice(['lines_missing', 'low_freq_noise', 'spike_noise'])
        corruption_types.append(primary_type)
        
        if random.random() < 0.4:
            secondary_type = random.choice(['high_freq_loss', 'phase_corruption'])
            if secondary_type not in corruption_types:
                corruption_types.append(secondary_type)
        
        kspace = fftpack.fftn(corrupted)
        kspace_shift = fftpack.fftshift(kspace)
        shape = volume.shape
        
        k_corruption_map = np.zeros_like(volume, dtype=np.float32)
        
        for corruption_type in corruption_types:
            if corruption_type == 'lines_missing':
                n_lines = random.randint(8, 25)
                axis = random.choice([0, 1, 2])
                
                indices = sorted(random.sample(range(shape[axis]), min(n_lines, shape[axis])))
                
                for idx in indices:
                    if axis == 0:
                        kspace_shift[idx, :, :] = 0
                    elif axis == 1:
                        kspace_shift[:, idx, :] = 0
                    else:
                        kspace_shift[:, :, idx] = 0
                
                applied_corruptions['lines_missing'] = {
                    'axis': int(axis),
                    'n_lines': len(indices),
                    'indices': indices[:10]
                }
                
            elif corruption_type == 'low_freq_noise':
                center = np.array(shape) // 2
                radius = int(min(shape) // 5)
                
                noise_strength = self.freq_corruption_intensity * 0.15
                
                z_start = max(0, center[0] - radius)
                z_end = min(shape[0], center[0] + radius)
                y_start = max(0, center[1] - radius)
                y_end = min(shape[1], center[1] + radius)
                x_start = max(0, center[2] - radius)
                x_end = min(shape[2], center[2] + radius)
                
                for x in range(z_start, z_end):
                    for y in range(y_start, y_end):
                        for z in range(x_start, x_end):
                            dist = np.linalg.norm(np.array([x, y, z]) - center)
                            if dist < radius:
                                noise = np.random.normal(0, np.abs(kspace_shift[x, y, z]) * noise_strength)
                                kspace_shift[x, y, z] += noise
                                k_corruption_map[x, y, z] = 1.0
                
                applied_corruptions['low_freq_noise'] = {
                    'radius': int(radius),
                    'strength': float(noise_strength)
                }
            
            elif corruption_type == 'high_freq_loss':
                center = np.array(shape) // 2
                radius = int(min(shape) // 2.5)
                attenuation = 0.05 + (self.freq_corruption_intensity * 0.15)
                
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        for z in range(shape[2]):
                            dist = np.linalg.norm(np.array([x, y, z]) - center)
                            if dist > radius:
                                kspace_shift[x, y, z] *= attenuation
                                k_corruption_map[x, y, z] = 1.0
                
                applied_corruptions['high_freq_loss'] = {
                    'radius': int(radius),
                    'attenuation': float(attenuation)
                }
            
            elif corruption_type == 'spike_noise':
                n_spikes = random.randint(20, 80)
                spike_positions = []
                
                for _ in range(n_spikes):
                    x = random.randint(0, shape[0]-1)
                    y = random.randint(0, shape[1]-1)
                    z = random.randint(0, shape[2]-1)
                    spike_mag = random.uniform(8, 25)
                    kspace_shift[x, y, z] *= spike_mag
                    spike_positions.append([int(x), int(y), int(z)])
                
                applied_corruptions['spike_noise'] = {
                    'n_spikes': n_spikes,
                    'sample_positions': spike_positions[:5]
                }
            
            elif corruption_type == 'phase_corruption':
                axis = random.choice([0, 1, 2])
                n_lines = random.randint(5, 15)
                
                indices = random.sample(range(shape[axis]), min(n_lines, shape[axis]))
                phase_shift = random.uniform(0.1, 0.5) * np.pi
                
                for idx in indices:
                    if axis == 0:
                        kspace_shift[idx, :, :] *= np.exp(1j * phase_shift)
                    elif axis == 1:
                        kspace_shift[:, idx, :] *= np.exp(1j * phase_shift)
                    else:
                        kspace_shift[:, :, idx] *= np.exp(1j * phase_shift)
                
                applied_corruptions['phase_corruption'] = {
                    'axis': int(axis),
                    'n_lines': len(indices),
                    'phase_shift': float(phase_shift)
                }
        
        kspace_filtered = fftpack.ifftshift(kspace_shift)
        corrupted = np.abs(fftpack.ifftn(kspace_filtered))
        
        if np.max(corrupted) > 0:
            corrupted = corrupted / np.max(corrupted)
        
        corruption_map = np.maximum(corruption_map, k_corruption_map)
        
        metadata = {
            'types_applied': corruption_types,
            'details': applied_corruptions,
            'severity': float(np.mean(k_corruption_map))
        }
        
        return corrupted, corruption_map, metadata
    

    def add_motion_artifact(self, volume: np.ndarray, corruption_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        corrupted = volume.copy()
        original = volume.copy()  
        brain_mask = volume > 0
        
        motion_dir = random.choice([0, 1, 2])
        shift_amount = max(1, int(self.motion_severity * random.uniform(0.8, 1.5)))
        n_ghosts = random.randint(2, 4)
        
        ghost_weights = [0.6]
        remaining_weight = 0.4
        
        for i in range(n_ghosts):
            weight = remaining_weight / n_ghosts
            ghost_weights.append(weight)
            shift = shift_amount * (i + 1)
            shifted = np.roll(corrupted, shift, axis=motion_dir)
            corrupted = corrupted * (1 - weight) + shifted * weight
        
        sigma = [0, 0, 0]
        sigma[motion_dir] = self.motion_severity * 0.8
        blurred = ndimage.gaussian_filter(corrupted, sigma=sigma)
        
        corrupted = np.where(brain_mask, blurred, original)
        
        motion_map = np.where(brain_mask, 1.0, 0.0).astype(np.float32)
        corruption_map = np.maximum(corruption_map, motion_map)
        
        metadata = {
            'direction': int(motion_dir),
            'direction_name': ['depth', 'height', 'width'][motion_dir],
            'severity': float(self.motion_severity),
            'n_ghosts': n_ghosts,
            'shift_amount': shift_amount,
        }
        
        return corrupted, corruption_map, metadata
    

    def add_bias_field(self, volume: np.ndarray, corruption_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        corrupted = volume.copy()
        original = volume.copy()
        shape = volume.shape
        brain_mask = volume > 0
        
        x = np.linspace(-1, 1, shape[0])
        y = np.linspace(-1, 1, shape[1])
        z = np.linspace(-1, 1, shape[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        coeff_x2 = random.uniform(-0.6, 0.6)
        coeff_y2 = random.uniform(-0.6, 0.6)
        coeff_z2 = random.uniform(-0.6, 0.6)
        coeff_xy = random.uniform(-0.4, 0.4)
        coeff_yz = random.uniform(-0.4, 0.4)
        
        bias = 1 + self.bias_field_strength * (
            coeff_x2 * X**2 +
            coeff_y2 * Y**2 +
            coeff_z2 * Z**2 +
            coeff_xy * X * Y +
            coeff_yz * Y * Z
        )
        
        sigma = max(shape[0] // 12, 3)
        bias = ndimage.gaussian_filter(bias, sigma=sigma)
        
        biased = corrupted * bias
        corrupted = np.where(brain_mask, biased, original)
        
        brain_max = np.max(corrupted[brain_mask]) if np.any(brain_mask) else 1.0
        if brain_max > 0:
            corrupted = np.where(brain_mask, corrupted / brain_max, corrupted)

        bias_corruption = np.where(brain_mask, 1.0, 0.0).astype(np.float32)
        corruption_map = np.maximum(corruption_map, bias_corruption)
        
        metadata = {
            'strength': float(self.bias_field_strength),
            'mean_bias': float(np.mean(bias[brain_mask])) if np.any(brain_mask) else 1.0,
            'max_bias': float(np.max(bias[brain_mask])) if np.any(brain_mask) else 1.0,
            'min_bias': float(np.min(bias[brain_mask])) if np.any(brain_mask) else 1.0,
        }
        
        return corrupted, corruption_map, metadata
    
    def process(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        gt_normalized = self.raw_volume.copy()
        corrupted = gt_normalized.copy()
        noise_mask = np.zeros_like(gt_normalized, dtype=np.float32)
        slice_mask = np.zeros_like(gt_normalized, dtype=np.float32)
        
        metadata = {
            'applied_corruptions': [],
            'missing_slices': {}
        }
        
        if random.random() < self.slice_removal_prob:
            corrupted, slice_mask, removed_info = self.slices_removal(corrupted)
            metadata['slice_removal'] = removed_info
            metadata['applied_corruptions'].append('slice_removal')
            
            for axis_key, axis_data in removed_info.items():
                axis = int(axis_key.split('_')[1])
                metadata['missing_slices'][axis] = axis_data['indices']
        
        if random.random() < self.noise_prob:
            corrupted, noise_mask, noise_info = self.add_noise(corrupted, noise_mask)
            metadata['noise'] = noise_info
            metadata['applied_corruptions'].append('noise')
        
        if random.random() < self.frequency_corruption_prob:
            corrupted, noise_mask, freq_info = self.k_space_corruption(corrupted, noise_mask)
            metadata['frequency'] = freq_info
            metadata['applied_corruptions'].append('frequency')
        
        if random.random() < self.motion_artifact_prob:
            corrupted, noise_mask, motion_info = self.add_motion_artifact(corrupted, noise_mask)
            metadata['motion'] = motion_info
            metadata['applied_corruptions'].append('motion')
        
        if random.random() < self.bias_field_prob:
            corrupted, noise_mask, bias_info = self.add_bias_field(corrupted, noise_mask)
            metadata['bias_field'] = bias_info
            metadata['applied_corruptions'].append('bias_field')
        
        noise_mask_binary = (noise_mask >= 0.5).astype(np.float32)
        
        gt = gt_normalized * (self.original_max - self.original_min) + self.original_min
        corrupted = corrupted * (self.original_max - self.original_min) + self.original_min
        corrupted = np.clip(corrupted, self.original_min, self.original_max)
        
        total_voxels = np.prod(noise_mask_binary.shape)
        corrupted_voxels = np.sum(noise_mask_binary)
        corruption_percentage = (corrupted_voxels / total_voxels) * 100
        
        brain_mask = gt_normalized > 0
        brain_voxels = np.sum(brain_mask)
        corrupted_brain_voxels = np.sum(noise_mask_binary[brain_mask])
        brain_corruption_percentage = (corrupted_brain_voxels / brain_voxels * 100) if brain_voxels > 0 else 0
        
        metadata['corruption_severity'] = self.corruption_severity
        metadata['corruption_percentage'] = float(corruption_percentage)
        metadata['brain_corruption_percentage'] = float(brain_corruption_percentage)
        metadata['total_voxels'] = int(total_voxels)
        metadata['corrupted_voxels'] = int(corrupted_voxels)
        metadata['brain_voxels'] = int(brain_voxels)
        metadata['corrupted_brain_voxels'] = int(corrupted_brain_voxels)
        metadata['original_range'] = {
            'min': float(self.original_min),
            'max': float(self.original_max)
        }
        
        return gt, corrupted, noise_mask_binary, metadata
    

    def save_nifti(self, data: np.ndarray, filepath: str):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = data.astype(np.float32)
        img = nib.Nifti1Image(data, self.affine)
        nib.save(img, str(filepath))
    
    def process_and_save(self, output_dir: str, prefix: str = "mri"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gt, corrupted, noise_mask, metadata = self.process()
        
        removed_info = metadata.get('slice_removal', {})
        for axis_key, axis_data in removed_info.items():
            axis = int(axis_key.split('_')[1])
            removed_indices = sorted(axis_data['indices'])
            if removed_indices:
                corrupted = np.delete(corrupted, removed_indices, axis=axis)
                noise_mask = np.delete(noise_mask, removed_indices, axis=axis)
        
        self.save_nifti(gt, output_dir / f"{prefix}_gt.nii.gz")
        self.save_nifti(corrupted, output_dir / f"{prefix}_corrupted.nii.gz")
        self.save_nifti(noise_mask, output_dir / f"{prefix}_noise_mask.nii.gz")
        
        metadata_path = output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return gt, corrupted, noise_mask, metadata
    

class MRIDatasetCorruptor:
    def __init__(self, data_root: str, output_root: str):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def load_nifti(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(str(filepath))
        data = img.get_fdata()
        affine = img.affine
        return data, affine
    
    def corrupt_single_file(
        self,
        input_path: str,
        output_dir: str,
        corruption_severity: str = 'moderate',
        prefix: Optional[str] = None,
        seed: Optional[int] = None
    ):
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if prefix is None:
            prefix = input_path.stem.replace('.nii', '')
        
        volume, affine = self.load_nifti(input_path)
        
        corruptor = MRICorruption(
            volume,
            corruption_severity=corruption_severity,
            affine=affine,
            seed=seed
        )
        
        gt, corrupted, corruption_map, metadata = corruptor.process_and_save(
            output_dir=str(output_dir),
            prefix=prefix
        )
        
        return gt, corrupted, corruption_map, metadata


"""
Mask: 1 - Corrupted Region, 0 - Non-Corrupted Region
"""

