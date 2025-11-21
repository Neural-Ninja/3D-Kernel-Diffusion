import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
from functools import lru_cache


class DiffusionMRIDataset(Dataset):
    def __init__(self, data_dir, split='train', train_split=0.7, val_split=0.15, 
                 normalize=True, target_shape=None, patch_size=None, patches_per_volume=1):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / 'gt'
        self.corrupted_dir = self.data_dir / 'corrupted'
        self.metadata_dir = self.data_dir / 'metadata'
        self.noise_mask_dir = self.data_dir / 'noise-mask'
        
        self.normalize = normalize
        self.target_shape = target_shape
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        
        for dir_path in [self.gt_dir, self.corrupted_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        all_files = sorted([f.name for f in self.gt_dir.glob('*.nii.gz')])
        
        n_total = len(all_files)
        
        if n_total == 0:
            raise ValueError(f"No .nii.gz files found in {self.gt_dir}")
        
        train_end = int(n_total * train_split)
        val_end = train_end + int(n_total * val_split)
        
        if split == 'train':
            self.files = all_files[:train_end]
        elif split == 'val':
            self.files = all_files[train_end:val_end]
        elif split == 'test':
            self.files = all_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        print(f"Loaded {len(self.files)} volumes for {split} split")
    
    def __len__(self):
        return len(self.files)
    
    def load_nifti(self, filepath):
        img = nib.load(str(filepath))
        data = img.get_fdata()
        return data
    
    def normalize_volume(self, volume):
        vol_min = np.min(volume)
        vol_max = np.max(volume)
        
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        else:
            volume = np.zeros_like(volume)
        
        return volume
    
    def create_unified_mask(self, gt_volume, noise_mask_volume, metadata):
        background_mask = (gt_volume == 0)
        brain_mask = (gt_volume != 0)
        
        slice_removal_info = metadata.get('slice_removal', {})
        reconstructed_noise_mask = self.reconstruct_noise_mask(noise_mask_volume, gt_volume.shape, slice_removal_info)
        
        damage_mask = np.zeros_like(gt_volume, dtype=np.float32)
        if slice_removal_info:
            for axis_key, axis_data in slice_removal_info.items():
                axis = int(axis_key.split('_')[1])
                removed_indices = axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data
                
                if removed_indices:
                    removed_indices = np.array(removed_indices, dtype=int)
                    if axis == 0:
                        damage_mask[removed_indices, :, :] = 1.0
                    elif axis == 1:
                        damage_mask[:, removed_indices, :] = 1.0
                    else:
                        damage_mask[:, :, removed_indices] = 1.0
        
        combined_damage = np.logical_or(damage_mask > 0, reconstructed_noise_mask > 0).astype(np.float32)
        
        unified_mask = np.zeros_like(gt_volume, dtype=np.float32)
        unified_mask[background_mask] = 0.0
        unified_mask[brain_mask & (combined_damage == 0)] = 0.5
        unified_mask[brain_mask & (combined_damage > 0)] = 1.0
        
        return unified_mask
    
    def resize_volume(self, volume, target_shape):
        from scipy.ndimage import zoom
        
        zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
        resized = zoom(volume, zoom_factors, order=1)
        
        return resized
    
    def load_metadata(self, filename):
        metadata_path = self.metadata_dir / filename.replace('.nii.gz', '.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def reconstruct_corrupted_volume(self, corrupted_volume, gt_shape, metadata, fill_method='interpolate'):
        slice_removal_info = metadata.get('slice_removal', {})
        
        if not slice_removal_info:
            return corrupted_volume
        
        reconstructed_corrupted = corrupted_volume.copy()
        
        for axis_key, axis_data in slice_removal_info.items():
            if isinstance(axis_key, str) and 'axis_' in axis_key:
                axis = int(axis_key.split('_')[1])
            else:
                axis = int(axis_key)
            
            if isinstance(axis_data, dict):
                removed_indices = sorted(axis_data.get('indices', []))
            else:
                removed_indices = sorted(axis_data)
            
            if not removed_indices:
                continue
            
            for idx in removed_indices:
                idx = int(idx)
                if fill_method == 'zero':
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed_corrupted.shape, axis))
                elif fill_method == 'interpolate':
                    fill_slice = self._interpolate_slice(reconstructed_corrupted, axis, idx)
                else:
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed_corrupted.shape, axis))
                
                reconstructed_corrupted = np.insert(reconstructed_corrupted, idx, fill_slice, axis=axis)
        
        return reconstructed_corrupted
    
    def _get_slice_shape(self, volume_shape, axis):
        shape_list = list(volume_shape)
        shape_list.pop(axis)
        return tuple(shape_list)
    
    def reconstruct_noise_mask(self, noise_mask, gt_shape, slice_removal_info):
        if not slice_removal_info:
            return noise_mask
        
        reconstructed = noise_mask.copy()
        for axis_key, axis_data in slice_removal_info.items():
            axis = int(axis_key.split('_')[1]) if isinstance(axis_key, str) and 'axis_' in axis_key else int(axis_key)
            removed_indices = sorted(axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data)
            
            if removed_indices:
                for idx in removed_indices:
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed.shape, axis))
                    reconstructed = np.insert(reconstructed, int(idx), fill_slice, axis=axis)
        
        return reconstructed
    
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
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        
        gt_path = self.gt_dir / filename
        corrupted_path = self.corrupted_dir / filename
        noise_mask_path = self.noise_mask_dir / filename
        
        gt_volume = self.load_nifti(gt_path)
        corrupted_volume = self.load_nifti(corrupted_path)
        noise_mask_volume = self.load_nifti(noise_mask_path) if noise_mask_path.exists() else np.zeros_like(corrupted_volume)
        
        metadata = self.load_metadata(filename)
        
        corrupted_volume = self.reconstruct_corrupted_volume(
            corrupted_volume, gt_volume.shape, metadata, fill_method='interpolate'
        )
        
        if self.target_shape is not None:
            gt_volume = self.resize_volume(gt_volume, self.target_shape)
            corrupted_volume = self.resize_volume(corrupted_volume, self.target_shape)
        
        if self.normalize:
            gt_volume = self.normalize_volume(gt_volume)
            corrupted_volume = self.normalize_volume(corrupted_volume)
        
        unified_mask = self.create_unified_mask(gt_volume, noise_mask_volume, metadata)
        
        if self.patch_size is not None:
            patches = self._extract_patches(gt_volume, corrupted_volume, unified_mask)
            return patches
        
        x_gt = torch.from_numpy(gt_volume.astype(np.float32)).unsqueeze(0)
        x_corr = torch.from_numpy(corrupted_volume.astype(np.float32)).unsqueeze(0)
        m_mask = torch.from_numpy(unified_mask.astype(np.float32)).unsqueeze(0)
        
        return {
            'x_gt': x_gt,
            'x_corr': x_corr,
            'm_mask': m_mask,
            'filename': filename
        }
    
    def _extract_patches(self, gt_volume, corrupted_volume, unified_mask):
        D, H, W = gt_volume.shape
        pd, ph, pw = self.patch_size
        
        patches = []
        for _ in range(self.patches_per_volume):
            d_start = np.random.randint(0, max(1, D - pd + 1))
            h_start = np.random.randint(0, max(1, H - ph + 1))
            w_start = np.random.randint(0, max(1, W - pw + 1))
            
            gt_patch = gt_volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            corr_patch = corrupted_volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            mask_patch = unified_mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
            
            x_gt = torch.from_numpy(gt_patch.astype(np.float32)).unsqueeze(0)
            x_corr = torch.from_numpy(corr_patch.astype(np.float32)).unsqueeze(0)
            m_mask = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)
            
            patches.append({
                'x_gt': x_gt,
                'x_corr': x_corr,
                'm_mask': m_mask,
                'filename': filename
            })
        
        return patches[0] if len(patches) == 1 else patches


class PatchBasedDiffusionDataset(Dataset):
    def __init__(self, data_dir, split='train', train_split=0.7, val_split=0.15,
                 normalize=True, patch_size=(64, 64, 64), stride=None, cache_size=10, preload_all=False):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / 'gt'
        self.corrupted_dir = self.data_dir / 'corrupted'
        self.metadata_dir = self.data_dir / 'metadata'
        self.noise_mask_dir = self.data_dir / 'noise-mask'
        
        self.normalize = normalize
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.preload_all = preload_all
        
        self.cache_size = cache_size
        self.volume_cache = {}
        self.cache_order = []
        
        for dir_path in [self.gt_dir, self.corrupted_dir]:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        all_files = sorted([f.name for f in self.gt_dir.glob('*.nii.gz')])
        n_total = len(all_files)
        
        if n_total == 0:
            raise ValueError(f"No .nii.gz files found in {self.gt_dir}")
        
        train_end = int(n_total * train_split)
        val_end = train_end + int(n_total * val_split)
        
        if split == 'train':
            self.files = all_files[:train_end]
        elif split == 'val':
            self.files = all_files[train_end:val_end]
        elif split == 'test':
            self.files = all_files[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        self.patch_indices = []
        print(f"Computing patch indices for {len(self.files)} volumes...")
        
        gt_path = self.gt_dir / self.files[0]
        gt_volume = self.load_nifti(gt_path)
        D, H, W = gt_volume.shape
        print(f"Volume shape: {(D, H, W)}")
        
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        patch_positions = []
        for d in range(0, D - pd + 1, sd):
            for h in range(0, H - ph + 1, sh):
                for w in range(0, W - pw + 1, sw):
                    patch_positions.append((d, h, w))
        
        print(f"Patches per volume: {len(patch_positions)}")
        
        for file_idx, filename in enumerate(self.files):
            for d, h, w in patch_positions:
                self.patch_indices.append({
                    'file_idx': file_idx,
                    'filename': filename,
                    'd_start': d,
                    'h_start': h,
                    'w_start': w
                })
        
        print(f"Total patches for {split}: {len(self.patch_indices)}")
        
        if self.preload_all:
            print(f"Preloading all {len(self.files)} volumes into memory...")
            from tqdm import tqdm
            for filename in tqdm(self.files, desc="Loading volumes"):
                self._load_and_cache_volume(filename)
            print(f"Preloading complete! {len(self.volume_cache)} volumes cached.")
    
    def __len__(self):
        return len(self.patch_indices)
    
    def load_nifti(self, filepath):
        img = nib.load(str(filepath))
        data = img.get_fdata()
        return data
    
    def normalize_volume(self, volume):
        vol_min = np.min(volume)
        vol_max = np.max(volume)
        
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        else:
            volume = np.zeros_like(volume)
        
        return volume
    
    def _load_and_cache_volume(self, filename):
        gt_path = self.gt_dir / filename
        corrupted_path = self.corrupted_dir / filename
        noise_mask_path = self.noise_mask_dir / filename
        
        gt_volume = self.load_nifti(gt_path)
        corrupted_volume = self.load_nifti(corrupted_path)
        noise_mask_volume = self.load_nifti(noise_mask_path) if noise_mask_path.exists() else np.zeros_like(corrupted_volume)
        
        metadata = self.load_metadata(filename)
        
        corrupted_volume = self.reconstruct_corrupted_volume(
            corrupted_volume, gt_volume.shape, metadata, fill_method='interpolate'
        )
        
        if self.normalize:
            gt_volume = self.normalize_volume(gt_volume)
            corrupted_volume = self.normalize_volume(corrupted_volume)
        
        unified_mask = self.create_unified_mask(gt_volume, noise_mask_volume, metadata)
        
        self.volume_cache[filename] = {
            'gt_volume': gt_volume,
            'corrupted_volume': corrupted_volume,
            'unified_mask': unified_mask
        }
        
        return self.volume_cache[filename]
    
    def create_unified_mask(self, gt_volume, noise_mask_volume, metadata):
        background_mask = (gt_volume == 0)
        brain_mask = (gt_volume != 0)
        
        slice_removal_info = metadata.get('slice_removal', {})
        reconstructed_noise_mask = self.reconstruct_noise_mask(noise_mask_volume, gt_volume.shape, slice_removal_info)
        
        damage_mask = np.zeros_like(gt_volume, dtype=np.float32)
        for axis_key, axis_data in slice_removal_info.items():
            axis = int(axis_key.split('_')[1]) if isinstance(axis_key, str) and 'axis_' in axis_key else int(axis_key)
            slice_indices = axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data
            
            for idx in slice_indices:
                idx = int(idx)
                if axis == 0:
                    damage_mask[idx, :, :] = 1.0
                elif axis == 1:
                    damage_mask[:, idx, :] = 1.0
                elif axis == 2:
                    damage_mask[:, :, idx] = 1.0
        
        combined_damage = np.logical_or(damage_mask > 0, reconstructed_noise_mask > 0).astype(np.float32)
        
        unified_mask = np.zeros_like(gt_volume, dtype=np.float32)
        unified_mask[background_mask] = 0.0
        unified_mask[brain_mask & (combined_damage == 0)] = 0.5
        unified_mask[brain_mask & (combined_damage > 0)] = 1.0
        
        return unified_mask
    
    def load_metadata(self, filename):
        metadata_path = self.metadata_dir / filename.replace('.nii.gz', '.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _get_slice_shape(self, volume_shape, axis):
        shape_list = list(volume_shape)
        shape_list.pop(axis)
        return tuple(shape_list)
    
    def reconstruct_noise_mask(self, noise_mask, gt_shape, slice_removal_info):
        if not slice_removal_info:
            return noise_mask
        
        reconstructed = noise_mask.copy()
        for axis_key, axis_data in slice_removal_info.items():
            axis = int(axis_key.split('_')[1]) if isinstance(axis_key, str) and 'axis_' in axis_key else int(axis_key)
            removed_indices = sorted(axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data)
            
            if removed_indices:
                for idx in removed_indices:
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed.shape, axis))
                    reconstructed = np.insert(reconstructed, int(idx), fill_slice, axis=axis)
        
        return reconstructed
    
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
    
    def reconstruct_corrupted_volume(self, corrupted_volume, gt_shape, metadata, fill_method='interpolate'):
        slice_removal_info = metadata.get('slice_removal', {})
        
        if not slice_removal_info:
            return corrupted_volume
        
        reconstructed_corrupted = corrupted_volume.copy()
        
        for axis_key, axis_data in slice_removal_info.items():
            if isinstance(axis_key, str) and 'axis_' in axis_key:
                axis = int(axis_key.split('_')[1])
            else:
                axis = int(axis_key)
            
            if isinstance(axis_data, dict):
                removed_indices = sorted(axis_data.get('indices', []))
            else:
                removed_indices = sorted(axis_data)
            
            if not removed_indices:
                continue
            
            for idx in removed_indices:
                idx = int(idx)  # Ensure idx is an integer
                if fill_method == 'zero':
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed.shape, axis))
                elif fill_method == 'interpolate':
                    fill_slice = self._interpolate_slice(reconstructed_corrupted, axis, idx)
                else:
                    fill_slice = np.zeros(self._get_slice_shape(reconstructed.shape, axis))
                
                reconstructed_corrupted = np.insert(reconstructed_corrupted, idx, fill_slice, axis=axis)
        
        return reconstructed_corrupted
    
    def __getitem__(self, idx):
        patch_info = self.patch_indices[idx]
        filename = patch_info['filename']
        
        if filename in self.volume_cache:
            if not self.preload_all:
                self.cache_order.remove(filename)
                self.cache_order.append(filename)
            cached_data = self.volume_cache[filename]
            gt_volume = cached_data['gt_volume']
            corrupted_volume = cached_data['corrupted_volume']
            unified_mask = cached_data['unified_mask']
        else:
            cached_data = self._load_and_cache_volume(filename)
            gt_volume = cached_data['gt_volume']
            corrupted_volume = cached_data['corrupted_volume']
            unified_mask = cached_data['unified_mask']
            
            if not self.preload_all:
                self.cache_order.append(filename)
                
                if len(self.cache_order) > self.cache_size:
                    oldest = self.cache_order.pop(0)
                    del self.volume_cache[oldest]
        
        d_start = patch_info['d_start']
        h_start = patch_info['h_start']
        w_start = patch_info['w_start']
        pd, ph, pw = self.patch_size
        
        gt_patch = gt_volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        corr_patch = corrupted_volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        mask_patch = unified_mask[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        
        x_gt = torch.from_numpy(gt_patch.astype(np.float32)).unsqueeze(0)
        x_corr = torch.from_numpy(corr_patch.astype(np.float32)).unsqueeze(0)
        m_mask = torch.from_numpy(mask_patch.astype(np.float32)).unsqueeze(0)
        
        return {
            'x_gt': x_gt,
            'x_corr': x_corr,
            'm_mask': m_mask,
            'filename': filename
        }


def get_diffusion_dataloaders(
    data_dir,
    batch_size=1,
    train_split=0.7,
    val_split=0.15,
    num_workers=4,
    pin_memory=True,
    target_shape=None,
    use_patches=True,
    patch_size=(64, 64, 64),
    stride=None,
    cache_size=10,
    preload_all=False
    ):
    
    if use_patches:
        train_dataset = PatchBasedDiffusionDataset(
            data_dir=data_dir,
            split='train',
            train_split=train_split,
            val_split=val_split,
            normalize=True,
            patch_size=patch_size,
            stride=stride,
            cache_size=cache_size,
            preload_all=preload_all
        )
        
        val_dataset = PatchBasedDiffusionDataset(
            data_dir=data_dir,
            split='val',
            train_split=train_split,
            val_split=val_split,
            normalize=True,
            patch_size=patch_size,
            stride=stride,
            cache_size=cache_size,
            preload_all=preload_all
        )
    else:
        train_dataset = DiffusionMRIDataset(
            data_dir=data_dir,
            split='train',
            train_split=train_split,
            val_split=val_split,
            normalize=True,
            target_shape=target_shape
        )
        
        val_dataset = DiffusionMRIDataset(
            data_dir=data_dir,
            split='val',
            train_split=train_split,
            val_split=val_split,
            normalize=True,
            target_shape=target_shape
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_diffusion_test_dataloader(
    data_dir,
    batch_size=1,
    train_split=0.7,
    val_split=0.15,
    num_workers=4,
    pin_memory=True,
    target_shape=None
):
    test_dataset = DiffusionMRIDataset(
        data_dir=data_dir,
        split='test',
        train_split=train_split,
        val_split=val_split,
        normalize=True,
        target_shape=target_shape
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return test_loader


if __name__ == '__main__':
    data_dir = 'Data/Preprocessed-Data'
    
    if Path(data_dir).exists():
        train_loader, val_loader = get_diffusion_dataloaders(
            data_dir=data_dir,
            batch_size=1,
            num_workers=0,
            augmentation=True
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        batch = next(iter(train_loader))
        print(f"x_gt shape: {batch['x_gt'].shape}")
        print(f"x_corr shape: {batch['x_corr'].shape}")
        print(f"m_mask shape: {batch['m_mask'].shape}")
