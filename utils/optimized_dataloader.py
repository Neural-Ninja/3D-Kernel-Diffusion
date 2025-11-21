import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import threading


class OptimizedVQVAEPatchDataset(Dataset):
    def __init__(self, data_dir, split='train', train_split=0.7, val_split=0.15,
                 patch_size=(64, 64, 64), stride=None):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / 'gt'
        
        self.patch_size = np.array(patch_size)
        self.stride = np.array(stride if stride is not None else patch_size)
        
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.gt_dir}")
        
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
            raise ValueError(f"Invalid split: {split}")
        
        print(f"Preloading volume shapes for {len(self.files)} volumes...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            shapes = list(executor.map(self._get_volume_shape, self.files))
        
        volume_shape = shapes[0]
        self.patch_indices = self._compute_patch_indices(volume_shape)
        
        print(f"Loaded {split} dataset (VQ-VAE):")
        print(f"  Volumes: {len(self.files)}")
        print(f"  Patches per volume: {len(self.patch_indices)}")
        print(f"  Total patches: {len(self.files) * len(self.patch_indices)}")
    
    def _get_volume_shape(self, filename):
        gt_path = self.gt_dir / filename
        return nib.load(str(gt_path)).shape
    
    def _compute_patch_indices(self, volume_shape):
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        patch_positions = []
        for d in range(0, max(1, D - pd + 1), sd):
            for h in range(0, max(1, H - ph + 1), sh):
                for w in range(0, max(1, W - pw + 1), sw):
                    patch_positions.append((d, h, w))
        
        if len(patch_positions) == 0:
            patch_positions = [(0, 0, 0)]
        
        return patch_positions
    
    def __len__(self):
        return len(self.files) * len(self.patch_indices)
    
    def _normalize(self, volume):
        vol_min = volume.min()
        vol_max = volume.max()
        
        if vol_max > vol_min:
            normalized = (volume - vol_min) / (vol_max - vol_min)
            return normalized * 2 - 1
        return np.full_like(volume, -1.0)
    
    def __getitem__(self, idx):
        volume_idx = idx // len(self.patch_indices)
        patch_idx = idx % len(self.patch_indices)
        
        filename = self.files[volume_idx]
        d_start, h_start, w_start = self.patch_indices[patch_idx]
        
        gt_path = self.gt_dir / filename
        gt_volume = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        gt_volume = self._normalize(gt_volume)
        
        pd, ph, pw = self.patch_size
        
        d_end = min(d_start + pd, gt_volume.shape[0])
        h_end = min(h_start + ph, gt_volume.shape[1])
        w_end = min(w_start + pw, gt_volume.shape[2])
        
        gt_patch = gt_volume[d_start:d_end, h_start:h_end, w_start:w_end]
        
        if gt_patch.shape != tuple(self.patch_size):
            gt_patch = np.pad(gt_patch, [
                (0, pd - gt_patch.shape[0]),
                (0, ph - gt_patch.shape[1]),
                (0, pw - gt_patch.shape[2])
            ], mode='constant', constant_values=0)
        
        x_gt = torch.from_numpy(gt_patch).unsqueeze(0)
        
        return {
            'x_gt': x_gt
        }


class OptimizedPatchDataset(Dataset):
    def __init__(self, data_dir, split='train', train_split=0.7, val_split=0.15,
                 patch_size=(64, 64, 64), stride=None, preload_metadata=True):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / 'gt'
        self.corrupted_dir = self.data_dir / 'corrupted'
        self.metadata_dir = self.data_dir / 'metadata'
        self.noise_mask_dir = self.data_dir / 'noise-mask'
        
        self.patch_size = np.array(patch_size)
        self.stride = np.array(stride if stride is not None else patch_size)
        
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
        
        if preload_metadata:
            print(f"Preloading metadata for {len(self.files)} volumes...")
            self.metadata_cache = {}
            self.volume_shapes = {}
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(self._load_single_metadata, self.files))
            
            for filename, metadata, shape in results:
                self.metadata_cache[filename] = metadata
                self.volume_shapes[filename] = shape
            
            volume_shape = list(self.volume_shapes.values())[0]
        else:
            gt_path = self.gt_dir / self.files[0]
            volume_shape = nib.load(str(gt_path)).shape
        
        print(f"Volume shape: {volume_shape}")
        
        self.patch_indices = self._compute_patch_indices(volume_shape)
        
        print(f"Loaded {split} dataset:")
        print(f"  Volumes: {len(self.files)}")
        print(f"  Patches per volume: {len(self.patch_indices)}")
        print(f"  Total patches: {len(self.files) * len(self.patch_indices)}")
    
    def _load_single_metadata(self, filename):
        metadata_path = self.metadata_dir / filename.replace('.nii.gz', '.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        gt_path = self.gt_dir / filename
        shape = nib.load(str(gt_path)).shape
        
        return filename, metadata, shape
    
    def _compute_patch_indices(self, volume_shape):
        D, H, W = volume_shape
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        
        patch_positions = []
        for d in range(0, max(1, D - pd + 1), sd):
            for h in range(0, max(1, H - ph + 1), sh):
                for w in range(0, max(1, W - pw + 1), sw):
                    patch_positions.append((d, h, w))
        
        if len(patch_positions) == 0:
            patch_positions = [(0, 0, 0)]
        
        return patch_positions
    
    def __len__(self):
        return len(self.files) * len(self.patch_indices)
    
    def _load_volume_data(self, filename):
        gt_path = self.gt_dir / filename
        corrupted_path = self.corrupted_dir / filename
        noise_mask_path = self.noise_mask_dir / filename
        
        gt_volume = nib.load(str(gt_path)).get_fdata().astype(np.float32)
        corrupted_volume = nib.load(str(corrupted_path)).get_fdata().astype(np.float32)
        
        if noise_mask_path.exists():
            noise_mask_volume = nib.load(str(noise_mask_path)).get_fdata().astype(np.float32)
        else:
            noise_mask_volume = np.zeros_like(corrupted_volume, dtype=np.float32)
        
        metadata = getattr(self, 'metadata_cache', {}).get(filename)
        if metadata is None:
            metadata_path = self.metadata_dir / filename.replace('.nii.gz', '.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
        
        corrupted_volume = self._reconstruct_corrupted(corrupted_volume, gt_volume.shape, metadata)
        noise_mask_volume = self._reconstruct_corrupted(noise_mask_volume, gt_volume.shape, metadata)
        
        gt_volume = self._normalize(gt_volume)
        corrupted_volume = self._normalize(corrupted_volume)
        
        unified_mask = self._create_mask(gt_volume, noise_mask_volume, metadata)
        
        return gt_volume, corrupted_volume, unified_mask
    
    def _normalize(self, volume):
        vol_min = volume.min()
        vol_max = volume.max()
        
        if vol_max > vol_min:
            normalized = (volume - vol_min) / (vol_max - vol_min)
            return normalized * 2 - 1
        return np.full_like(volume, -1.0)
    
    def _reconstruct_corrupted(self, corrupted, target_shape, metadata):
        slice_removal = metadata.get('slice_removal', {})
        if not slice_removal:
            return corrupted
        
        result = corrupted.copy()
        for axis_key, axis_data in slice_removal.items():
            axis = int(axis_key.split('_')[1]) if 'axis_' in str(axis_key) else int(axis_key)
            indices = sorted(axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data)
            
            for idx in indices:
                idx = int(idx)
                fill_slice = np.zeros(np.delete(result.shape, axis), dtype=result.dtype)
                result = np.insert(result, idx, fill_slice, axis=axis)
        
        return result
    
    def _create_mask(self, gt_volume, noise_mask, metadata):
        background = (gt_volume == 0)
        brain = (gt_volume != 0)
        
        slice_removal = metadata.get('slice_removal', {})
        damage = np.zeros_like(gt_volume, dtype=np.float32)
        
        for axis_key, axis_data in slice_removal.items():
            axis = int(axis_key.split('_')[1]) if 'axis_' in str(axis_key) else int(axis_key)
            indices = axis_data.get('indices', []) if isinstance(axis_data, dict) else axis_data
            
            for idx in indices:
                idx = int(idx)
                if axis == 0:
                    damage[idx, :, :] = 1.0
                elif axis == 1:
                    damage[:, idx, :] = 1.0
                else:
                    damage[:, :, idx] = 1.0
        
        combined_damage = np.logical_or(damage > 0, noise_mask > 0).astype(np.float32)
        
        mask = combined_damage
        
        return mask
    
    def __getitem__(self, idx):
        volume_idx = idx // len(self.patch_indices)
        patch_idx = idx % len(self.patch_indices)
        
        filename = self.files[volume_idx]
        d_start, h_start, w_start = self.patch_indices[patch_idx]
        
        gt_volume, corrupted_volume, unified_mask = self._load_volume_data(filename)
        
        pd, ph, pw = self.patch_size
        
        d_end = min(d_start + pd, gt_volume.shape[0])
        h_end = min(h_start + ph, gt_volume.shape[1])
        w_end = min(w_start + pw, gt_volume.shape[2])
        
        gt_patch = gt_volume[d_start:d_end, h_start:h_end, w_start:w_end]
        corr_patch = corrupted_volume[d_start:d_end, h_start:h_end, w_start:w_end]
        mask_patch = unified_mask[d_start:d_end, h_start:h_end, w_start:w_end]
        
        if gt_patch.shape != tuple(self.patch_size):
            gt_patch = np.pad(gt_patch, [
                (0, pd - gt_patch.shape[0]),
                (0, ph - gt_patch.shape[1]),
                (0, pw - gt_patch.shape[2])
            ], mode='constant', constant_values=0)
            corr_patch = np.pad(corr_patch, [
                (0, pd - corr_patch.shape[0]),
                (0, ph - corr_patch.shape[1]),
                (0, pw - corr_patch.shape[2])
            ], mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, [
                (0, pd - mask_patch.shape[0]),
                (0, ph - mask_patch.shape[1]),
                (0, pw - mask_patch.shape[2])
            ], mode='constant', constant_values=0)
        
        x_gt = torch.from_numpy(gt_patch).unsqueeze(0)
        x_corr = torch.from_numpy(corr_patch).unsqueeze(0)
        m_mask = torch.from_numpy(mask_patch).unsqueeze(0)
        
        return {
            'x_gt': x_gt,
            'x_corr': x_corr,
            'm_mask': m_mask,
            'filename': filename
        }


def get_vqvae_dataloaders(
    data_dir,
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    num_workers=4,
    pin_memory=True,
    patch_size=(64, 64, 64),
    stride=None
):
    train_dataset = OptimizedVQVAEPatchDataset(
        data_dir=data_dir,
        split='train',
        train_split=train_split,
        val_split=val_split,
        patch_size=patch_size,
        stride=stride
    )
    
    val_dataset = OptimizedVQVAEPatchDataset(
        data_dir=data_dir,
        split='val',
        train_split=train_split,
        val_split=val_split,
        patch_size=patch_size,
        stride=stride
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_optimized_dataloaders(
    data_dir,
    batch_size=16,
    train_split=0.7,
    val_split=0.15,
    num_workers=4,
    pin_memory=True,
    patch_size=(64, 64, 64),
    stride=None
):
    train_dataset = OptimizedPatchDataset(
        data_dir=data_dir,
        split='train',
        train_split=train_split,
        val_split=val_split,
        patch_size=patch_size,
        stride=stride,
        preload_metadata=True
    )
    
    val_dataset = OptimizedPatchDataset(
        data_dir=data_dir,
        split='val',
        train_split=train_split,
        val_split=val_split,
        patch_size=patch_size,
        stride=stride,
        preload_metadata=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return train_loader, val_loader
