import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import cv2


class SlicePairDataset(Dataset):
    def __init__(self, data_dir, split='train', train_split=0.7, val_split=0.15, 
                 augmentation=True, img_size=(256, 256), multi_scale=True):
        self.data_dir = Path(data_dir)
        self.corrupted_dir = self.data_dir / 'corrupted'
        self.metadata_dir = self.data_dir / 'metadata'
        self.augmentation = augmentation and split == 'train'
        self.img_size = img_size
        self.multi_scale = multi_scale and split == 'train'
        self.scale_sizes = [(192, 192), (224, 224), (256, 256), (288, 288), (320, 320)]
        
        if not self.corrupted_dir.exists():
            raise FileNotFoundError(f"Corrupted directory not found: {self.corrupted_dir}")
        if not self.metadata_dir.exists():
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")
        
        all_files = sorted([f.name for f in self.corrupted_dir.glob('*.nii.gz')])
        n_total = len(all_files)
        
        if n_total == 0:
            raise ValueError(f"No .nii.gz files found in {self.corrupted_dir}")
        
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
        
        self.slice_pairs = []
        print(f"Loading {split} dataset with {len(self.files)} volumes...")
        self._extract_all_slice_pairs_parallel()
        print(f"Loaded {len(self.slice_pairs)} slice pairs")
    
    def _load_single_metadata(self, file_name):
        metadata_file = self.metadata_dir / file_name.replace('.nii.gz', '.json')
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            slice_removal_info = metadata.get('slice_removal', {})
            
            original_shape = (182, 218, 182)
            n_removed = [
                slice_removal_info.get('axis_0', {}).get('n_slices', 0),
                slice_removal_info.get('axis_1', {}).get('n_slices', 0),
                slice_removal_info.get('axis_2', {}).get('n_slices', 0)
            ]
            volume_shape = tuple(original_shape[i] - n_removed[i] for i in range(3))
        else:
            corrupted = nib.load(self.corrupted_dir / file_name)
            volume_shape = corrupted.shape
            slice_removal_info = {}
        
        return file_name, slice_removal_info, volume_shape
    
    def _extract_all_slice_pairs_parallel(self):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(self._load_single_metadata, f): f for f in self.files}
            
            for future in tqdm(as_completed(futures), total=len(self.files), desc="Loading metadata"):
                try:
                    file_name, slice_removal_info, volume_shape = future.result()
                    pairs = self._extract_pairs_from_metadata(volume_shape, slice_removal_info, file_name)
                    self.slice_pairs.extend(pairs)
                except Exception as e:
                    print(f"Error loading {futures[future]}: {e}")
    
    def _extract_pairs_from_metadata(self, volume_shape, slice_removal_info, file_name):
        pairs = []
        D, H, W = volume_shape
        original_shape = (182, 218, 182)
        
        for axis_idx, axis_key in enumerate(['axis_0', 'axis_1', 'axis_2']):
            axis_size = [D, H, W][axis_idx]
            original_axis_size = original_shape[axis_idx]
            
            if axis_key in slice_removal_info:
                removed_indices = sorted(slice_removal_info[axis_key].get('indices', []))
                
                if len(removed_indices) > 1:
                    for i in range(len(removed_indices) - 1):
                        orig_idx1 = removed_indices[i]
                        orig_idx2 = removed_indices[i + 1]
                        gap_count = orig_idx2 - orig_idx1 - 1
                        
                        n_removed_before_idx1 = sum(1 for r in removed_indices if r < orig_idx1)
                        n_removed_before_idx2 = sum(1 for r in removed_indices if r < orig_idx2)
                        
                        corrupted_idx1 = orig_idx1 - n_removed_before_idx1
                        corrupted_idx2 = orig_idx2 - n_removed_before_idx2
                        
                        if corrupted_idx1 < axis_size and corrupted_idx2 < axis_size and gap_count > 0:
                            pairs.append({
                                'file_name': file_name,
                                'axis': axis_idx,
                                'slice_idx1': corrupted_idx1,
                                'slice_idx2': corrupted_idx2,
                                'gap_count': gap_count,
                                'volume_shape': volume_shape
                            })
            else:
                for idx in range(0, axis_size - 1, max(1, axis_size // 10)):
                    if idx + 1 < axis_size:
                        pairs.append({
                            'file_name': file_name,
                            'axis': axis_idx,
                            'slice_idx1': idx,
                            'slice_idx2': idx + 1,
                            'gap_count': 0,
                            'volume_shape': volume_shape
                        })
        
        return pairs
    
    def __len__(self):
        return len(self.slice_pairs)
    
    def __getitem__(self, idx):
        pair_info = self.slice_pairs[idx]
        
        corrupted = nib.load(self.corrupted_dir / pair_info['file_name']).get_fdata()
        
        axis = pair_info['axis']
        idx1 = pair_info['slice_idx1']
        idx2 = pair_info['slice_idx2']
        gap_count = pair_info['gap_count']
        
        D, H, W = corrupted.shape
        axis_size = [D, H, W][axis]
        
        idx1 = min(idx1, axis_size - 1)
        idx2 = min(idx2, axis_size - 1)
        
        if idx1 >= axis_size or idx2 >= axis_size:
            idx1 = max(0, axis_size - 2)
            idx2 = axis_size - 1
        
        if axis == 0:
            slice1 = corrupted[idx1, :, :]
            slice2 = corrupted[idx2, :, :]
        elif axis == 1:
            slice1 = corrupted[:, idx1, :]
            slice2 = corrupted[:, idx2, :]
        else:
            slice1 = corrupted[:, :, idx1]
            slice2 = corrupted[:, :, idx2]
        
        slice1 = self._normalize_slice(slice1)
        slice2 = self._normalize_slice(slice2)
        
        if self.multi_scale:
            target_size = self.scale_sizes[np.random.randint(0, len(self.scale_sizes))]
            slice1 = cv2.resize(slice1, target_size, interpolation=cv2.INTER_LINEAR)
            slice2 = cv2.resize(slice2, target_size, interpolation=cv2.INTER_LINEAR)
        
        if self.augmentation:
            slice1, slice2 = self._apply_augmentation(slice1, slice2)
        
        slice1 = torch.from_numpy(slice1.astype(np.float32)).unsqueeze(0).clone()
        slice2 = torch.from_numpy(slice2.astype(np.float32)).unsqueeze(0).clone()
        
        has_gap = 1 if gap_count > 0 else 0
        
        return {
            'slice1': slice1,
            'slice2': slice2,
            'axis': torch.tensor(axis, dtype=torch.long),
            'has_gap': torch.tensor(has_gap, dtype=torch.long),
            'gap_count': torch.tensor(float(gap_count), dtype=torch.float32)
        }
    
    def _normalize_slice(self, slice_data):
        if slice_data.max() > 0:
            slice_data = slice_data / slice_data.max()
        return slice_data
    
    def _apply_augmentation(self, slice1, slice2):
        if np.random.rand() > 0.5:
            slice1 = np.fliplr(slice1)
            slice2 = np.fliplr(slice2)
        
        if np.random.rand() > 0.5:
            slice1 = np.flipud(slice1)
            slice2 = np.flipud(slice2)
        
        intensity_factor = np.random.uniform(0.9, 1.1)
        slice1 = np.clip(slice1 * intensity_factor, 0, 1)
        slice2 = np.clip(slice2 * intensity_factor, 0, 1)
        
        if np.random.rand() > 0.8:
            noise1 = np.random.normal(0, 0.02, slice1.shape)
            noise2 = np.random.normal(0, 0.02, slice2.shape)
            slice1 = np.clip(slice1 + noise1, 0, 1)
            slice2 = np.clip(slice2 + noise2, 0, 1)
        
        return slice1, slice2


def custom_collate_fn(batch):
    slice1_list = []
    slice2_list = []
    axis_list = []
    has_gap_list = []
    gap_count_list = []
    
    target_size = None
    for item in batch:
        if target_size is None:
            target_size = (item['slice1'].shape[-2], item['slice1'].shape[-1])
        
        current_size = (item['slice1'].shape[-2], item['slice1'].shape[-1])
        if current_size != target_size:
            import cv2
            slice1_np = item['slice1'].squeeze(0).numpy()
            slice2_np = item['slice2'].squeeze(0).numpy()
            slice1_np = cv2.resize(slice1_np, target_size[::-1], interpolation=cv2.INTER_LINEAR)
            slice2_np = cv2.resize(slice2_np, target_size[::-1], interpolation=cv2.INTER_LINEAR)
            slice1_list.append(torch.from_numpy(slice1_np).unsqueeze(0))
            slice2_list.append(torch.from_numpy(slice2_np).unsqueeze(0))
        else:
            slice1_list.append(item['slice1'])
            slice2_list.append(item['slice2'])
        
        axis_list.append(item['axis'])
        has_gap_list.append(item['has_gap'])
        gap_count_list.append(item['gap_count'])
    
    return {
        'slice1': torch.stack(slice1_list),
        'slice2': torch.stack(slice2_list),
        'axis': torch.stack(axis_list),
        'has_gap': torch.stack(has_gap_list),
        'gap_count': torch.stack(gap_count_list)
    }


def get_slice_pair_dataloaders(data_config, batch_size=16):
    data_dir = data_config.get('data_dir', 'Data/Preprocessed-Data')
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    multi_scale = data_config.get('multi_scale', True)
    
    img_size = tuple(data_config.get('img_size', [256, 256]))
    
    train_dataset = SlicePairDataset(
        data_dir,
        split='train',
        train_split=train_split,
        val_split=val_split,
        augmentation=True,
        img_size=img_size,
        multi_scale=multi_scale
    )
    
    val_dataset = SlicePairDataset(
        data_dir,
        split='val',
        train_split=train_split,
        val_split=val_split,
        augmentation=False,
        img_size=img_size,
        multi_scale=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader


def get_slice_pair_test_dataloader(data_config, batch_size=1):
    data_dir = data_config.get('data_dir', 'Data/Preprocessed-Data')
    train_split = data_config.get('train_split', 0.7)
    val_split = data_config.get('val_split', 0.15)
    
    img_size = tuple(data_config.get('img_size', [256, 256]))
    
    test_dataset = SlicePairDataset(
        data_dir,
        split='test',
        train_split=train_split,
        val_split=val_split,
        augmentation=False,
        img_size=img_size,
        multi_scale=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=data_config.get('pin_memory', True),
        collate_fn=custom_collate_fn
    )
    
    return test_loader
