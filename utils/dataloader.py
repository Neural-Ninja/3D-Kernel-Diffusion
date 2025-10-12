import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
import numpy as np


class MRIDataset(Dataset):
    def __init__(self, data_dir, split='train', val_split=0.2):
        self.data_dir = Path(data_dir)
        self.gt_dir = self.data_dir / 'gt'
        self.corrupted_dir = self.data_dir / 'corrupted'
        self.mask_dir = self.data_dir / 'mask'
        
        self.files = sorted([f.name for f in self.gt_dir.glob('*.nii.gz')])
        
        split_idx = int(len(self.files) * (1 - val_split))
        if split == 'train':
            self.files = self.files[:split_idx]
        else:
            self.files = self.files[split_idx:]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_name = self.files[idx]
        
        gt = nib.load(self.gt_dir / file_name).get_fdata()
        corrupted = nib.load(self.corrupted_dir / file_name).get_fdata()
        mask = nib.load(self.mask_dir / file_name).get_fdata()
        
        gt = torch.from_numpy(np.ascontiguousarray(gt, dtype=np.float32)).unsqueeze(0)
        corrupted = torch.from_numpy(np.ascontiguousarray(corrupted, dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)
        
        return corrupted, gt, mask


def get_dataloaders(config):
    train_dataset = MRIDataset(
        config['data']['preprocessed_dir'],
        split='train',
        val_split=config['data']['val_split']
    )
    
    val_dataset = MRIDataset(
        config['data']['preprocessed_dir'],
        split='val',
        val_split=config['data']['val_split']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data'].get('pin_memory', True)
    )
    
    return train_loader, val_loader
