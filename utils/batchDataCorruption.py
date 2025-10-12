import warnings
warnings.filterwarnings("ignore")

import yaml
import json
import random
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, List
from tqdm import tqdm
import time
import sys
sys.path.append(str(Path(__file__).parent))
from dataCorruption import MRICorruption


class VolumeCorruptor:
    def __init__(self, volume: np.ndarray, severity: str, severity_configs: dict, 
                 corruption_probs: dict, affine: np.ndarray = None, seed: int = None):
        self.volume = volume
        self.severity = severity
        self.severity_configs = severity_configs
        self.corruption_probs = corruption_probs
        self.affine = affine if affine is not None else np.eye(4)
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def corrupt(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        config_dict = {
            'severity': str(self.severity),
            'slice_removal_prob': float(self.corruption_probs['slice_removal_prob']),
            'noise_prob': float(self.corruption_probs['noise_prob']),
            'frequency_corruption_prob': float(self.corruption_probs['frequency_corruption_prob']),
            'motion_artifact_prob': float(self.corruption_probs['motion_artifact_prob']),
            'bias_field_prob': float(self.corruption_probs['bias_field_prob']),
            'contiguous_slices': True,
            'affine': [[float(x) for x in row] for row in self.affine.tolist()],
            'seed': int(self.seed) if self.seed is not None else None,
            'severity_configs': {
                str(k): {
                    'slice_removal_range': [int(x) for x in v['slice_removal_range']],
                    'noise_std_range': [float(x) for x in v['noise_std_range']],
                    'n_noise_regions': [int(x) for x in v['n_noise_regions']],
                    'motion_severity': float(v['motion_severity']),
                    'bias_field_strength': float(v['bias_field_strength']),
                    'freq_corruption_intensity': float(v['freq_corruption_intensity'])
                }
                for k, v in self.severity_configs.items()
            }
        }
        
        config_path = Path(f'/tmp/corruption_config_{id(self)}.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        corruptor = MRICorruption(self.volume, str(config_path))
        gt, corrupted, mask, metadata = corruptor.process()
        
        removed_slices = metadata.get('slice_removal', {})
        for axis_key, axis_data in removed_slices.items():
            axis = int(axis_key.split('_')[1])
            indices = sorted(axis_data['indices'])
            if indices:
                corrupted = np.delete(corrupted, indices, axis=axis)
        
        config_path.unlink(missing_ok=True)
        return gt, corrupted, mask, metadata


class BatchDatasetCorruptor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_config = self.config['dataset']
        self.processing_config = self.config['processing']
        self.corruption_probs = self.config['corruption_probs']
        self.severity_configs = self.config['severity_configs']
        self.severity_distribution = self.config['severity_distribution']
        self.output_config = self.config['output_structure']
        self.random_config = self.config['random_seed']
        
        self.input_folder = Path(self.dataset_config['input_folder'])
        self.output_folder = Path(self.dataset_config['output_folder'])
        self.modality = self.dataset_config['modality']
        
        self._setup_output_structure()
    
    def _setup_output_structure(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if self.output_config['create_subfolders']:
            (self.output_folder / 'gt').mkdir(exist_ok=True)
            (self.output_folder / 'corrupted').mkdir(exist_ok=True)
            (self.output_folder / 'mask').mkdir(exist_ok=True)
            if self.output_config['save_metadata']:
                (self.output_folder / 'metadata').mkdir(exist_ok=True)
    
    def _find_volumes(self) -> List[Path]:
        volumes = []
        for pattern in [f"*{self.modality}*.nii.gz", f"*{self.modality}*.nii"]:
            volumes.extend(list(self.input_folder.rglob(pattern)))
        return sorted(volumes)
    
    def _load_nifti(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        img = nib.load(str(filepath))
        return img.get_fdata(), img.affine
    
    def _save_nifti(self, data: np.ndarray, affine: np.ndarray, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        img = nib.Nifti1Image(data.astype(np.float32), affine)
        nib.save(img, str(filepath))
    
    def _get_volume_id(self, filepath: Path) -> str:
        if filepath.parent.name != self.input_folder.name:
            return filepath.parent.name
        return filepath.stem.replace('.nii', '').replace(f'_{self.modality}', '')
    
    def _select_severities(self, n_variants: int) -> List[str]:
        severity_names = list(self.severity_distribution.keys())
        severity_probs = list(self.severity_distribution.values())
        return [np.random.choice(severity_names, p=severity_probs) for _ in range(n_variants)]
    
    def _generate_seed(self, vol_id: str, variant_idx: int) -> int:
        if self.random_config['use_fixed_seed']:
            base_seed = self.random_config['base_seed']
            seed_str = f"{vol_id}_{variant_idx}_{base_seed}"
            return hash(seed_str) % (2**32)
        return int(time.time() * 1000000) % (2**32) + variant_idx
    
    def process_volume(self, volume_path: Path, vol_id: str) -> int:
        volume, affine = self._load_nifti(volume_path)
        
        n_variants = random.randint(
            self.processing_config['min_variants'],
            self.processing_config['max_variants']
        )
        severities = self._select_severities(n_variants)
        
        for variant_idx, severity in enumerate(severities):
            seed = self._generate_seed(vol_id, variant_idx)
            
            corruptor = VolumeCorruptor(
                volume=volume,
                severity=severity,
                severity_configs=self.severity_configs,
                corruption_probs=self.corruption_probs,
                affine=affine,
                seed=seed
            )
            
            gt, corrupted, mask, metadata = corruptor.corrupt()
            
            filename = self.output_config['naming_format'].format(
                vol_id=vol_id,
                severity=severity,
                variant_idx=variant_idx
            )
            
            if self.output_config['create_subfolders']:
                gt_path = self.output_folder / 'gt' / f"{filename}.nii.gz"
                corrupted_path = self.output_folder / 'corrupted' / f"{filename}.nii.gz"
                mask_path = self.output_folder / 'mask' / f"{filename}.nii.gz"
            else:
                gt_path = self.output_folder / f"{filename}_gt.nii.gz"
                corrupted_path = self.output_folder / f"{filename}_corrupted.nii.gz"
                mask_path = self.output_folder / f"{filename}_mask.nii.gz"
            
            self._save_nifti(gt, affine, gt_path)
            self._save_nifti(corrupted, affine, corrupted_path)
            self._save_nifti(mask, affine, mask_path)
            
            if self.output_config['save_metadata']:
                metadata['volume_id'] = vol_id
                metadata['variant_idx'] = variant_idx
                metadata['seed'] = seed
                metadata['filename'] = filename
                
                if self.output_config['create_subfolders']:
                    metadata_path = self.output_folder / 'metadata' / f"{filename}.json"
                else:
                    metadata_path = self.output_folder / f"{filename}_metadata.json"
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        return n_variants
    
    def process_dataset(self):
        volumes = self._find_volumes()
        
        if not volumes:
            raise ValueError(f"No volumes found with modality '{self.modality}' in {self.input_folder}")
        
        total_variants = 0
        for volume_path in tqdm(volumes, desc="Processing volumes", unit="vol"):
            vol_id = self._get_volume_id(volume_path)
            try:
                n_variants = self.process_volume(volume_path, vol_id)
                total_variants += n_variants
            except Exception as e:
                tqdm.write(f"Error: {vol_id} - {str(e)}")
                continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    corruptor = BatchDatasetCorruptor(args.config)
    corruptor.process_dataset()
