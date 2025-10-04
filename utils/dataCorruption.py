import numpy as np
import random
import scipy.ndimage as ndimage
import scipy.fftpack as fftpack

class CorruptData:
    def __init__(self, volume: np.ndarray, mode: str = 'distribution', dist: str = None, 
                total_slices: int = 10, n_noise_regions: int = 5, 
                noise_std: float = 0.1, freq_corruption: str = 'low', 
                contiguous: bool = False, mask_value: float = 0.0):

        self.raw_volume = volume
        self.mode = mode
        self.dist = dist
        self.total_slices = total_slices
        self.n_noise_regions = n_noise_regions
        self.noise_std = noise_std
        self.freq_corruption = freq_corruption
        self.contiguous = contiguous
        self.mask_value = mask_value

    def remove_slices_multi_axis(self, volume):
        corrupted_volume = volume.copy()
        mask = np.ones_like(volume, dtype=np.uint8)
        shape = volume.shape
        
        n_axes_to_remove = random.randint(1, 3)
        axes = random.sample([0, 1, 2], n_axes_to_remove)

        proportions = np.random.dirichlet(np.ones(n_axes_to_remove), 1)[0]
        slices_per_axis = np.round(proportions * self.total_slices).astype(int)
        
        while slices_per_axis.sum() < self.total_slices:
            slices_per_axis[np.argmin(slices_per_axis)] += 1
        while slices_per_axis.sum() > self.total_slices:
            slices_per_axis[np.argmax(slices_per_axis)] -= 1

        removed_info = {}
        for i, axis in enumerate(axes):
            n_slices = slices_per_axis[i]
            max_idx = shape[axis]
            if n_slices == 0 or max_idx <= 1:
                continue
            if self.contiguous and n_slices < max_idx:
                start = random.randint(0, max_idx - n_slices)
                indices = list(range(start, start + n_slices))
            else:
                indices = random.sample(range(max_idx), min(n_slices, max_idx))
            removed_info[axis] = indices
            for idx in indices:
                if axis == 0:
                    corrupted_volume[idx, :, :] = self.mask_value
                    mask[idx, :, :] = 0
                elif axis == 1:
                    corrupted_volume[:, idx, :] = self.mask_value
                    mask[:, idx, :] = 0
                elif axis == 2:
                    corrupted_volume[:, :, idx] = self.mask_value
                    mask[:, :, idx] = 0
        return corrupted_volume, mask, removed_info

    def add_noise_regions(self, volume):
        corrupted = volume.copy()
        shape = volume.shape
        noise_regions = []
        for _ in range(self.n_noise_regions):
            x = random.randint(0, shape[0] - 10)
            y = random.randint(0, shape[1] - 10)
            z = random.randint(0, shape[2] - 10)
            dx = random.randint(5, min(20, shape[0]-x))
            dy = random.randint(5, min(20, shape[1]-y))
            dz = random.randint(5, min(20, shape[2]-z))
            patch = corrupted[x:x+dx, y:y+dy, z:z+dz]
            noise = np.random.normal(0, self.noise_std, size=patch.shape)
            corrupted[x:x+dx, y:y+dy, z:z+dz] += noise
            noise_regions.append(((x, y, z), (dx, dy, dz)))
        return corrupted, noise_regions

    def frequency_corruption(self, volume):
        corrupted = volume.copy()
        shape = volume.shape
        
        def apply_low_pass(img):
            return ndimage.gaussian_filter(img, sigma=1.5)
        
        def apply_high_pass(img):
            fft = fftpack.fftn(img)
            fft_shift = fftpack.fftshift(fft)
            center = np.array(shape) // 2
            radius = min(shape) // 8
            mask_freq = np.ones(shape)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    for z in range(shape[2]):
                        if np.linalg.norm(np.array([x, y, z]) - center) < radius:
                            mask_freq[x, y, z] = 0
            fft_shift *= mask_freq
            fft_filtered = fftpack.ifftshift(fft_shift)
            return np.abs(fftpack.ifftn(fft_filtered))

        if self.freq_corruption == 'low':
            corrupted = apply_low_pass(corrupted)
        elif self.freq_corruption == 'high':
            corrupted = apply_high_pass(corrupted)
        elif self.freq_corruption == 'both':
            corrupted = apply_low_pass(corrupted)
            corrupted = apply_high_pass(corrupted)
        return corrupted

    def process(self):
        gt = self.raw_volume.copy()
        corrupted, mask, removed_info = self.remove_slices_multi_axis(gt)
        corrupted, noise_regions = self.add_noise_regions(corrupted)
        if self.freq_corruption in ['low', 'high', 'both']:
            corrupted = self.frequency_corruption(corrupted)
        metadata = {
            "removed_slices": removed_info,
            "noise_regions": noise_regions,
            "frequency_type": self.freq_corruption
        }
        return gt, corrupted, mask, metadata
