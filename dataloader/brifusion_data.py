"""Aligned image pairs; one shared crop/flip for all observations and references."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image directory not found: {folder}')
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            key = path.stem.casefold()
            if key in result:
                raise ValueError(f'Duplicate image stem {key!r} in {folder}')
            result[key] = path
    if not result:
        raise ValueError(f'No images found in {folder}')
    return result


def load_image(path):
    with Image.open(path) as image:
        if image.mode in ('P', 'RGBA', 'CMYK', 'LA'):
            image = image.convert('RGB')
        data = np.array(image)
    if data.dtype == np.uint8:
        divisor = 255.0
    elif data.dtype == np.uint16:
        divisor = 65535.0
    else:
        raise ValueError(f'Expected an 8-bit or 16-bit integer image: {path} ({data.dtype})')
    if data.ndim == 2:
        data = data[:, :, None]
    if data.ndim != 3 or data.shape[2] not in (1, 3):
        raise ValueError(f'Unsupported image shape {data.shape}: {path}')
    return torch.from_numpy(data.astype(np.float32) / divisor).permute(2, 0, 1).contiguous()


def luminance(image):
    if image.shape[0] == 1:
        return image
    return (image * image.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)


def colorize(y, visible):
    if visible.shape[0] == 1:
        return y
    # Preserve visible chroma; chrominance restoration is outside this model.
    return (visible + y - luminance(visible)).clamp(0, 1)


def save_image(image, path):
    array = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    Image.fromarray(array).save(path)


class FusionDataset(Dataset):
    def __init__(self, root, mode='paired', crop_size=256, augment=True, medical=False):
        self.root, self.mode, self.crop_size, self.augment = Path(root), mode, crop_size, augment
        if mode not in ('paired', 'synthetic'):
            raise ValueError('mode must be paired or synthetic')
        names = ('CT', 'MRI', 'CT_gt', 'MRI_gt') if medical else ('Vis', 'Inf', 'Vis_gt', 'Inf_gt')
        if mode == 'paired':
            folders = names
        else:
            # In synthetic mode the caller explicitly declares these images clean references.
            clean = names[2:] if all((self.root / p).is_dir() for p in names[2:]) else names[:2]
            folders = (*clean, *clean)
        self.indices = [image_index(self.root / folder) for folder in folders]
        keys = set(self.indices[0])
        for folder, index in zip(folders, self.indices):
            if set(index) != keys:
                raise ValueError(f'Unmatched image stems in {folder}: missing={sorted(keys-set(index))[:5]}, extra={sorted(set(index)-keys)[:5]}')
        self.keys = sorted(keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        images = [luminance(load_image(mapping[key])) for mapping in self.indices]
        if len({tuple(x.shape) for x in images}) != 1:
            raise ValueError(f'Images must already be registered and have identical dimensions: {key}')
        if self.crop_size:
            h, w = images[0].shape[-2:]
            pad = (0, max(0, self.crop_size - w), 0, max(0, self.crop_size - h))
            images = [F.pad(x, pad, mode='replicate') for x in images]
            h, w = images[0].shape[-2:]
            top = int(torch.randint(h - self.crop_size + 1, ()).item()) if self.augment else (h - self.crop_size) // 2
            left = int(torch.randint(w - self.crop_size + 1, ()).item()) if self.augment else (w - self.crop_size) // 2
            images = [x[:, top:top+self.crop_size, left:left+self.crop_size] for x in images]
        if self.augment:
            for dimension in (-1, -2):
                if torch.rand(()).item() < 0.5:
                    images = [x.flip(dimension) for x in images]
        return dict(zip(('vi', 'ir', 'target_vi', 'target_ir'), images), name=self.indices[0][key].name)
