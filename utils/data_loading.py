import logging
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

class BFDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_root = Path(data_dir)
        dirs = [x.parts[-1] for x in self.data_root.iterdir() if x.is_dir()]
        assert 'B' in dirs and 'F' in dirs and 'M' in dirs, \
        f'Directories B, F, or M were not found in the data directory\
        \ndirectories found:{dirs}'

        # Bringing in the list of our data
        self.bright_fields = [x for x in (self.data_root / 'B').iterdir() if not x.is_dir()]
        self.fluorescents = [x for x in (self.data_root / 'F').iterdir() if not x.is_dir()]
        self.masks = [x for x in (self.data_root / 'M').iterdir() if not x.is_dir()]
        
        assert len(self.bright_fields) == len(self.fluorescents) and len(self.masks) == len(self.fluorescents),\
            f'different amount of data: B-{len(self.bright_fields)} F-{len(self.fluorescents)} M-{len(self.fluorescents)}'

    def __len__(self):
        return len(self.bright_fields)

    @staticmethod
    def preprocess(img, which):
        img = np.asarray(img)
        if which == 0:
            # we have a brightfield image
            img = img.astype('float') / 255
            img = 1 - img.mean(axis=-1)
        elif which == 1:
            # we have a fluorescence image
            img = img.astype('float') / 255
            img = img[:,:,1]
        elif which == 2:
            # We have a mask
            img = img.copy()
            img[img < 0.] = 0.0
            img[img > 1.] = 1.0
        return img

    def __getitem__(self, idx):
        B_file = self.bright_fields[idx]
        F_file = self.fluorescents[idx]
        M_file = self.masks[idx]

        brightfield = load_image(B_file)
        fluorescence = load_image(F_file)
        mask = load_image(M_file)

        brightfield = self.preprocess(brightfield, 0)
        fluorescence = self.preprocess(fluorescence, 1)
        mask = self.preprocess(mask, 2)

        img = torch.cat((TF.to_tensor(brightfield.copy()), TF.to_tensor(fluorescence.copy())), 0)

        return {
            'image': img.float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
    
    def getitem(self, idx):
        B_file = self.bright_fields[idx]
        F_file = self.fluorescents[idx]
        M_file = self.masks[idx]

        brightfield = load_image(B_file)
        fluorescence = load_image(F_file)
        mask = load_image(M_file)

        brightfield = self.preprocess(brightfield, 0)
        fluorescence = self.preprocess(fluorescence, 1)
        mask = self.preprocess(mask, 2)

        return {
            'brightfield': torch.as_tensor(brightfield.copy()).float().contiguous(),
            'fluorescence': torch.as_tensor(fluorescence.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }
    
    



def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
