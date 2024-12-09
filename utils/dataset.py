import os
from os.path import splitext, join, isfile
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.ids = [
            splitext(f)[0]
            for f in os.listdir(self.images_dir)
            if isfile(join(self.images_dir, f)) and not f.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, \
                    make sure you put your images there"
            )

        self.mask_values = [255, 0]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        newW, newH = scale, scale
        assert (
            newW > 0 and newH > 0
        ), "Scale is too small, resized images would have no pixel"

        if is_mask:
            pil_img = pil_img.resize((newW, newH), Image.NEAREST)
        else:
            pil_img = pil_img.resize((newW, newH), Image.BILINEAR)

        img = np.asarray(pil_img)
        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return np.expand_dims(mask, axis=0)

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, index):
        name = self.ids[index]
        image_path = list(self.images_dir.glob(name + ".*"))
        mask_path = list(self.masks_dir.glob(name + ".*"))

        assert (
            len(image_path) == 1
        ), f"Either no image or multiple images found for the ID {name}: {image_path}"
        assert (
            len(mask_path) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_path}"

        image = Image.open(image_path[0])
        mask = Image.open(mask_path[0])

        image = self.preprocess(self.mask_values, image, 256, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, 256, is_mask=True)

        image = torch.as_tensor(image.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        return image, mask
