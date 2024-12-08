import os
import cv2
from os.path import splitext, join, isfile
from pathlib import Path
from copy import deepcopy
import argparse
import torch
import numpy as np
from PIL import Image
from metrics import masked_ssim_score

from utils.dataset import TrainDataset

from cdmodgan.models import CoModGANGenerator

def load_model(G_path):
    G = CoModGANGenerator(resolution=256, input_channels=1)
    G.load_state_dict(torch.load(G_path))
    return G

def preprocess(pil_img, scale, is_mask):
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
            if img.ndim == 2:
                mask[img == 0] = 1
            else:
                mask[(img == 0).all(-1)] = 1

            return np.expand_dims(mask, axis=0)

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img
        
def invert(mask):
    inverted_mask = np.where(mask == 0, 1, 0)
    return inverted_mask

@torch.no_grad()
def inference(G, input_image, mask, device):
    latents = torch.randn(1, G.latent_dim).to(device)
    input_image = input_image.to(device)
    mask = mask.to(device)
    output = G(latents, input_image, mask)
    return output

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/test_input")
    parser.add_argument("--mask_path", type=str, default="data/test_mask")
    parser.add_argument("--true_path", required=False, type=str, default=None)
    parser.add_argument("--model_step", type=int, default=1465)
    parser.add_argument("--output_path", type=str, default="/home/cv/output")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_path = os.path.join(os.path.curdir, "checkpoints", f"network-snapshot-{args.model_step:06d}.pt")
    checkpoint = torch.load(G_path)
    G = CoModGANGenerator(resolution=256, input_channels=1).to(device)
    G.load_state_dict(checkpoint["G_ema"])

    input_dir = Path(args.input_path)
    mask_dir = Path(args.mask_path)
    if args.true_path is not None:
        true_dir = Path(args.true_path)
    else:
        true_dir = None
    ids = [
            splitext(f)[0]
            for f in os.listdir(input_dir)
            if isfile(join(input_dir, f)) and not f.startswith(".")
        ]

    masked_ssim = 0
    for name in ids:
        image_path = list(input_dir.glob(name + ".*"))
        mask_path = list(mask_dir.glob(name + ".*"))

        assert (
            len(image_path) == 1
        ), f"Either no image or multiple images found for the ID {name}: {image_path}"
        assert (
            len(mask_path) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_path}"

        if true_dir is not None:
            true_path = list(true_dir.glob(name + ".*"))
            assert len(true_path) == 1, f"Either no true image or multiple true images found for the ID {name}: {true_path}"
            true_image = Image.open(true_path[0])
            true_image = preprocess(true_image, 256, False)
            true_image = torch.as_tensor(true_image.copy()).unsqueeze(0).float().contiguous()
            true_np = (true_image.squeeze().cpu().numpy() * 255).astype("uint8")

        image = Image.open(image_path[0])
        mask = Image.open(mask_path[0])
        image = preprocess(image, 256, False)
        mask = preprocess(mask, 256, True)
        image = torch.as_tensor(image.copy()).unsqueeze(0).float().contiguous()
        mask = torch.as_tensor(mask.copy()).unsqueeze(0).long().contiguous()

        output = inference(G, image, mask, device)
        # torch.Size([1, 1, 256, 256])

        pred_np = (output.detach().squeeze().cpu().numpy() * 255).astype("uint8")
        image_np = (image.squeeze().cpu().numpy() * 255).astype("uint8")
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
        mask_np = (mask_np == 0).astype("uint8")
        blended = image_np * (1 - mask_np) + pred_np * mask_np
        
        if true_dir is not None:
            masked_ssim += masked_ssim_score(true_np, pred_np, mask_np)

        # result = Image.fromarray(blended)
        # result.save(join(args.output_path, f"{name}.png"))
        pred_pil = Image.fromarray(pred_np)
        pred_pil.save(join(args.output_path, f"{name}.png"))
    if true_dir is not None:
        print(f"Masked SSIM: {masked_ssim/len(ids)}")
