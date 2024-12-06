import os
from os.path import splitext, join, isfile
from pathlib import Path
import argparse
import torch
from PIL import Image

from utils.dataset import TrainDataset

from cdmodgan.models import CoModGANGenerator, CoModGANDiscriminator

def load_model(G_path):
    G = CoModGANGenerator(resolution=256, input_channels=1)
    G.load_state_dict(torch.load(G_path))
    return G

@torch.no_grad()
def inference(G, input_image, mask):
    latents = torch.randn(1, G.latent_dim).to(G.device)
    input_image = input_image.to(G.device)
    mask = mask.to(G.device)
    output = G(latents, input_image, mask)
    return output

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/test_input")
    parser.add_argument("--mask_path", type=str, default="data/test_mask")
    parser.add_argument("--model_step", type=int, default=1465)
    parser.add_argument("--output_path", type=str, default="output")

    args = parser.parse_args()

    os.path.makedirs(args.output_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G_path = os.path.join(os.path.curdir, "checkpoints", f"network-snapshot-{args.model_step:06d}.pt")
    G = load_model(G_path).to(device)

    input_path = Path(args.input_path)
    mask_path = Path(args.mask_path)
    ids = [
            splitext(f)[0]
            for f in os.listdir(input_path)
            if isfile(join(input_path, f)) and not f.startswith(".")
        ]

    for name in ids:
        image_path = list(input_path.glob(name + ".*"))
        mask_path = list(mask_path.glob(name + ".*"))

        assert (
            len(image_path) == 1
        ), f"Either no image or multiple images found for the ID {name}: {image_path}"
        assert (
            len(mask_path) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_path}"

        image = Image.open(image_path[0])
        mask = Image.open(mask_path[0])
        image = TrainDataset.preprocess([0, 255], image, 256, False)
        mask = TrainDataset.preprocess([0, 255], mask, 256, True)
        image = torch.as_tensor(image.copy()).float().contiguous()
        mask = torch.as_tensor(mask.copy()).long().contiguous()

        output = inference(G, image, mask)
        print(output.shape)
        # torch.Size([1, 1, 256, 256])

        image_pil = Image.fromarray((image.detach().squeeze().view(1, 2, 0).numpy() * 255).astype("uint8"))
        image_pil.save(join(args.output_path, f"{name}.png"))
