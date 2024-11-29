import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from cdmodgan.train import Trainer
from cdmodgan.models import CoModGANGenerator, CoModGANDiscriminator
from cdmodgan.loss import GeneratorLoss, DiscriminatorLoss
from utils.dataset import TrainDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/train_gray")
    parser.add_argument("--mask_path", type=str, default="data/train_mask")

    args = parser.parse_args()

    # FIXME: gray 이미지 아닐경우 input_channels 4로 수정
    G = CoModGANGenerator(resolution=512, input_channels=1)
    D = CoModGANDiscriminator(resolution=512, input_channels=1)
    G_loss = GeneratorLoss()
    D_loss = DiscriminatorLoss()
    input_dir = args.input_path
    mask_dir = args.mask_path
    dataset = TrainDataset(input_dir, mask_dir)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )
    total_kimg = len(dataset)  # 30k images

    assert torch.cuda.is_available(), "CUDA must be available"

    train = Trainer(
        G=G,
        D=D,
        G_loss=G_loss,
        D_loss=D_loss,
        dataset=dataset,
        data_loader=data_loader,
        device=torch.device("cuda"),
        total_kimg=total_kimg,
    )

    train.train()
