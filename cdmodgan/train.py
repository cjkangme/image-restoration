import torch
import torch.optim as optim
import numpy as np
import wandb
from copy import deepcopy
from torchvision.utils import make_grid
from tqdm import tqdm
from collections import defaultdict

from metrics import ssim_score, masked_ssim_score, histogram_similarity


class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution=None,
        lod_training_kimg=600,
        lod_transition_kimg=600,
        minibatch_size_base=32,
        minibatch_size_dict={},
        G_lrate_base=0.002,
        G_lrate_dict={},
        D_lrate_base=0.002,
        D_lrate_dict={},
        lrate_rampup_kimg=0,
    ):
        resolution_log2 = 512**0.5
        self.kimg = cur_nimg / 1000.0

        # Level-of-detail and resolution
        if lod_initial_resolution is None:
            self.lod = 0.0
        else:
            self.lod = resolution_log2
            self.lod -= np.floor(np.log2(lod_initial_resolution))
            phase_dur = lod_training_kimg + lod_transition_kimg
            phase_idx = int(np.floor(self.kimg / phase_dur)) if phase_dur > 0 else 0
            phase_kimg = self.kimg - phase_idx * phase_dur
            self.lod -= phase_idx
            if lod_transition_kimg > 0:
                self.lod -= (
                    max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
                )
            self.lod = max(self.lod, 0.0)

        self.resolution = 2 ** (resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size
        self.minibatch_size = minibatch_size_dict.get(
            self.resolution, minibatch_size_base
        )

        # Learning rates
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        if lrate_rampup_kimg > 0:
            rampup = min(self.kimg / lrate_rampup_kimg, 1.0)
            self.G_lrate *= rampup
            self.D_lrate *= rampup


class Trainer:
    def __init__(
        self,
        G,
        D,
        G_loss,
        D_loss,
        dataset,
        data_loader,
        device,
        total_kimg=25000,
        mirror_augment=False,
        image_snapshot_ticks=50,
        network_snapshot_ticks=50,
        resume_pkl=None,
        G_smoothing_kimg=10.0,
        minibatch_repeats=4,
        G_reg_interval=4,
        D_reg_interval=16,
    ):
        self.G = G.to(device)
        self.D = D.to(device)
        self.G_ema = deepcopy(G).to(device)
        self.G_ema.load_state_dict(G.state_dict())

        # Optimizers
        learning_rate = 0.002
        self.G_opt = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.0, 0.99))
        self.D_opt = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.0, 0.99))

        self.device = device
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.dataset = dataset
        self.data_loader = data_loader
        self.total_kimg = total_kimg
        self.mirror_augment = mirror_augment
        self.image_snapshot_ticks = image_snapshot_ticks
        self.network_snapshot_ticks = network_snapshot_ticks
        self.G_smoothing_kimg = G_smoothing_kimg
        self.minibatch_repeats = minibatch_repeats
        self.G_reg_interval = G_reg_interval
        self.D_reg_interval = D_reg_interval

        # Setup logging
        self.running_stats = defaultdict(float)

        wandb.login(key="32679292006fe631ef3f272e8621a3e2b6b44ce9")
        wandb.init(project="image-inpainting", config={"lr": learning_rate})

    def process_batch(self, batch, mirror_augment=False):
        """Process a batch of images"""
        images, masks = batch
        if mirror_augment and torch.rand(1).item() < 0.5:
            images = torch.flip(images, [-1])
            masks = torch.flip(masks, [-1])
        return images.to(self.device), masks.to(self.device)

    @torch.no_grad()
    def update_g_ema(self, beta):
        """Update exponential moving average of G weights"""
        for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
            p_ema.copy_(p.lerp(p_ema, beta))

    def save_snapshot(self, filepath):
        """Save model snapshot"""
        torch.save(
            {
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "G_ema": self.G_ema.state_dict(),
                "G_opt": self.G_opt.state_dict(),
                "D_opt": self.D_opt.state_dict(),
            },
            filepath,
        )

    def get_metrics(self, true, pred, mask):
        """Return current training metrics"""
        true = (true.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(
            np.uint8
        )
        pred = (pred.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(
            np.uint8
        )
        mask = (mask.detach().cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(
            np.uint8
        )

        global_ssim_avg = ssim_score(true, pred)
        local_ssim_avg = masked_ssim_score(true, pred, mask)
        # histogram_similarity_avg = histogram_similarity(true, pred)
        return {
            "global_ssim_avg": global_ssim_avg,
            "local_ssim_avg": local_ssim_avg,
            # "histogram_similarity": histogram_similarity_avg,
            # "score": 0.2 * global_ssim_avg
            # + 0.4 * local_ssim_avg
            # + 0.4 * histogram_similarity_avg,
        }

    def train(self):
        cur_nimg = 0
        epoch = 0
        tick_start_nimg = cur_nimg

        while cur_nimg < self.total_kimg * 1000:
            # Initialize epoch
            batch_idx = 0

            # Setup training schedule
            sched = TrainingSchedule(cur_nimg=cur_nimg, training_set=self.dataset)

            # Training loop
            for batch_idx, batch in tqdm(
                enumerate(self.data_loader), total=len(self.data_loader)
            ):
                # Process batch
                reals, masks = self.process_batch(batch, self.mirror_augment)
                batch_size = reals.size(0)

                # Train discriminator
                self.D_opt.zero_grad()
                latents = torch.randn(batch_size, self.G.latent_dim).to(self.device)
                D_loss, D_loss_dict = self.D_loss(self.G, self.D, latents, reals, masks)
                D_loss.backward()
                self.D_opt.step()

                # Train generator
                if batch_idx % self.minibatch_repeats == 0:
                    self.G_opt.zero_grad()
                    latents = torch.randn(batch_size, self.G.latent_dim).to(self.device)
                    G_loss, G_loss_dict = self.G_loss(
                        self.G, self.D, latents, reals, masks
                    )
                    G_loss.backward()
                    self.G_opt.step()

                    # Update EMA
                    beta = 0.5 ** (batch_size / (self.G_smoothing_kimg * 1000))
                    self.update_g_ema(beta)

                # Update counts and stats
                cur_nimg += batch_size
                self.running_stats["D_loss"] += D_loss_dict["d_total_loss"]
                self.running_stats["G_loss"] += G_loss_dict["g_total_loss"]

                # Perform maintenance tasks once per tick
                done = cur_nimg >= self.total_kimg * 1000
                if cur_nimg >= tick_start_nimg * 1000 or done:
                    epoch += 1

                    for name, value in self.running_stats.items():
                        avg_value = value / batch_size
                        wandb.log({f"Loss/{name}": avg_value})

                    metrics = self.get_metrics(
                        reals, self.G_ema(latents, reals, masks), masks
                    )
                    for name, value in metrics.items():
                        wandb.log({f"Metrics/{name}": value})

                    # Save snapshots
                    if self.image_snapshot_ticks is not None and (
                        epoch % self.image_snapshot_ticks == 0 or done
                    ):
                        with torch.no_grad():
                            sample_z = torch.randn(16, self.G.latent_dim).to(
                                self.device
                            )
                            samples = self.G_ema(sample_z, reals[:16], masks[:16])
                            grid = make_grid(samples, normalize=True)
                            wandb.log({"Images": [wandb.Image(grid)]})

                    if self.network_snapshot_ticks is not None and (
                        epoch % self.network_snapshot_ticks == 0 or done
                    ):
                        self.save_snapshot(
                            f"network-snapshot-{cur_nimg // 1000:06d}.pt"
                        )

                    # Reset tick
                    tick_start_nimg = cur_nimg
                    self.running_stats.clear()

        # Save final snapshot
        self.save_snapshot("network-final.pt")


# Data processing utilities
def process_reals(images, lod, mirror_augment, drange_data, drange_net):
    """Process real images before feeding to networks"""
    images = images.float()
    if drange_data != drange_net:
        scale = (drange_net[1] - drange_net[0]) / (drange_data[1] - drange_data[0])
        bias = drange_net[0] - drange_data[0] * scale
        images = images * scale + bias

    if mirror_augment:
        images = torch.where(
            torch.rand(images.size(0), 1, 1, 1, device=images.device) < 0.5,
            images,
            torch.flip(images, [-1]),
        )

    if lod > 0:  # Smooth fade between consecutive levels-of-detail
        batch_size, channels, height, width = images.shape
        y = images.view(batch_size, channels, height // 2, 2, width // 2, 2)
        y = y.mean([3, 5], keepdim=True)
        y = y.repeat(1, 1, 1, 2, 1, 2)
        y = y.view(batch_size, channels, height, width)
        images = torch.lerp(images, y, lod - torch.floor(lod))

    return images
