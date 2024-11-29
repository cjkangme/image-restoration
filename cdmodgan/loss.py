import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self, l1_weight=0):
        super().__init__()
        self.l1_weight = l1_weight

    def forward(self, G, D, latents, reals, masks):
        """Generator loss computation

        Args:
            G: Generator network
            D: Discriminator network
            latents: Random latent vectors [batch_size, latent_dim]
            labels: Conditioning labels (if any) [batch_size, label_dim]
            reals: Real images [batch_size, channels, height, width]
            masks: Binary masks [batch_size, 1, height, width]

        Returns:
            Total generator loss and dict of individual loss terms
        """
        fake_images = G(latents, reals, masks)
        fake_scores = D(fake_images, masks)
        logistic_loss = F.softplus(-fake_scores).mean()
        l1_loss = torch.abs(fake_images - reals).mean()
        total_loss = logistic_loss + self.l1_weight * l1_loss

        loss_dict = {
            "g_logistic_loss": logistic_loss.item(),
            "g_l1_loss": l1_loss.item(),
            "g_total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


class DiscriminatorLoss(nn.Module):
    def __init__(self, gamma=10.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, G, D, latents, reals, masks):
        """Discriminator loss computation

        Args:
            G: Generator network
            D: Discriminator network
            latents: Random latent vectors [batch_size, latent_dim]
            labels: Conditioning labels (if any) [batch_size, label_dim]
            reals: Real images [batch_size, channels, height, width]
            masks: Binary masks [batch_size, 1, height, width]

        Returns:
            Total discriminator loss and dict of individual loss terms
        """
        reals.requires_grad_(True)

        with torch.no_grad():
            fake_images = G(latents, reals, masks)

        fake_scores = D(fake_images.detach(), masks)
        real_scores = D(reals, masks)

        d_loss_real = F.softplus(-real_scores).mean()
        d_loss_fake = F.softplus(fake_scores).mean()

        real_scores.requires_grad_(True)
        grad_real = torch.autograd.grad(
            outputs=real_scores.sum(), inputs=reals, create_graph=True
        )[0]
        gradient_penalty = grad_real.pow(2).sum([1, 2, 3]).mean()
        grad_reg = gradient_penalty * (self.gamma * 0.5)

        total_loss = d_loss_real + d_loss_fake + grad_reg

        loss_dict = {
            "d_real_loss": d_loss_real.item(),
            "d_fake_loss": d_loss_fake.item(),
            "d_real_score": real_scores.mean().item(),
            "d_fake_score": fake_scores.mean().item(),
            "d_gradient_penalty": gradient_penalty.item(),
            "d_total_loss": total_loss.item(),
        }

        return total_loss, loss_dict


def compute_generator_loss(G, D, latents, reals, masks, l1_weight=0):
    """Functional interface for generator loss computation"""
    loss_fn = GeneratorLoss(l1_weight=l1_weight)
    return loss_fn(G, D, latents, reals, masks)


def compute_discriminator_loss(G, D, latents, reals, masks, gamma=10.0):
    """Functional interface for discriminator loss computation"""
    loss_fn = DiscriminatorLoss(gamma=gamma)
    return loss_fn(G, D, latents, reals, masks)


def training_step(G, D, g_optimizer, d_optimizer, latents, reals, masks):
    """Single training step for both G and D

    Args:
        G: Generator network
        D: Discriminator network
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        latents: Random latent vectors
        labels: Conditioning labels
        reals: Real images
        masks: Binary masks

    Returns:
        Dictionary of loss metrics
    """
    # Train discriminator
    d_optimizer.zero_grad()
    d_loss, d_losses = compute_discriminator_loss(
        G, D, latents, reals, masks, gamma=10.0
    )
    d_loss.backward()
    d_optimizer.step()

    # Train generator
    g_optimizer.zero_grad()
    g_loss, g_losses = compute_generator_loss(
        G, D, latents, reals, masks, l1_weight=1.0
    )
    g_loss.backward()
    g_optimizer.step()

    # Combine metrics
    metrics = {}
    metrics.update(g_losses)
    metrics.update(d_losses)

    return metrics
