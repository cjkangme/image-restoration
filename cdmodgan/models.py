import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, gain=1.0, use_wscale=True, lrmul=1.0):
        super().__init__()
        he_std = gain * in_features ** (-0.5)  # He init

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * init_std)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.out_features = out_features

    def forward(self, x):
        return F.linear(x, self.weight * self.w_mul, self.bias)


class EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        gain=1.0,
        use_wscale=True,
        lrmul=1.0,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        he_std = gain * (in_channels * kernel_size**2) ** (-0.5)  # He init

        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * init_std
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight * self.w_mul,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        stride=1,
        padding=0,
        demodulate=True,
        fused_modconv=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.demodulate = demodulate
        self.fused_modconv = fused_modconv

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.mod_weight = EqualizedLinear(style_dim, in_channels)
        self.mod_bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x, y):
        # x: [batch, in_channels, height, width]
        # y: [batch, in_channels]
        batch, in_channels, height, width = x.shape

        # Modulation
        style = self.mod_weight(y) + 1
        style = style.view(batch, 1, in_channels, 1, 1)  # [batch, 1, in_channels, 1, 1]
        weight = self.weight.unsqueeze(0) * style

        if self.demodulate:
            d = torch.rsqrt(
                weight.pow(2).sum([2, 3, 4]) + 1e-8
            )  # [batch, out_channels]
            weight = weight * d.view(
                batch, self.out_channels, 1, 1, 1
            )  # [batch, out_channels, in_channels, kernel, kernel]

        if self.fused_modconv:
            x = x.view(1, -1, height, width)  # [1, batch*in_channels, height, width]
            weight = weight.view(
                -1, in_channels, self.kernel_size, self.kernel_size
            )  # [batch*out_channels, in_channels, kernel, kernel]
            x = F.conv2d(
                x, weight, padding=self.padding, groups=batch
            )  # [1, batch*out_channels, height, width]
            x = x.view(batch, self.out_channels, height, width)
        else:
            x = x * style.squeeze(1)  # [batch, in_channels, height, width]
            x = F.conv2d(
                x, self.weight, padding=self.padding
            )  # [batch, out_channels, height, width]
            if self.demodulate:
                x = x * d.view(batch, self.out_channels, 1, 1)

        return x


class NoiseLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.noise_type = "random"

    def forward(self, x, noise=None):
        if noise is None and self.noise_type == "random":
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


class CoModGANGenerator(nn.Module):
    def __init__(
        self,
        resolution=512,
        latent_size=512,
        style_dim=512,
        n_mlp=8,
        channel_base=16384,
        channel_max=512,
        input_channels=3,  # 3 = RGB, 1 = Gray
    ):
        super().__init__()

        self.resolution = resolution
        self.latent_size = latent_size
        self.latent_dim = style_dim
        self.dropout_rate = 0.5
        log2_res = int(np.log2(resolution))

        # Mapping network
        layers = []
        for i in range(n_mlp):
            layers.append(EqualizedLinear(style_dim, style_dim, lrmul=0.01))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

        # Synthesis network
        self.channels = {
            res: min(channel_base // (2 ** (res - 2)), channel_max)
            for res in range(1, log2_res + 1)
        }

        self.encoder = nn.ModuleDict()
        self.decoder = nn.ModuleDict()

        self.input_channels = input_channels  # Add mask channel
        in_channels = input_channels + 1

        # From RGB layer
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(in_channels, self.channels[log2_res - 1], 1),
            nn.LeakyReLU(0.2),
        )

        in_channels = self.channels[log2_res - 1]
        # Encoder blocks
        for res in range(log2_res, 2, -1):
            mid_channels = self.channels[res - 1]
            out_channels = self.channels[res - 2]
            self.encoder[f"{2**res}x{2**res}"] = nn.Sequential(
                EqualizedConv2d(in_channels, mid_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(mid_channels, out_channels, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )
            in_channels = out_channels

        # Encoder Final 4x4 layer
        self.encoder["4x4"] = nn.Sequential(
            EqualizedConv2d(in_channels, self.channels[1], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            EqualizedLinear(self.channels[1] * 16, self.channels[1] * 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(self.dropout_rate),
        )

        # Decoder blocks
        self.const = nn.Parameter(torch.randn(1, self.channels[2], 4, 4))

        # Early decode
        self.early_decode_dense = nn.Sequential(
            EqualizedLinear(self.channels[1] * 16, self.channels[1] * 16),
            nn.LeakyReLU(0.2),
        )

        self.decoder["4x4"] = nn.ModuleDict(
            {
                "conv1": ModulatedConv2d(
                    self.channels[1],
                    self.channels[1],
                    3,
                    padding=1,
                    style_dim=style_dim,
                ),
                "noise1": NoiseLayer(self.channels[1]),
                "activate": nn.LeakyReLU(0.2),
            }
        )

        for res in range(3, log2_res + 1):
            in_channels = self.channels[res - 1]
            out_channels = self.channels[res]

            self.decoder[f"{2**res}x{2**res}"] = nn.ModuleDict(
                {
                    "conv1": ModulatedConv2d(
                        in_channels, out_channels, 3, padding=1, style_dim=style_dim
                    ),
                    "conv2": ModulatedConv2d(
                        out_channels, out_channels, 3, padding=1, style_dim=style_dim
                    ),
                    "noise1": NoiseLayer(out_channels),
                    "noise2": NoiseLayer(out_channels),
                    "activate": nn.LeakyReLU(0.2),
                }
            )

        self.to_rgb = ModulatedConv2d(
            self.channels[log2_res], self.input_channels, 1, style_dim=style_dim
        )

    def forward(self, latent, image, mask):
        assert (
            image.size(1) == self.input_channels
        ), f"Expected {self.input_channels - 1} channels, got {image.size(1)} instead"

        # Map latent to style
        style = self.style(latent)

        # Encode input
        x = torch.cat([mask - 0.5, image * mask], dim=1)
        x = self.from_rgb(x)

        batch = x.size(0)
        features = {}
        for res in range(int(np.log2(self.resolution)), 1, -1):
            name = f"{2**res}x{2**res}"
            x = self.encoder[name][0](x)  # Conv1
            x = self.encoder[name][1](x)  # LeakyReLU
            features[res] = x
            x = self.encoder[name][2](x)  # Conv2
            x = self.encoder[name][3](x)  # LeakyReLU

        # Decode
        x = self.early_decode_dense(x)
        x = x.view(batch, self.channels[1], 4, 4)
        x = x + features[2]
        x = self.decoder["4x4"]["conv1"](x, style)
        x = self.decoder["4x4"]["noise1"](x)
        x = self.decoder["4x4"]["activate"](x)

        for res in range(3, int(np.log2(self.resolution)) + 1):
            name = f"{2**res}x{2**res}"
            block = self.decoder[name]

            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = x + features[res]

            x = block["conv1"](x, style)
            x = block["noise1"](x)
            x = block["activate"](x)

            x = block["conv2"](x, style)
            x = block["noise2"](x)
            x = block["activate"](x)

        # Combine with input image using mask
        rgb = self.to_rgb(x, style)
        out = rgb * (1 - mask) + image * mask

        return out


class CoModGANDiscriminator(nn.Module):
    def __init__(
        self, resolution=512, channel_base=16384, channel_max=512, input_channels=3
    ):
        super().__init__()

        log2_res = int(np.log2(resolution))
        self.resolution = resolution

        self.channels = {
            res: min(channel_base // (2 ** (res - 2)), channel_max)
            for res in range(2, log2_res + 1)
        }

        in_channels = input_channels + 1  # Add mask channel
        self.from_rgb = EqualizedConv2d(in_channels, self.channels[log2_res], 1)

        self.blocks = nn.ModuleDict()
        for res in range(log2_res, 2, -1):
            in_channels = self.channels[res]
            out_channels = self.channels[res - 1]

            self.blocks[f"{2**res}x{2**res}"] = nn.Sequential(
                EqualizedConv2d(in_channels, in_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(in_channels, out_channels, 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            )

        self.final_block = nn.Sequential(
            EqualizedConv2d(self.channels[2], self.channels[2], 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            EqualizedLinear(16 * self.channels[2], 1),
        )

    def forward(self, image, mask):
        x = torch.cat([mask - 0.5, image], dim=1)
        x = self.from_rgb(x)

        for res in range(int(np.log2(self.resolution)), 2, -1):
            name = f"{2**res}x{2**res}"
            x = self.blocks[name](x)

        return self.final_block(x)
