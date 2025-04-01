import torch.nn as nn
from diffusers import UNet2DModel


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()

        self.sample_size = config["sample_size"]
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.layers_per_block = config["layers_per_block"]
        self.block_out_channels = config["block_out_channels"]
        self.down_block_types = config["down_block_types"]
        self.up_block_types = config["up_block_types"]

        self.unet = UNet2DModel(
            sample_size=self.sample_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers_per_block=self.layers_per_block,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",  # Mirroring upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, timesteps):
        """
        Forward pass of the model.
        Args:
            x: The noisy input tensor of shape (batch_size, channels, height, width)
            timesteps: The number of timesteps to denoise an input
        """
        return self.unet(x, timesteps)
