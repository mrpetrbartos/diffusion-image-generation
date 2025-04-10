import torch.nn as nn
from diffusers import UNet2DConditionModel


class ConditionalUNet(nn.Module):
    def __init__(self, config):
        super(ConditionalUNet, self).__init__()

        self.sample_size = config["in_size"]
        self.in_channels = config["in_channels"]
        self.out_channels = config["out_channels"]
        self.layers_per_block = config["layers_per_block"]
        self.block_out_channels = config["block_out_channels"]
        self.down_block_types = config["down_block_types"]
        self.up_block_types = config["up_block_types"]
        self.cross_attention_dim = config["cross_attention_dim"]

        self.unet = UNet2DConditionModel(
            sample_size=self.sample_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            layers_per_block=self.layers_per_block,
            block_out_channels=self.block_out_channels,
            down_block_types=self.down_block_types,
            up_block_types=self.up_block_types,
            cross_attention_dim=self.cross_attention_dim,
        )

    def forward(self, x, timesteps, encoder_hidden_states):
        """
        Forward pass of the conditional UNet.
        Args:
            x: The noisy input tensor of shape (batch_size, channels, height, width)
            timesteps: The number of timesteps to denoise an input
            encoder_hidden_states: Conditioning information (e.g., text embeddings)
        """
        return self.unet(x, timesteps, encoder_hidden_states=encoder_hidden_states)
