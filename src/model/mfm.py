import torch
from torch import nn


class MFM(nn.Module):
    def forward(self, x):
        channels = x.shape[1]

        if channels % 2 != 0:
            raise ValueError(
                "The number of input channels for MFM must be even."
            )

        first_half, second_half = torch.chunk(
            x,
            chunks=2,
            dim=1,
        )

        return torch.maximum(first_half, second_half)