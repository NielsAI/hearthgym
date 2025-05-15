import torch
from torch import nn

class SiLU(nn.Module):
    """
    Sigmoid-Linear Unit (SiLU), aka Swish: x * sigmoid(x)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
