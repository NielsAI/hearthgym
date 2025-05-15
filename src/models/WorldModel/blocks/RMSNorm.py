import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (DreamerV3 style).
    y = x * (rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)) * weight
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        # one learnable scale parameter per feature
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean of squares over last dimension
        mean2 = x.pow(2).mean(dim=-1, keepdim=True)                    
        # Add eps inside the sqrt for numeric safety
        norm_factor = torch.rsqrt(mean2 + self.eps)                     
        # Apply normalization and per-feature scale
        return x * norm_factor * self.weight
