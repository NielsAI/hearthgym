import torch
from torch import nn

class BlockLinear(nn.Module):
    """
    Splits the input into g groups along the last dim, applies an independent
    Linear(in_per, out_per) to each, and concatenates the results.
    """
    def __init__(self, in_features: int, out_features: int, num_blocks: int):
        super().__init__()
        assert in_features % num_blocks == 0, "in_features must divisible by num_blocks"
        assert out_features % num_blocks == 0, "out_features must divisible by num_blocks"
        self.num_blocks = num_blocks
        self.in_per = in_features // num_blocks
        self.out_per = out_features // num_blocks
        # weight shape: (num_blocks, out_per, in_per)
        self.weight = nn.Parameter(
            torch.randn(num_blocks, self.out_per, self.in_per) * (2 / (in_features + out_features))**0.5
        )
        # bias shape: (num_blocks * out_per,)
        self.bias = nn.Parameter(torch.zeros(num_blocks * self.out_per))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)
        returns: (batch, out_features)
        """
        B, _ = x.shape
        # reshape to (batch, num_blocks, in_per)
        xg = x.view(B, self.num_blocks, self.in_per)
        # block‐wise matmul: (batch, num_blocks, out_per)
        out = torch.einsum('bgi,goi->bgo', xg, self.weight)
        # flatten back to (batch, out_features)
        out = out.contiguous().view(B, -1)
        out = out + self.bias
        return out


class BlockGRUCell(nn.Module):
    """
    A GRU cell that splits its hidden state into blocks of size `block_size`
    and applies gate computations per block via BlockLinear.
    """
    def __init__(self, input_size: int, hidden_size: int, num_blocks: int):
        super().__init__()
        assert hidden_size % num_blocks == 0, "hidden_size must divisible by num_blocks"
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        # One grouped linear for reset/candidate/update gates:
        # input is [h, x], output is 3 * hidden_size
        self.gates = BlockLinear(input_size + hidden_size, 3 * hidden_size, num_blocks)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_size)
        h: (batch, hidden_size)
        returns: new hidden state (batch, hidden_size)
        """
        # Concatenate input and previous hidden
        hx = torch.cat([h, x], dim=-1)  # (batch, input+hidden)
        # Compute all 3 gates in one grouped linear
        gate_out = self.gates(hx)       # (batch, 3*hidden_size)
        # Split into reset, cand, update
        r, c, u = torch.split(gate_out, self.hidden_size, dim=-1)
        # Activation per gate
        reset = torch.sigmoid(r)
        # candidate uses tanh with reset applied to hidden part
        cand = torch.tanh(reset * c)
        update = torch.sigmoid(u)
        # new hidden
        h_new = update * cand + (1 - update) * h
        return h_new
    
