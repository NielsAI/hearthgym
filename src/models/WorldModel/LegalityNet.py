import torch.nn as nn

class LegalityNet(nn.Module):
    """
    Multilayer Perceptron (MLP) for action legality prediction.
    Takes in a feature vector and outputs logits for each action.
    The logits are then passed through a sigmoid activation to get the legality mask.
    """
    def __init__(
        self, in_dim: int, n_actions: int,
        widths=(1024, 1024)):
        """	
        Initialize the LegalityNet.
        :param in_dim: Input dimension (size of the feature vector).
        :param n_actions: Number of actions (output dimension).
        :param widths: List of hidden layer widths.
        """
        super().__init__()
        layers = []
        last = in_dim
        # Create the MLP layers
        for w in widths:
            layers += [nn.Linear(last, w), nn.SiLU()]
            last = w
        layers += [nn.Linear(last, n_actions)]       # logits
        self.net = nn.Sequential(*layers)

        # Initialize weights and biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization for weights and zeros for biases
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, feat):
        """
        Forward pass through the network.
        :param feat: Input feature vector (size Z+H).
        :return: Logits for each action (size A).
        """
        return self.net(feat)
