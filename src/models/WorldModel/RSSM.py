import torch, torch.nn as nn, torch.nn.functional as F
from models.WorldModel.blocks.BlockGRUCell import BlockGRUCell
from models.WorldModel.blocks.RMSNorm       import RMSNorm
from models.WorldModel.blocks.SiLU          import SiLU
from models.WorldModel.utils                 import unimix_logits, two_hot

FREE_NATS      = 1.0
UNIMIX_ALPHA   = 0.01
NUM_RETURN_BINS= 255

class RSSM(nn.Module):
    """Recurrent State-Space Model (RSSM) for sequential data.
    This model uses a neural network with block GRU cells to learn a latent representation of the input data.
    """    
    def __init__(
        self, num_cats, cat_dim, action_dim,
        hidden_dim=256,
        free_bits=FREE_NATS,
        return_bins=NUM_RETURN_BINS,
        unimix_alpha=UNIMIX_ALPHA
        ):
        """Initialize the RSSM model.
        :param num_cats: Number of categorical latent variables.
        :param cat_dim: Dimension of each categorical latent variable.
        :param action_dim: Dimension of the action space.
        :param hidden_dim: Dimension of the hidden state.
        :param free_bits: Minimum KL divergence for the latent variables.
        :param return_bins: Number of bins for the reward and value distributions.
        :param unimix_alpha: Unimix alpha parameter for the categorical distributions.
        """
        
        super().__init__()
        self.num_cats     = num_cats
        self.cat_dim      = cat_dim
        self.hidden_dim   = hidden_dim
        self.free_bits    = free_bits
        self.return_bins  = return_bins
        self.unimix_alpha = unimix_alpha

        # split hidden into num_cats blocks
        self.rnn      = BlockGRUCell(
            input_size  = num_cats*cat_dim + action_dim,
            hidden_size = hidden_dim,
            num_blocks  = num_cats
        )
        self.norm     = RMSNorm(hidden_dim)
        self.actv     = SiLU()

        # prior & posterior logits
        self.prior_fc = nn.Linear(hidden_dim,          num_cats*cat_dim)
        self.post_fc  = nn.Linear(hidden_dim+num_cats*cat_dim, num_cats*cat_dim)

        # distributional heads
        self.reward_head = nn.Linear(hidden_dim+num_cats*cat_dim, return_bins)
        self.value_head  = nn.Linear(hidden_dim+num_cats*cat_dim, return_bins)
        self.cont_head   = nn.Linear(hidden_dim+num_cats*cat_dim, 1)

        # initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def init_state(self, B, device):
        """
        Initialize the hidden and latent states of the RSSM.
        :param B: Batch size.
        :param device: Device to use (CPU or GPU).
        :return: Tuple of (h0, z0) where h0 is the initial hidden state and z0 is the initial latent state.
        """
        # Initialize hidden state and latent state
        h0 = torch.zeros(B, self.hidden_dim,      device=device)
        z0 = torch.zeros(B, self.num_cats*self.cat_dim, device=device)
        return (h0, z0)

    def _split(self, x):
        return x.view(-1, self.num_cats, self.cat_dim)

    def observe(self, prev, action, embed_obs):
        """
        Teacher-forced step.
        :param prev: (h_prev, z_prev) tuple of previous hidden state and latent state.
        :param action: action taken at the previous step.
        :param embed_obs: embedded observation at the current step (from autoencoder).
        :return: (h, z), p_logits, q_logits, r_logits, v_logits, c_logits
        """
        h_prev, z_prev = prev
        x = torch.cat([z_prev, action], dim=-1)
        h = self.actv(self.norm(self.rnn(x, h_prev)))

        # Prior 
        p_logits = self._split(self.prior_fc(h))
        p_logits = unimix_logits(p_logits, self.unimix_alpha)

        # Posterior
        post_in  = torch.cat([h, embed_obs], dim=-1)
        q_logits = self._split(self.post_fc(post_in))
        q_logits = unimix_logits(q_logits, self.unimix_alpha)

        # Softmax over the categorical distributions
        z = F.gumbel_softmax(q_logits, tau=1.0, hard=True, dim=-1)
        z = z.view(-1, self.num_cats*self.cat_dim)

        feat = torch.cat([h, z], dim=-1)
        return (h, z), p_logits, q_logits, self.reward_head(feat), self.value_head(feat), self.cont_head(feat)

    def imagine(self, prev, action):
        """
        Prior-only imagination step to predict future states.
        :param prev: (h_prev, z_prev) tuple of previous hidden state and latent state.
        :param action: action taken at the previous step.
        :return: (h, z), p_logits, r_logits, v_logits, c_logits
        """
        
        # Prepare the input for the RNN cell
        h_prev, z_prev = prev
        x = torch.cat([z_prev, action], dim=-1)
        h = self.actv(self.norm(self.rnn(x, h_prev)))

        # Prior
        p_logits = self._split(self.prior_fc(h))
        p_logits = unimix_logits(p_logits, self.unimix_alpha)

        # Sample from the prior distribution
        z = F.gumbel_softmax(p_logits, tau=1.0, hard=True, dim=-1)
        z = z.view(-1, self.num_cats*self.cat_dim)

        feat = torch.cat([h, z], dim=-1)
        return (h, z), p_logits, self.reward_head(feat), self.value_head(feat), self.cont_head(feat)

    def loss(self, p_logits, q_logits, r_logits, r_t, v_logits, v_t, c_logits, c_t):
        """Compute the loss for the RSSM model.
        :param p_logits: Prior logits from the model.
        :param q_logits: Posterior logits from the model.
        :param r_logits: Reward logits from the model.
        :param r_t: True reward values.
        :param v_logits: Value logits from the model.
        :param v_t: True value estimates.
        :param c_logits: Continuous logits from the model.
        :param c_t: True continuous values.
        :return: Computed loss value.
        """
        # KL Divergence
        p = F.softmax(p_logits, -1); logp = (p+1e-8).log()
        q = F.softmax(q_logits, -1); logq = (q+1e-8).log()
        kl = (p * (logp - logq)).sum(-1)
        kl = torch.clamp(kl, min=self.free_bits).mean()

        # Two-hot targets
        rt = two_hot(r_t, -1, 1, self.return_bins)
        vt = two_hot(v_t, -1, 1, self.return_bins)

        loss_r = F.cross_entropy(r_logits, rt)
        loss_v = F.cross_entropy(v_logits, vt)
        
        loss_c = F.binary_cross_entropy_with_logits(c_logits.squeeze(-1), c_t)
        return kl + loss_r + loss_v + loss_c
