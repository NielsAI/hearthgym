# Load modules
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from agents.RandomAgent import RandomAgent
from agents.GreedyAgent import GreedyAgent

from env.hearthstone.HearthGym import MAX_HAND_SIZE, MAX_NUM_CHARACTERS, MAX_TOTAL_CHARACTERS, MAX_DISCOVER_OPTIONS, MAX_GENERIC_CHOICES
from models.WorldModel.LegalityNet import LegalityNet
from pytorch_optimizer import LaProp

# Robustness Constants
OVERSHOOT_H = 5           # latent overshooting
PCT_RETURN = 95.0         # percentile return normalization

# ----------------------------------------------------------------------------
# Symlog / Symexp Transforms

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric logarithm function.
    :param x: Input tensor.
    :return: Symmetric logarithm of x.
    """
    return torch.sign(x) * torch.log1p(x.abs())

def symexp(x: torch.Tensor) -> torch.Tensor:
    """
    Symmetric exponential function.
    :param x: Input tensor.
    :return: Symmetric exponential of x.
    """
    return torch.sign(x) * torch.expm1(x.abs())

def two_hot(x: torch.Tensor, v_min: float, v_max: float, num_bins: int) -> torch.Tensor:
    """
    Convert 1D tensor x of shape (B,) into a two-hot distribution of shape (B, num_bins).
    :param x: Input tensor of shape (B,).
    :param v_min: Minimum value for clamping.
    :param v_max: Maximum value for clamping.
    :param num_bins: Number of bins for the distribution.
    :return: Two-hot distribution tensor of shape (B, num_bins).
    """
    # Clamp & compute positions in [0, num_bins-1]
    # Convert to tensor if x is a numpy array
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x_clamped = x.clamp(v_min, v_max)            # (B,)  
    v_min      = torch.tensor(v_min, device=x.device)
    v_max      = torch.tensor(v_max, device=x.device)
    num_bins   = torch.tensor(num_bins, device=x.device)
    
    # Compute bin width and position
    width      = (v_max - v_min) / (num_bins - 1)
    pos        = (x_clamped - v_min) / width

    # Lower & upper bin indices
    l = pos.floor().long().clamp(0, num_bins-2)  # (B,)
    u = l + 1                                    # (B,)

    # Fractional offset
    off = (pos - l.float())                      # (B,)

    # Build distribution via indexing
    B = x.size(0)
    dist = x.new_zeros((B, num_bins))            # same device & dtype as x
    idx = torch.arange(B, device=x.device)
    dist[idx, l] = 1.0 - off
    dist[idx, u] = off

    return dist

def normalize_returns(returns: torch.Tensor, percentile: float = PCT_RETURN) -> torch.Tensor:
    """
    Map returns to [-1, 1] by percentile clipping at `percentile` value.
    :param returns: Input tensor of shape (B,).
    :param percentile: Percentile for clipping.
    :return: Normalized tensor of shape (B,).
    """
    # Determine high and low percentiles
    high = np.percentile(returns.cpu().numpy(), percentile)
    low  = np.percentile(returns.cpu().numpy(), 100 - percentile)
    clipped = returns.clamp(low, high)
    # Scale to [-1,1]
    return 2 * (clipped - low) / (high - low) - 1

def unimix_logits(logits: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """
    Mix categorical logits with alpha fraction of uniform prior.
    :param logits: Input logits tensor of shape (B, num_classes).
    :param alpha: Mixing coefficient for uniform prior.
    :return: Mixed logits tensor of shape (B, num_classes).
    """
    num_classes = logits.size(-1)
    uni_logit = torch.log(torch.full_like(logits, 1.0 / num_classes))
    
    # Compute log probabilities for the uniform distribution
    return torch.logaddexp(
        torch.log(torch.tensor(1 - alpha, device=logits.device)) + logits,
        torch.log(torch.tensor(alpha, device=logits.device)) + uni_logit
    )

def action_to_index(action: dict) -> int:
        """Convert an action dict to a flat index using precomputed multipliers.
        :param action: Action dictionary containing various indices.
        :return: Flat index representing the action.
        """        
        # Flatten the action in the given order
        idx = action['action_type']
        idx = idx * MAX_HAND_SIZE + action['card_index']
        idx = idx * MAX_NUM_CHARACTERS + action['attacker_index']
        idx = idx * MAX_TOTAL_CHARACTERS + action['target_index']
        idx = idx * MAX_DISCOVER_OPTIONS + action['discover_index']
        idx = idx * MAX_GENERIC_CHOICES + action['choose_index']
        return idx

def one_hot_encode(action, num_actions: int) -> np.ndarray:
    """
    Convert an action to a one-hot encoded vector.
    :param action: Action to be converted (can be an integer or a dictionary).
    :param num_actions: Total number of actions.
    :return: One-hot encoded vector of shape (num_actions,).
    """
    # Check if the action is a dictionary and convert it to an integer index
    if type(action) == dict:
        # Explicitly convert the action to an integer
        action_index = action_to_index(action)
    else:
        action_index = action
    
    one_hot = np.zeros(num_actions, dtype=np.float32)
    one_hot[action_index] = 1.0
    return one_hot

def collect_training_data(env, num_episodes=50, collect_masks=False, sampling_agent=None, score_method=None) -> tuple:
    """
    Collect training data from the environment using a sampling agent.
    :param env: The environment instance.
    :param num_episodes: Number of episodes to collect data from.
    :param collect_masks: Whether to collect action masks.
    :param sampling_agent: The agent to use for sampling actions.
    :param score_method: The scoring method for the agent.
    :return: Tuple of (data, mask_data) where data is a list of episodes and mask_data is a list of compressed action masks.
    """
    data = []
    mask_data = []
    
    # Initialize the sampling agent
    if sampling_agent == "GreedyAgent":
        sampling_agent = GreedyAgent(score_method=score_method)
    else:
        sampling_agent = RandomAgent()  
    
    # Loop through the specified number of episodes
    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        episode_data = []
        mask_pairs = []
        
        # Collect data for the episode
        while not done:
            valid_actions, action_mask = env.get_valid_actions()
            action = sampling_agent.act(observation=obs, valid_actions=valid_actions, action_mask=action_mask, env=env)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Store the data in the episode_data list
            episode_data.append((obs, action, reward, next_obs, done))
            if collect_masks:
                compressed_mask = np.packbits(action_mask, bitorder='big')
                mask_pairs.append((obs, action, compressed_mask))
            obs = next_obs
        data.append(episode_data)
        if collect_masks:
            mask_data.append(mask_pairs)
            
    mask_obj = np.array(mask_data, dtype=object)
    return data, mask_obj

def train_autoencoder(
    autoencoder,
    training_states,
    num_epochs=50,
    batch_size=128,
    lr=1e-3,
    device='cuda',
    cont_indices=None,
    disc_indices=None,
    lambda_disc=1.0
    ):
    """
    Train the autoencoder on the provided training states.
    :param autoencoder: The autoencoder model to be trained.
    :param training_states: The training data (states) for the autoencoder.
    :param num_epochs: Number of epochs for training.
    :param batch_size: Batch size for training.
    :param lr: Learning rate for the optimizer.
    :param device: Device to use for training (CPU or GPU).
    :param cont_indices: Indices of continuous features in the observation space.
    :param disc_indices: Indices of discrete features in the observation space.
    :param lambda_disc: Weight for the discrete loss term.
    :return: None
    """
    # Set the autoencoder to training mode and move it to the specified device
    autoencoder.train().to(device)

    # use LaProp optimizer
    optim = LaProp(
        autoencoder.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-20
    )

    data = torch.as_tensor(training_states, dtype=torch.float32, device=device)
    N = data.size(0)

    # Loop through the specified number of epochs
    for epoch in range(num_epochs):
        # Shuffle the data for each epoch
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            idx  = perm[i:i+batch_size]
            batch = data[idx]

            # Process the batch
            optim.zero_grad()
            recon, _ = autoencoder(batch, cont_indices, disc_indices)

            tgt_c = batch[:, cont_indices]
            tgt_d = batch[:, disc_indices]

            rec_c = recon[:, cont_indices]
            rec_d = recon[:, disc_indices]

            # Compute the loss
            loss = F.mse_loss(rec_c, tgt_c) + \
                   lambda_disc * F.binary_cross_entropy(rec_d, tgt_d)

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            optim.step()

            epoch_loss += loss.item() * batch.size(0)

        print(f"[AE] epoch {epoch+1:3d}/{num_epochs}   "
              f"loss = {epoch_loss/N:8.4f}")

def train_legality(dataloader, legality_net: LegalityNet, epochs=5, lr=3e-4, device="cuda"):
    """
    Train the legality network on the provided dataset.
    :param dataloader: DataLoader for the training data.
    :param legality_net: The legality network model to be trained.
    :param epochs: Number of epochs for training.
    :param lr: Learning rate for the optimizer.
    :param device: Device to use for training (CPU or GPU).
    :return: Trained legality network.
    """
    # Set the legality network to training mode and move it to the specified device
    legality_net.train().to(device)
    
    # use LaProp optimizer
    optim = LaProp(
        legality_net.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-20
    )

    # Loop through the specified number of epochs
    for ep in range(1, epochs+1):
        total_loss=0
        n_tuples=0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass and compute the loss
            loss = F.binary_cross_entropy_with_logits(legality_net(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Compute the total loss and number of samples
            total_loss += loss.item() * x.size(0)
            n_tuples += x.size(0)
        print(f"[LegNet]  epoch {ep}/{epochs}  BCE = {total_loss/n_tuples:.4f}")
    return legality_net.to(device)

def train_rssm(
    model,
    dataloader,
    epochs: int = 60,
    overshoot: int = 5,
    lr: float = 2e-4,
    device: str = "cuda",
    print_every: int = 1
):
    """
    Train the RSSM model on the provided dataset.
    :param model: The RSSM model to be trained.
    :param dataloader: DataLoader for the training data.
    :param epochs: Number of epochs for training.
    :param overshoot: Number of overshooting steps.
    :param lr: Learning rate for the optimizer.
    :param device: Device to use for training (CPU or GPU).
    :param print_every: Frequency of printing training progress.
    :return: None
    """
    # Set the model to training mode and move it to the specified device
    model.to(device).train()
    
    # use LaProp optimizer
    optim = LaProp(model.parameters(), lr=lr, betas=(0.9,0.999), eps=1e-20)
    
    # Get the size of the categorical features (same as the latent size in the autoencoder)
    cat_size = model.num_cats * model.cat_dim  
    
    best_loss    = float('inf')
    grad_norm_ema= 0.0
    beta         = 0.9

    # Loop through the specified number of epochs
    for ep in range(1, epochs+1):
        total_loss = 0.0
        batch_count = 0

        for x, z_obs_seq, r_seq, c_seq in dataloader:
            # Get batch size and sequence length
            B, T, _ = z_obs_seq.shape
            z_obs_seq = z_obs_seq.to(device)
            x         = x.to(device)
            r_seq     = r_seq.to(device)
            c_seq     = c_seq.to(device)

            # Initialize the model state
            state = model.init_state(B, device)
            loss  = 0.0
            batch_count += 1

            # Posterior pass
            for t in range(T):
                # Split out action from x
                a_t = x[:, t, cat_size:]           # [B, action_dim]
                e_t = z_obs_seq[:, t]              # [B, cat_size]
                r_t = r_seq[:, t].squeeze(-1)      # [B]
                c_t = c_seq[:, t].squeeze(-1)      # [B]

                # Observe step abd compute loss
                state, p_logits, q_logits, r_log, v_log, c_log = model.observe(state, a_t, e_t)

                loss = loss + model.loss(
                    p_logits, q_logits,
                    r_log,     r_t,
                    v_log,     r_t,
                    c_log,     c_t
                )

            # Latent overshooting
            for h in range(1, overshoot+1):
                if h >= T: break
                so = model.init_state(B, device)
                for t in range(T-h):
                    a_t = x[:, t, cat_size:]
                    so, p_logits, r_log, v_log, c_log = model.imagine(so, a_t)
                    r_t = r_seq[:, t+h].squeeze(-1)
                    c_t = c_seq[:, t+h].squeeze(-1)

                    loss = loss + model.loss(
                        p_logits, p_logits,
                        r_log,     r_t,
                        v_log,     r_t,
                        c_log,     c_t
                    ) / overshoot

            optim.zero_grad()
            loss.backward()

            # Adaptive gradient clipping
            total_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
            ]))
            grad_norm_ema = beta*grad_norm_ema + (1-beta)*total_norm.item()
            clip_grad_norm_(model.parameters(), max_norm=2*grad_norm_ema)

            optim.step()
            total_loss += loss.item()

        # Average loss for the epoch
        avg = total_loss / batch_count
        if ep % print_every == 0:
            print(f"[RSSM] epoch {ep}/{epochs}  loss={avg:.4f}")
        best_loss = min(best_loss, avg)

    # Save the model state
    model.optim_state = optim.state_dict()
