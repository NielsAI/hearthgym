# Load modules
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from agents.GreedyAgent import GreedyAgent
from agents.RandomAgent import RandomAgent
from env.hearthstone.HearthGym import MAX_HAND_SIZE, MAX_NUM_CHARACTERS, MAX_TOTAL_CHARACTERS, MAX_DISCOVER_OPTIONS, MAX_GENERIC_CHOICES

def action_to_index(action: dict) -> int:
        """Convert an action dict to a flat index using precomputed multipliers."""
        # Precompute multipliers
        m1 = MAX_HAND_SIZE
        m2 = m1 * MAX_NUM_CHARACTERS
        m3 = m2 * MAX_TOTAL_CHARACTERS
        m4 = m3 * MAX_DISCOVER_OPTIONS
        m5 = m4 * MAX_GENERIC_CHOICES
        
        # Flatten the action in the given order
        idx = action['action_type']
        idx = idx * MAX_HAND_SIZE + action['card_index']
        idx = idx * MAX_NUM_CHARACTERS + action['attacker_index']
        idx = idx * MAX_TOTAL_CHARACTERS + action['target_index']
        idx = idx * MAX_DISCOVER_OPTIONS + action['discover_index']
        idx = idx * MAX_GENERIC_CHOICES + action['choose_index']
        return idx
    
def collect_training_data(env, num_episodes=50, sampling_agent=None, score_method=None):
    data = []
    
    if sampling_agent == "GreedyAgent":
        sampling_agent = GreedyAgent(score_method=score_method)
    else:
        sampling_agent = RandomAgent()  
    
    for ep in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False
        episode_data = []
        while not done:
            valid_actions, action_mask = env.get_valid_actions()
            action = sampling_agent.act(observation=obs, valid_actions=valid_actions, action_mask=action_mask, env=env)
            next_obs, reward, done, truncated, info = env.step(action)
            episode_data.append((obs, action, reward, next_obs, done))
            obs = next_obs
        data.append(episode_data)            
    return data


def train_autoencoder(autoencoder,
                      training_states,
                      num_epochs=50,
                      batch_size=128,
                      lr=1e-3,
                      device='cuda',
                      cont_indices=None,
                      disc_indices=None,
                      lambda_disc=1.0):

    optim = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    autoencoder.train().to(device)

    data = torch.as_tensor(training_states, dtype=torch.float32, device=device)
    N = data.size(0)

    for epoch in range(num_epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            idx  = perm[i:i+batch_size]
            batch = data[idx]

            optim.zero_grad()
            recon, _ = autoencoder(batch, cont_indices, disc_indices)

            tgt_c = batch[:, cont_indices]
            tgt_d = batch[:, disc_indices]

            rec_c = recon[:, cont_indices]
            rec_d = recon[:, disc_indices]

            loss = F.mse_loss(rec_c, tgt_c) + \
                   lambda_disc * F.binary_cross_entropy(rec_d, tgt_d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5.0)
            optim.step()

            epoch_loss += loss.item() * batch.size(0)

        print(f"[AE] epoch {epoch+1:3d}/{num_epochs}   "
              f"loss = {epoch_loss/N:8.4f}")

def mask_fn(env):
    _, action_mask = env.get_valid_actions()
    # action_mask is currently shape (102400,).

    # Double-check dtype
    action_mask = action_mask.astype(np.int8)

    return action_mask