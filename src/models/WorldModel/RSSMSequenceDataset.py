import torch
from torch.utils.data import IterableDataset
from models.WorldModel.utils import one_hot_encode, symlog

class RSSMSequenceDataset(IterableDataset):
    """
    Iterable dataset for training the RSSM. Streams features and targets
    without storing all in memory.
    """
    def __init__(self, world_model, episodes, sequence_length, cont_indices, disc_indices, num_actions, device):
        """
        Initialize the RSSMSequenceDataset.
        :param world_model: The world model used for simulating future states.
        :param episodes: List of episodes, where each episode is a list of (obs, act, rew, next_obs, done).
        :param sequence_length: Length of the sequences to generate.
        :param cont_indices: Indices of continuous features in the observation space.
        :param disc_indices: Indices of discrete features in the observation space.
        :param num_actions: Number of actions in the action space.
        :param device: The device to use for computations (CPU or GPU).
        """
        self.episodes        = episodes
        self.sequence_length = sequence_length
        self.cont_idx        = cont_indices
        self.disc_idx        = disc_indices
        self.num_actions     = num_actions
        self.device          = device
        self.world_model     = world_model

    def __iter__(self):
        # Iterate over the episodes and generate sequences
        for ep in self.episodes:
            T = len(ep)
            # Uniformly spaced windows
            for s in range(0, T - self.sequence_length, self.sequence_length):
                z_in, a_in, z_tgt, r_tgt, c_tgt = [], [], [], [], []
                for t in range(s, s + self.sequence_length):
                    obs, act, rew, next_obs, done = ep[t]
                    # Encode observation and next observation
                    with torch.no_grad():
                        z_t   = self.world_model.autoencoder.encode(
                            torch.as_tensor(obs, device=self.device).unsqueeze(0),
                            self.cont_idx, self.disc_idx
                        ).squeeze(0)
                        z_t1  = self.world_model.autoencoder.encode(
                            torch.as_tensor(next_obs, device=self.device).unsqueeze(0),
                            self.cont_idx, self.disc_idx
                        ).squeeze(0)
                        
                    # Append features and targets to the lists
                    z_in.append(symlog(z_t))
                    a_in.append(
                        torch.as_tensor(
                            one_hot_encode(act, self.num_actions),
                            device=self.device)
                    )
                    z_tgt.append(symlog(z_t1))
                    r_tgt.append(symlog(torch.tensor(rew, device=self.device)))
                    c_tgt.append(torch.tensor(0.0 if done else 1.0, device=self.device))

                # Stack the features and targets
                X = torch.cat([torch.stack(z_in), torch.stack(a_in)], dim=1)
                Y = torch.stack(z_tgt)
                R = torch.stack(r_tgt).unsqueeze(-1)
                C = torch.stack(c_tgt).unsqueeze(-1)
                yield X, Y, R, C