from torch.utils.data import IterableDataset
import torch
from models.WorldModel.utils import symlog, one_hot_encode
import numpy as np

class LegalitySequenceDataset(IterableDataset):
    """
    Iterable dataset for legality net training. Streams features and masks
    without storing all in memory.

    Each element is (feature_vector, mask_vector), where:
      - feature_vector: concatenated [z, h] from RSSM state (size Z+H)
      - mask_vector: valid action mask for that step
    """
    def __init__(
        self,
        mask_data,       # list of episodes: list of (obs, act, mask)
        world_model,     # has .autoencoder & .rssm trained
        num_actions: int,
        cont_indices=None,
        disc_indices=None,
        device: str = "cuda"
    ):
        self.mask_data     = mask_data
        self.wm            = world_model
        self.num_actions   = num_actions
        self.cont_idx      = cont_indices
        self.disc_idx      = disc_indices
        self.device        = device

    def __iter__(self):
        ae   = self.wm.autoencoder.to(self.device).eval()
        rssm = self.wm.rssm.to(self.device).eval()

        for episode in self.mask_data:
            # reset RSSM
            state = rssm.init_state(1, self.device)
            prev_a = torch.zeros(1, self.num_actions, device=self.device)

            for obs, act, mask_packed in episode:
                # Unpack mask_packed
                mask = np.unpackbits(mask_packed, bitorder="big")[:self.num_actions].astype(bool)
                mask = mask.astype(bool)
                
                # Encode observation
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
                
                with torch.no_grad():
                    z_obs = ae.encode(obs_t, self.cont_idx, self.disc_idx)
                z_star = symlog(z_obs)

                # RSSM observe step
                with torch.no_grad():
                    state, *_ = rssm.observe(state, prev_a, z_star)

                # Build feature: [z, h]
                h, z = state
                feature = torch.cat([z.squeeze(0), h.squeeze(0)], dim=-1)

                # Get Mask tensor
                mask_vec = torch.as_tensor(mask, dtype=torch.float32)

                yield feature.cpu(), mask_vec

                # Prepare next prev_a
                a_onehot = one_hot_encode(act, self.num_actions)
                prev_a = torch.as_tensor(
                    a_onehot, dtype=torch.float32,
                    device=self.device).unsqueeze(0)
