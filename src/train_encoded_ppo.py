from models.EncodedPPO.RunEncodedPPO import train_encoded_ppo
from fireplace import cards 
from fireplace.logging import log
import logging

log.setLevel(logging.ERROR)

# This is to avoid the Simplex error when using the MaskablePPO model:
from torch.distributions import Distribution 
Distribution.set_default_validate_args(False)

# Setup gymnasium
import gymnasium

gymnasium.register(
    id="hearthstone_env/HearthGym-v0",
    entry_point="env.hearthstone.HearthGym:HearthstoneEnv"
)

cards.db.initialize()

if __name__ == "__main__":
    train_encoded_ppo()
