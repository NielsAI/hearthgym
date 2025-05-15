import torch
import torch.nn as nn

class MultiHeadAutoEncoder(nn.Module):
    """
    Class for a multi-head autoencoder that can handle both continuous and discrete
    features. The encoder consists of two separate branches for continuous and discrete features,
    which are then fused into a latent space. The decoder reconstructs the original input from the
    latent space by scattering the outputs back to their original positions.
    """

    def __init__(
        self,
        cont_input_dim: int,
        disc_input_dim: int,
        latent_dim: int,
        cont_hidden_dim: int = 128,
        disc_hidden_dim: int = 32
        ):
        """ Initialize the multi-head autoencoder.
        
        :param cont_input_dim: Number of continuous features.
        :param disc_input_dim: Number of discrete features.
        :param latent_dim: Dimension of the latent space.
        :cont_hidden_dim: Hidden dimension for the continuous encoder/decoder.
        :disc_hidden_dim: Hidden dimension for the discrete encoder/decoder.
        """
        
        super().__init__()

        # Set up the encoder and decoder networks
        self.cont_encoder = nn.Sequential(
            nn.Linear(cont_input_dim, cont_hidden_dim),
            nn.ReLU(),
            nn.Linear(cont_hidden_dim, cont_hidden_dim),
            nn.ReLU()
        )
        self.disc_encoder = nn.Sequential(
            nn.Linear(disc_input_dim, disc_hidden_dim),
            nn.ReLU(),
            nn.Linear(disc_hidden_dim, disc_hidden_dim),
            nn.ReLU()
        )
        self.fuse = nn.Linear(cont_hidden_dim + disc_hidden_dim, latent_dim)

        # Create decoder forward pass
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, cont_hidden_dim + disc_hidden_dim),
            nn.ReLU()
        )
        
        # Branch back to original dims
        self.cont_decoder = nn.Linear(cont_hidden_dim, cont_input_dim)
        self.disc_decoder = nn.Sequential(
            nn.Linear(disc_hidden_dim, disc_input_dim),
            nn.Sigmoid()
        )

        # Save dims for later use
        self.cont_hidden_dim = cont_hidden_dim
        self.disc_hidden_dim = disc_hidden_dim
        self.latent_dim = latent_dim

    def encode(self, obs, cont_idx, disc_idx):
        """
        Encode the input observation into a latent space representation.
        
        :param obs: [B, D] full observation
        :param cont_idx: 1-D tensor of indices for continuous features.
        :param disc_idx: 1-D tensor of indices for discrete features.
        :return: Latent space representation of the input observation.
        """
        
        # Slice the input observation into continuous and discrete parts
        cont_x = obs[:, cont_idx]
        disc_x = obs[:, disc_idx]
        
        # Pass the continuous and discrete parts through their respective encoders
        cont_h = self.cont_encoder(cont_x)
        disc_h = self.disc_encoder(disc_x)
        
        # Concatenate the outputs of the two branches and pass through the fusion layer
        z = self.fuse(torch.cat([cont_h, disc_h], dim=1))
        return z

    def decode(self, z, full_shape, cont_idx, disc_idx):
        """
        Reconstruct the *full* observation (same shape as input)
        by scattering branch outputs back to their positions.
        
        :param z: Latent space representation.
        :param full_shape: Shape of the full observation [B, D].
        :param cont_idx: 1-D tensor of indices for continuous features.
        :param disc_idx: 1-D tensor of indices for discrete features.
        :return: Reconstructed observation.
        """
        # Pass the latent space representation through the decoder
        fused   = self.decoder_fc(z)        
        cont_h  = fused[:, :self.cont_hidden_dim]
        disc_h  = fused[:, self.cont_hidden_dim:]

        cont_rec  = self.cont_decoder(cont_h)   # [B, |cont|]
        disc_rec  = self.disc_decoder(disc_h)   # [B, |disc|]

        # Create a zero tensor to hold the reconstructed observation
        B, D = full_shape
        recon = torch.zeros((B, D), device=z.device)

        # Scatter the reconstructed continuous and discrete parts back to their original positions
        cont_t = torch.as_tensor(cont_idx, dtype=torch.long, device=z.device)
        disc_t = torch.as_tensor(disc_idx, dtype=torch.long, device=z.device)

        recon.index_copy_(1, cont_t,  cont_rec)
        recon.index_copy_(1, disc_t,  disc_rec)
        return recon

    def forward(self, obs, cont_idx, disc_idx):
        """
        Run the forward pass of the autoencoder.
        
        :param obs: [B, D] full observation
        :param cont_idx: 1-D tensor of indices for continuous features.
        :param disc_idx: 1-D tensor of indices for discrete features.
        :return: Reconstructed observation and latent space representation.
        """
        
        # If the input indices are not tensors, convert them to tensors
        if not torch.is_tensor(cont_idx):
            cont_idx = torch.as_tensor(cont_idx, dtype=torch.long, device=obs.device)
        if not torch.is_tensor(disc_idx):
            disc_idx = torch.as_tensor(disc_idx, dtype=torch.long, device=obs.device)

        # Encode the input observation and reconstruct it
        z     = self.encode(obs, cont_idx, disc_idx)
        recon = self.decode(z, obs.shape, cont_idx, disc_idx)
        return recon, z
