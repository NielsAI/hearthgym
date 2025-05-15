import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
from models.WorldModel.utils import mdn_loss_fn

# --- Analysis Functions ---
def analyze_autoencoder(autoencoder, test_states, device, plot_dir, cont_indices=None, disc_indices=None):
    autoencoder.eval()
    test_tensor = torch.tensor(test_states, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon, latent = autoencoder(test_tensor, cont_indices, disc_indices)
    recon = recon.cpu().numpy()
    latent = latent.cpu().numpy()
    
    mse = np.mean((test_states - recon) ** 2)
    print(f"Autoencoder Average Reconstruction MSE: {mse:.4f}")
    
    # Save sample reconstructions.
    n_samples = min(5, len(test_states))
    for i in range(n_samples):
        plt.figure(figsize=(10, 2))
        plt.plot(test_states[i], label='Original')
        plt.plot(recon[i], label='Reconstruction')
        plt.legend()
        plt.title(f"Sample {i}")
        sample_path = os.path.join(plot_dir, f"autoencoder_sample_{i}.png")
        plt.savefig(sample_path)
        plt.close()
    
    # PCA on latent codes.
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent)
    plt.figure(figsize=(6, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6)
    plt.title("PCA of Latent Space")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    pca_path = os.path.join(plot_dir, "latent_space_pca.png")
    plt.savefig(pca_path)
    plt.close()
    print(f"Autoencoder plots saved to {plot_dir}")

def analyze_mdn_rnn(mdn_rnn, sequences, device, plot_dir):
    mdn_rnn.eval()
    total_loss = 0.0
    total_timesteps = 0
    for (input_seq, target_seq, reward_target) in sequences:
        input_tensor = torch.tensor(input_seq, dtype=torch.float32, device=device).unsqueeze(0)
        target_tensor = torch.tensor(target_seq, dtype=torch.float32, device=device).unsqueeze(0)
        reward_target_tensor = torch.tensor(reward_target, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(2)
        
        with torch.no_grad():
            logits, means, std, reward_pred, hidden, log_std = mdn_rnn(input_tensor)
            loss = mdn_loss_fn(logits, means, std, target_tensor, reward_pred, reward_target_tensor)
        timesteps = input_tensor.shape[1]
        total_loss += loss.item() * timesteps
        total_timesteps += timesteps
    avg_loss = total_loss / total_timesteps
    print(f"MDN-RNN Average Loss per timestep: {avg_loss:.4f}")
    
    # Qualitative analysis: sample rollout from first sequence.
    input_seq, target_seq, _ = sequences[0]
    input_tensor = torch.tensor(input_seq, dtype=torch.float32, device=device).unsqueeze(0)
    hidden = None
    sampled_latents = []
    mdn_rnn.eval()
    with torch.no_grad():
        seq_len = input_tensor.shape[1]
        for t in range(seq_len):
            current_input = input_tensor[:, t:t+1, :]
            logits, means, std, reward, hidden, log_std = mdn_rnn(current_input, hidden)
            logits = logits.squeeze(0).squeeze(0)
            means = means.squeeze(0).squeeze(0)
            std = std.squeeze(0).squeeze(0)
            pi = F.softmax(logits, dim=0).cpu().numpy()
            comp = np.random.choice(len(pi), p=pi)
            sampled_latent = np.random.normal(loc=means[comp].cpu().numpy(), scale=std[comp].cpu().numpy())
            sampled_latents.append(sampled_latent)
    sampled_latents = np.array(sampled_latents)
    target_seq_np = np.array(target_seq)
    
    # Save a plot comparing sampled vs. target for the first latent dimension.
    plt.figure(figsize=(8, 4))
    plt.plot(sampled_latents[:, 0], label='Sampled latent dim 0')
    plt.plot(target_seq_np[:, 0], label='Target latent dim 0')
    plt.legend()
    plt.title("MDN-RNN Sample vs. Target (Latent Dimension 0)")
    mdn_plot_path = os.path.join(plot_dir, "mdn_rnn_latent0_comparison.png")
    plt.savefig(mdn_plot_path)
    plt.close()
    print(f"MDN-RNN analysis plots saved to {plot_dir}")

def analyze_dream_rollout(world_model, start_obs, sequence_length, fixed_action, device, plot_dir, cont_indices=None, disc_indices=None):
    """
    Starting from a real observation, use the world model to produce a 'dream' sequence.
    The procedure is:
      1. Encode the start_obs using the autoencoder (to obtain the initial latent code).
      2. For each timestep in the sequence:
           a. Concatenate the current latent code with a one-hot representation of a fixed action.
           b. Feed that into the MDN-RNN to predict parameters for the next latent code.
           c. Sample the next latent code from the predicted mixture.
      3. Decode each latent code using the autoencoder decoder to produce 'dream' observations.
    It then plots:
      - The evolution of the first latent dimension over time.
      - PCA scatter plots comparing the dream latent sequence with the start latent code.
    """
    # Ensure the world model is in evaluation mode.
    world_model.autoencoder.eval()
    world_model.mdn_rnn.eval()
    
    # 1. Encode start_obs to obtain initial latent code.
    obs_tensor = torch.tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)  # shape (1, obs_dim)
    with torch.no_grad():
        initial_latent = world_model.autoencoder.encode(obs_tensor, cont_indices, disc_indices)  # shape (1, latent_dim)
    current_latent = initial_latent.squeeze(0)  # shape (latent_dim)
    
    # Prepare to collect latent sequence.
    latent_sequence = [current_latent.cpu().numpy()]
    
    # Fixed action: one-hot encode the fixed_action.
    def one_hot_encode(action, num_actions):
        one_hot = np.zeros(num_actions, dtype=np.float32)
        one_hot[int(action)] = 1.0
        return one_hot
    
    num_actions = fixed_action.shape[0] if isinstance(fixed_action, np.ndarray) else world_model.mdn_rnn.action_dim
    fixed_action_onehot = torch.tensor(one_hot_encode(fixed_action, num_actions), dtype=torch.float32, device=device).unsqueeze(0)  # shape (1, num_actions)
    
    hidden = None
    # 2. Roll out the dream for sequence_length timesteps.
    for t in range(sequence_length):
        # Current latent is shape (latent_dim) -> make it (1, latent_dim)
        current_latent_tensor = current_latent.unsqueeze(0)  # (1, latent_dim)
        # Expand to (batch, 1, latent_dim)
        current_latent_tensor = current_latent_tensor.unsqueeze(1)
        # Concatenate fixed action to latent: shape becomes (1, 1, latent_dim + num_actions)
        mdn_input = torch.cat([current_latent_tensor, fixed_action_onehot.unsqueeze(1)], dim=2)
        
        with torch.no_grad():
            logits, means, std, reward, hidden, log_std = world_model.mdn_rnn(mdn_input, hidden)
        
        # Squeeze to remove time and batch dimensions.
        logits = logits.squeeze(0).squeeze(0)
        means = means.squeeze(0).squeeze(0)
        std   = std.squeeze(0).squeeze(0)
        
        # Compute mixture probabilities.
        probs = F.softmax(logits, dim=0).cpu().numpy()
        # Sample one of the mixtures.
        comp = np.random.choice(len(probs), p=probs)
        # Sample next latent.
        next_latent_np = np.random.normal(loc=means[comp].cpu().numpy(), scale=std[comp].cpu().numpy())
        next_latent = torch.tensor(next_latent_np, dtype=torch.float32, device=device)
        latent_sequence.append(next_latent.cpu().numpy())
        current_latent = next_latent

    latent_sequence = np.array(latent_sequence)  # shape (sequence_length+1, latent_dim)
    
    # 3. Decode the latent sequence to observations.
    decoded_observations = []
    for z in latent_sequence:
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            decoded = world_model.autoencoder.decode(z_tensor, (1, start_obs.shape[0]), cont_indices, disc_indices)
        decoded_observations.append(decoded.squeeze(0).cpu().numpy())
    decoded_observations = np.array(decoded_observations)
    
    # 4. Plot the evolution of the first latent dimension.
    plt.figure(figsize=(8, 4))
    plt.plot(latent_sequence[:, 0], marker='o')
    plt.title("Dream Rollout - Latent Dimension 0 Evolution")
    plt.xlabel("Timestep")
    plt.ylabel("Latent Dim 0 value")
    dream_latent_path = os.path.join(plot_dir, "dream_latent_dimension0.png")
    plt.savefig(dream_latent_path)
    plt.close()
    
    # 5. PCA Scatter Plot of the latent sequence.
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_sequence)
    plt.figure(figsize=(6,6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=np.arange(latent_sequence.shape[0]), cmap='viridis')
    plt.title("PCA of Dream Latent Trajectory")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    colorbar = plt.colorbar()
    colorbar.set_label("Timestep")
    dream_pca_path = os.path.join(plot_dir, "dream_latent_pca.png")
    plt.savefig(dream_pca_path)
    plt.close()
    
    # 6. (Optional) Plot evolution of one dimension from the decoded observations.
    plt.figure(figsize=(8,4))
    # Here, we plot the first 50 elements of the decoded observation vector if it is high-dimensional.
    plt.plot(decoded_observations[:, 0], marker='o')
    plt.title("Dream Rollout - Decoded Observation (Dimension 0)")
    plt.xlabel("Timestep")
    plt.ylabel("Decoded Value (Dim 0)")
    decoded_plot_path = os.path.join(plot_dir, "dream_decoded_dimension0.png")
    plt.savefig(decoded_plot_path)
    plt.close()
    
    print(f"Dream rollout analysis plots saved to {plot_dir}.")
    return latent_sequence, decoded_observations

def assess_dream_quality(world_model, real_states, dream_latent_seq, decoded_obs, device, plot_dir, cont_indices=None, disc_indices=None):
    """
    Compare the latent codes of real observations vs. a dream rollout and compute metrics.
    
    Parameters:
      world_model: An object with autoencoder and mdn_rnn attributes.
      real_states: A NumPy array of real observations, shape (N, obs_dim).
      dream_latent_seq: A NumPy array of latent codes from a dream rollout, shape (T, latent_dim).
      decoded_obs: A NumPy array of dream observations produced by decoding dream_latent_seq, shape (T, obs_dim).
      device: torch.device (e.g., cuda or cpu).
      plot_dir: Directory to save generated plots.
      
    Returns:
      A dictionary containing computed metrics:
        - 'latent_mean_diff': L2 difference between the means of real and dream latent codes.
        - 'latent_cov_diff': Frobenius norm of the difference between the covariance matrices.
        - 'avg_nearest_neighbor_dist': Average nearest–neighbor distance from each dream latent to the real latent codes.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    # 1. Compute latent representations for a sample (or all) of the real observations.
    world_model.autoencoder.eval()
    with torch.no_grad():
        real_tensor = torch.tensor(real_states, dtype=torch.float32, device=device)
        # Pass in batches if data is very large.
        real_latents = world_model.autoencoder.encode(real_tensor, cont_indices, disc_indices).cpu().numpy()
    
    # 2. Calculate mean and covariance for both real and dream latents.
    real_mean = np.mean(real_latents, axis=0)
    dream_mean = np.mean(dream_latent_seq, axis=0)
    mean_diff = np.linalg.norm(real_mean - dream_mean)
    
    real_cov = np.cov(real_latents.T)
    dream_cov = np.cov(dream_latent_seq.T)
    cov_diff = np.linalg.norm(real_cov - dream_cov, ord='fro')
    
    # 3. Compute average nearest neighbor distance from each dream latent to the set of real latents.
    # Create a distance matrix between dream_latent_seq and real_latents.
    dists = distance_matrix(dream_latent_seq, real_latents)
    nearest_dists = np.min(dists, axis=1)
    avg_nn_dist = np.mean(nearest_dists)
    
    metrics = {
        'latent_mean_diff': mean_diff,
        'latent_cov_diff': cov_diff,
        'avg_nearest_neighbor_dist': avg_nn_dist,
    }
    
    print("Dream Quality Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # 4. Create PCA plots overlaying real and dream latent codes.
    pca = PCA(n_components=2)
    combined_latents = np.concatenate([real_latents, dream_latent_seq], axis=0)
    combined_2d = pca.fit_transform(combined_latents)
    real_2d = combined_2d[:real_latents.shape[0], :]
    dream_2d = combined_2d[real_latents.shape[0]:, :]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(real_2d[:, 0], real_2d[:, 1], alpha=0.3, label='Real Latents')
    plt.scatter(dream_2d[:, 0], dream_2d[:, 1], color='red', alpha=0.7, label='Dream Latents')
    plt.legend()
    plt.title("PCA of Real vs. Dream Latents")
    pca_path = os.path.join(plot_dir, "pca_real_vs_dream.png")
    plt.savefig(pca_path)
    plt.close()
    
    # 5. Plot the evolution of the first latent dimension over time for the dream rollout.
    plt.figure(figsize=(8, 4))
    plt.plot(dream_latent_seq[:, 0], marker='o')
    plt.title("Dream Rollout - Latent Dimension 0 Evolution")
    plt.xlabel("Timestep")
    plt.ylabel("Latent Dim 0")
    latent_time_path = os.path.join(plot_dir, "dream_latent_dim0_evolution.png")
    plt.savefig(latent_time_path)
    plt.close()
    
    # 6. Plot the evolution of the first dimension of the decoded observations.
    plt.figure(figsize=(8, 4))
    plt.plot(decoded_obs[:, 0], marker='o')
    plt.title("Dream Rollout - Decoded Observation Dimension 0")
    plt.xlabel("Timestep")
    plt.ylabel("Decoded Value (Dim 0)")
    decoded_path = os.path.join(plot_dir, "dream_decoded_dim0_evolution.png")
    plt.savefig(decoded_path)
    plt.close()
    
    return metrics

def render_observation(obs, feature_names=None):
    """
    Convert a decoded observation vector into a human-readable string.
    This function assumes that the observation is in the same format as
    produced by your _get_obs() function.
    
    Optionally, provide feature_names (a list of strings) so that you can
    show name-value pairs.
    
    For more complex rendering (e.g., reconstructing a textual game state),
    you may need to parse the observation vector into its components.
    """
    # For demonstration, we simply zip feature names with their values.
    if feature_names is None:
        # If not provided, just list indices and values.
        lines = [f"Feature {i}: {val:.2f}" for i, val in enumerate(obs)]
    else:
        lines = [f"{fname}: {val:.2f}" for fname, val in zip(feature_names, obs) if "embedding" not in fname]
    
    return "\n".join(lines)

def analyze_dream_rollout_with_render(
    world_model, start_obs, sequence_length, fixed_action, device, plot_dir, feature_names=None,
    cont_indices=None, disc_indices=None
    ):
    """
    Generate a dream rollout, and for each timestep, decode the latent state to an observation,
    then convert it to a human-readable string using render_observation().
    
    Returns:
      A list of strings, one per timestep, that you can inspect manually.
    """
    # Ensure evaluation mode.
    world_model.autoencoder.eval()
    world_model.mdn_rnn.eval()
    
    # Encode the start observation.
    obs_tensor = torch.tensor(start_obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        initial_latent = world_model.autoencoder.encode(obs_tensor, cont_indices, disc_indices)  # (1, latent_dim)
    current_latent = initial_latent.squeeze(0)
    
    rendered_sequence = []
    latent_sequence = [current_latent.cpu().numpy()]
    
    # Fixed action: one-hot encode the fixed_action.
    def one_hot_encode(action, num_actions):
        one_hot = np.zeros(num_actions, dtype=np.float32)
        one_hot[int(action)] = 1.0
        return one_hot
    
    num_actions = fixed_action.shape[0] if isinstance(fixed_action, np.ndarray) else world_model.mdn_rnn.action_dim
    fixed_action_onehot = torch.tensor(one_hot_encode(fixed_action, num_actions), dtype=torch.float32, device=device).unsqueeze(0)
    
    hidden = None
    for t in range(sequence_length):
        # Prepare input for the MDN-RNN.
        current_latent_tensor = current_latent.unsqueeze(0).unsqueeze(1)  # (1, 1, latent_dim)
        mdn_input = torch.cat([current_latent_tensor, fixed_action_onehot.unsqueeze(1)], dim=2)
        with torch.no_grad():
            logits, means, std, reward, hidden, log_std = world_model.mdn_rnn(mdn_input, hidden)
        logits = logits.squeeze(0).squeeze(0)
        means = means.squeeze(0).squeeze(0)
        std   = std.squeeze(0).squeeze(0)
        
        pi = F.softmax(logits, dim=0).cpu().numpy()
        comp = np.random.choice(len(pi), p=pi)
        next_latent_np = np.random.normal(loc=means[comp].cpu().numpy(), scale=std[comp].cpu().numpy())
        next_latent = torch.tensor(next_latent_np, dtype=torch.float32, device=device)
        
        latent_sequence.append(next_latent.cpu().numpy())
        current_latent = next_latent
        
        # Decode the current latent state.
        z_tensor = current_latent.unsqueeze(0)
        with torch.no_grad():
            decoded_obs = world_model.autoencoder.decode(z_tensor, (1, start_obs.shape[0]), cont_indices, disc_indices)
        decoded_obs_np = decoded_obs.squeeze(0).cpu().numpy()
        # Render the decoded observation.
        rendered_str = render_observation(decoded_obs_np, feature_names=feature_names)
        rendered_sequence.append(rendered_str)
    
    # Optionally, you could also generate plots from latent_sequence as before.
    # (Omitted here for brevity.)
    
    # Save the rendered sequence to a file.
    render_log_path = os.path.join(plot_dir, "dream_rollout_render.txt")
    with open(render_log_path, "w") as f:
        for t, r in enumerate(rendered_sequence):
            f.write(f"Time step {t}:\n")
            f.write(r + "\n\n")
    
    print(f"Dream rollout render log saved to {render_log_path}")
    return latent_sequence, rendered_sequence
