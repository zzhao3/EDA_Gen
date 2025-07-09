import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch import optim
import numpy as np
import argparse
import logging
from tqdm import tqdm
from types import SimpleNamespace
import matplotlib.pyplot as plt

# Assuming a 1D U-Net is available in modules.modules
from modules.modules import UNet_1D as UNet

from model.eda_diffusion import EDADiffusion

# --- Setup Logging ---
def setup_logging(run_name, output_path):
    """Configures logging to file and console."""
    log_dir = os.path.join(output_path, run_name)
    log_path = os.path.join(log_dir, "train.log")
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler to save logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # Console handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)

# --- Configuration ---
def get_config():
    return SimpleNamespace(
        run_name="EDA_Diffusion_Training",
        epochs=100,
        batch_size=16,
        seed=42,
        device="cuda:1" if torch.cuda.is_available() else "cpu", # Default to GPU 1
        lr=1e-4,
        noise_steps=1000,
        sequence_length=3840, # 60s * 64Hz
        num_channels=1, # EDA only
        # --- Paths ---
        dataset_path="/fd24T/zzhao3/EDA/preprocessed_data/60s_0.25s",
        output_path="/fd24T/zzhao3/EDA/results/eda_diffusion",
        # --- Fold Management ---
        fold_for_training=17,
        # --- Dataset Size ---
        dataset_percentage=1.0 # Use 100% of the data by default
    )

# --- Data Loading ---
class WESADDataset(Dataset):
    """Loads the 'Y_train' EDA signals from a single preprocessed WESAD .npz fold."""
    def __init__(self, data_path, fold_number):
        super().__init__()
        
        file_path = os.path.join(data_path, f"fold_{fold_number}.npz")
        logging.info(f"Loading training data from: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with np.load(file_path) as data:
            # Y_train contains signals from all subjects except the one for the fold number.
            signals = data['Y_train']
            # Add a channel dimension: (num_windows, 1, seq_len)
            self.signals = signals[:, np.newaxis, :]
        
        logging.info(f"Loaded {self.signals.shape[0]} windows from fold {fold_number}.")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return torch.from_numpy(self.signals[idx]).float()

# --- Training and Sampling ---
def save_plot(generated_signals, ground_truth_signals, path, num_samples=3):
    """Saves a plot comparing generated signals and ground truth signals by overlaying them."""
    fig, axes = plt.subplots(num_samples, 1, figsize=(15, 4 * num_samples), sharex=True, sharey=True)
    if num_samples == 1:
        axes = [axes] # Ensure axes is always iterable

    gen_signals_to_plot = generated_signals.cpu().numpy()
    gt_signals_to_plot = ground_truth_signals.cpu().numpy()

    for i in range(num_samples):
        ax = axes[i]
        ax.plot(gt_signals_to_plot[i, 0, :], 'g', label='Ground Truth')
        ax.plot(gen_signals_to_plot[i, 0, :], 'b', label='Generated', alpha=0.7)
        ax.set_title(f"Comparison Sample {i+1}")
        ax.set_ylabel("Signal Value")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle("Generated vs. Ground Truth (Overlay)", fontsize=16)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    plt.savefig(path)
    plt.close()

def plot_loss_curves(train_losses, val_losses, path):
    """Saves a plot of training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train(config):
    device = config.device
    
    # Data
    dataset = WESADDataset(config.dataset_path, config.fold_for_training)

    # Reduce dataset size based on percentage argument
    if config.dataset_percentage < 1.0:
        original_size = len(dataset)
        subset_size = int(original_size * config.dataset_percentage)
        generator = torch.Generator().manual_seed(config.seed)
        dataset, _ = torch.utils.data.random_split(dataset, [subset_size, original_size - subset_size], generator=generator)
        logging.info(f"Using {config.dataset_percentage*100:.0f}% of the dataset: {len(dataset)} out of {original_size} windows.")

    # Split into training and validation sets (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    logging.info(f"Training set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Model and Diffusion
    model = UNet(in_channels=config.num_channels, out_channels=config.num_channels).to(device)
    diffusion = EDADiffusion(
        noise_steps=config.noise_steps,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        device=device
    )
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    
    # --- Log Initial Information ---
    logging.info("--- Starting New Training Run ---")
    logging.info(f"Run Name: {config.run_name}")
    logging.info("Configuration:")
    for key, value in vars(config).items():
        logging.info(f"  {key}: {value}")
    logging.info("\nModel Architecture:")
    logging.info(model)
    logging.info(f"\nTraining set size: {len(train_dataset)}")
    logging.info(f"Validation set size: {len(val_dataset)}\n")

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    run_output_path = os.path.join(config.output_path, config.run_name)
    
    for epoch in range(config.epochs):
        logging.info(f"--- Epoch {epoch+1}/{config.epochs} ---")
        
        # --- Training Loop ---
        model.train()
        train_loss = 0
        pbar_train = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        for signals in pbar_train:
            signals = signals.to(device)
            t = diffusion.sample_timesteps(signals.shape[0]).to(device)
            
            x_t, noise = diffusion.noise_signals(signals, t)
            predicted_noise = model(x_t, t)
            
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar_train.set_postfix(MSE=loss.item())

        avg_train_loss = train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logging.info(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar_val = tqdm(val_dataloader, desc=f"Validating Epoch {epoch+1}")
            for signals in pbar_val:
                signals = signals.to(device)
                t = diffusion.sample_timesteps(signals.shape[0]).to(device)
                x_t, noise = diffusion.noise_signals(signals, t)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
                val_loss += loss.item()
                pbar_val.set_postfix(MSE=loss.item())
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        logging.info(f"Epoch {epoch+1} average validation loss: {avg_val_loss:.4f}")

        # --- Save Losses Every Epoch ---
        loss_path = os.path.join(run_output_path, "losses.npz")
        np.savez(loss_path, train_losses=np.array(train_losses), val_losses=np.array(val_losses))

        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(run_output_path, "models", "best_ckpt.pt")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best validation loss. Saved model to {best_model_path}")
        
        # --- Sampling and Checkpoint Saving ---
        if (epoch + 1) % 10 == 0:
            logging.info("Sampling new signals for comparison...")
            
            # Create a directory for this epoch's samples
            epoch_sample_dir = os.path.join(run_output_path, "samples", f"epoch_{epoch+1}")
            os.makedirs(epoch_sample_dir, exist_ok=True)
            
            # Get random ground truth samples from the validation set to compare against
            val_indices = np.random.choice(len(val_dataset), size=3 * 3, replace=False) # 3 plots * 3 samples
            ground_truth_batch = torch.stack([val_dataset[i] for i in val_indices]).to(device)

            for i in range(3): # Generate 3 comparison plots
                # Generate a batch of new samples
                sampled_signals = diffusion.sample(model, n=3)
                
                # Select a slice of ground truth samples for this specific plot
                start_idx = i * 3
                end_idx = start_idx + 3
                gt_samples_for_plot = ground_truth_batch[start_idx:end_idx]

                # Save the comparison plot
                plot_path = os.path.join(epoch_sample_dir, f"comparison_{i+1}.png")
                save_plot(sampled_signals, gt_samples_for_plot, plot_path, num_samples=3)
            
            logging.info(f"Saved 3 sample comparison plots to {epoch_sample_dir}")
            
            # Plot the loss curve up to this point
            loss_curve_path = os.path.join(epoch_sample_dir, "loss_curve.png")
            plot_loss_curves(train_losses, val_losses, loss_curve_path)
            logging.info(f"Saved intermediate loss curve plot to {loss_curve_path}")

            # Save periodic checkpoint
            model_path = os.path.join(run_output_path, "models", f"ckpt_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved periodic checkpoint to {model_path}")

    # --- Save Final Plot ---
    logging.info("Training complete. Plotting final loss curves.")
    plot_path = os.path.join(run_output_path, "loss_curves.png")
    plot_loss_curves(train_losses, val_losses, plot_path)
    logging.info(f"Saved loss curves plot to {plot_path}")

# --- Main Execution ---
def main():
    config = get_config()
    parser = argparse.ArgumentParser(description='Train an EDA Diffusion Model')
    parser.add_argument('--run_name', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument('--epochs', type=int, default=config.epochs)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--device', type=str, default=config.device, help="Specify the device (e.g., 'cuda:0', 'cuda:1', 'cpu')")
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path)
    parser.add_argument('--fold_for_training', type=int, default=config.fold_for_training, help="The fold number to train on (e.g., 17 loads fold_17.npz).")
    parser.add_argument('--dataset_percentage', type=float, default=config.dataset_percentage, help="Percentage of the dataset to use (0.0 to 1.0). Default: 1.0")

    args = parser.parse_args()
    
    # Update config with parsed args
    for key, value in vars(args).items():
        setattr(config, key, value)
    
    # Setup output directories and logging
    run_output_dir = os.path.join(config.output_path, args.run_name)
    os.makedirs(os.path.join(run_output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_output_dir, "samples"), exist_ok=True)
    setup_logging(config.run_name, config.output_path)
        
    train(config)

if __name__ == '__main__':
    main() 