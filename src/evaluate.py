import os
import torch
import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt

# Add src to path to allow project-level imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.modules import UNet_1D as UNet
from model.eda_diffusion import EDADiffusion
from utils import get_config, WESADDataset, plot_reconstructions

def evaluate(args):
    """Loads a trained model and performs one-to-one reconstruction on validation data."""
    config = get_config()
    device = args.device
    num_samples_per_plot = 3
    num_plots = 2
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_path)), "reconstructions")
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

    dataset = WESADDataset(config.dataset_path, config.fold_for_training)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(config.seed)
    _, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    model = UNet(in_channels=config.num_channels, out_channels=config.num_channels).to(device)
    logging.info(f"Loading model checkpoint from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    diffusion = EDADiffusion(
        noise_steps=config.noise_steps,
        sequence_length=config.sequence_length,
        num_channels=config.num_channels,
        device=device
    )

    for i in range(num_plots):
        logging.info(f"--- Generating Plot {i+1}/{num_plots} ---")
        
        # Get random ground truth samples
        indices = torch.randperm(len(val_dataset))[:num_samples_per_plot]
        ground_truth_batch = torch.stack([val_dataset.dataset[val_dataset.indices[i]] for i in indices.tolist()]).to(device)

        # Reconstruct them
        T = torch.tensor([config.noise_steps - 1] * num_samples_per_plot, device=device)
        noisy_signals, _ = diffusion.q_sample(ground_truth_batch, T)
        reconstructions = diffusion.sample(model, n=num_samples_per_plot, x_start=noisy_signals)
        
        # Plot and save
        plot_path = os.path.join(output_dir, f"reconstruction_comparison_plot_{i+1}.png")
        plot_reconstructions(ground_truth_batch, reconstructions, plot_path, num_samples=num_samples_per_plot)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained diffusion model by reconstructing signals.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file). e.g., results/eda_diffusion/RUN_NAME/models/best_ckpt.pt')
    parser.add_argument('--device', type=str, default="cuda:1", help="Device to use ('cuda:0', 'cpu', etc.)")
    
    args = parser.parse_args()
    evaluate(args) 