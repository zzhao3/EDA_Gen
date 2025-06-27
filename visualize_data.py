import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

# --- Configuration ---
PREPROCESSED_DIR = '/fd24T/zzhao3/EDA/preprocessed_data'
NUM_SAMPLES_TO_PLOT = 4
FS = 64 # The target sample rate from the preprocessing script

# --- Label Mapping ---
LABEL_DICT = {
    1: 'Baseline',
    2: 'Stress',
    3: 'Amusement'
}

def plot_fold_data(fold_number):
    """
    Loads a preprocessed data fold and plots a few random samples
    from the test set.
    """
    # 1. Construct the file path and load the data
    file_path = os.path.join(PREPROCESSED_DIR, f'fold_{fold_number}.npz')
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at {file_path}")
        return

    print(f"Loading data from {file_path}...")
    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # 2. Extract the necessary arrays from the test set
    X_test = data['X_test']
    Y_test = data['Y_test']
    L_test = data['L_test']
    feature_names = data['feature_names']

    # --- Print the signal names ---
    print("\nPlotting the following signals:")
    for name in feature_names:
        print(f"  - {name}")
    print("  - Target EDA (on separate axis)\n")
    # --------------------------------

    if X_test.size == 0:
        print("No test data available in this fold to plot.")
        return
        
    # 3. Select random samples to visualize
    num_test_samples = X_test.shape[0]
    if num_test_samples == 0:
        print("No samples in the test set.")
        return
        
    plot_indices = np.random.choice(
        num_test_samples,
        size=min(NUM_SAMPLES_TO_PLOT, num_test_samples),
        replace=False
    )
    
    print(f"Plotting {len(plot_indices)} random samples...")

    # 4. Create the plots
    fig, axes = plt.subplots(
        len(plot_indices), 1, 
        figsize=(12, 3 * len(plot_indices)), 
        sharex=True
    )
    if len(plot_indices) == 1:
        axes = [axes] # Ensure axes is always iterable

    time_axis = np.linspace(0, X_test.shape[1] / FS, num=X_test.shape[1])

    for i, sample_idx in enumerate(plot_indices):
        ax = axes[i]
        
        # Plot input signals (X_test)
        for feature_idx in range(X_test.shape[2]):
            ax.plot(time_axis, X_test[sample_idx, :, feature_idx], label=feature_names[feature_idx])
            
        ax.legend(loc='upper right')
        ax.set_ylabel('Input Signals (Z-scored)')
        
        # Plot target EDA signal (Y_test) on a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(time_axis, Y_test[sample_idx, :], 'k--', linewidth=2, label='Target EDA')
        ax2.legend(loc='lower right')
        ax2.set_ylabel('Target EDA (Z-scored)')
        
        # Set title with the sample's label
        label_int = L_test[sample_idx]
        label_str = LABEL_DICT.get(label_int, 'Unknown')
        ax.set_title(f'Test Sample #{sample_idx} - Label: {label_str} ({label_int})')

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    plt.savefig(f'fold_{fold_number}.png')


def main():
    """Main function to parse arguments and trigger plotting."""
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed WESAD data from a specific fold."
    )
    parser.add_argument(
        'fold_number', type=int,
        help="The fold number to load and visualize (e.g., 2 for 'fold_2.npz')."
    )
    args = parser.parse_args()

    plot_fold_data(args.fold_number)


if __name__ == '__main__':
    main() 