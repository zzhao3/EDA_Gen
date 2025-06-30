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
    from the test set, creating a separate plot for each label.
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

    # Find temp index if it exists, to be used for all plots in this fold
    temp_idx = -1
    try:
        temp_idx = np.where(feature_names == 'temp')[0][0]
    except IndexError:
        print("Info: 'temp' signal not found in features. It will not be plotted on a separate axis.")

    # --- Print the signal names ---
    print("\nPlotting the following signals:")
    for name in feature_names:
        print(f"  - {name}")
    print("  - Target EDA (on separate axis)")
    # --------------------------------

    if X_test.size == 0:
        print("No test data available in this fold to plot.")
        return

    # Create base output directory for the fold
    output_dir = os.path.join('figures', f'fold_{fold_number}')
    os.makedirs(output_dir, exist_ok=True)

    time_axis = np.linspace(0, X_test.shape[1] / FS, num=X_test.shape[1])

    # 3. Iterate through each label to create a plot
    for label_int, label_name in LABEL_DICT.items():
        # Find indices for the current label
        indices_for_label = np.where(L_test == label_int)[0]

        if len(indices_for_label) == 0:
            print(f"\nInfo: No test samples for label '{label_name}' ({label_int}). Skipping plot.")
            continue

        # Choose up to 4 samples
        num_to_select = min(NUM_SAMPLES_TO_PLOT, len(indices_for_label))
        if len(indices_for_label) < NUM_SAMPLES_TO_PLOT:
            print(f"Warning: Only {len(indices_for_label)} samples for label '{label_name}'. "
                  f"Plotting all of them.")

        plot_indices = np.random.choice(
            indices_for_label, size=num_to_select, replace=False
        )

        print(f"\nPlotting {len(plot_indices)} samples for label '{label_name}'...")

        # 4. Create the plot for the current label
        fig, axes = plt.subplots(
            len(plot_indices), 1,
            figsize=(12, 3 * len(plot_indices)),
            sharex=True
        )
        if len(plot_indices) == 1:
            axes = [axes]  # Ensure axes is always iterable

        fig.suptitle(f'Test Samples for Label: {label_name}', fontsize=16)

        for i, sample_idx in enumerate(plot_indices):
            ax = axes[i]

            # Plot input signals (X_test), excluding temp
            for feature_idx in range(X_test.shape[2]):
                if feature_idx == temp_idx:
                    continue
                ax.plot(time_axis, X_test[sample_idx, :, feature_idx], label=feature_names[feature_idx])

            ax.legend(loc='upper left')
            ax.set_ylabel('Input Signals (Z-scored)')

            # Plot target EDA signal (Y_test) on a secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(time_axis, Y_test[sample_idx, :], 'k--', linewidth=2, label='Target EDA')
            ax2.legend(loc='upper right')
            ax2.set_ylabel('Target EDA (Z-scored)')

            # Plot temp signal on a third y-axis if found
            if temp_idx != -1:
                ax3 = ax.twinx()
                # Offset the new spine to prevent overlap
                ax3.spines['right'].set_position(('outward', 60))
                
                temp_line, = ax3.plot(time_axis, X_test[sample_idx, :, temp_idx], 'm-', linewidth=2, label='Temp')
                ax3.set_ylabel('Temp (Z-scored)', color=temp_line.get_color())
                ax3.tick_params(axis='y', colors=temp_line.get_color())
                ax3.legend(handles=[temp_line], loc='lower right')

            # Set title for the subplot
            ax.set_title(f'Test Sample #{sample_idx}')

        axes[-1].set_xlabel('Time (s)')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

        # Save the figure
        save_path = os.path.join(output_dir, f'{label_name}_samples.png')
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close(fig)  # Close the figure to free up memory


def main():
    """Main function to parse arguments and trigger plotting."""
    parser = argparse.ArgumentParser(
        description="Visualize preprocessed WESAD data from a specific fold, creating separate plots for each label."
    )
    parser.add_argument(
        'fold_number', type=int,
        help="The fold number to load and visualize (e.g., 2 for 'fold_2.npz')."
    )
    args = parser.parse_args()

    plot_fold_data(args.fold_number)


if __name__ == '__main__':
    main() 