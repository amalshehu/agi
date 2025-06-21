import numpy as np
import matplotlib.pyplot as plt


def visualize_example(input_grid: np.ndarray, target_grid: np.ndarray, predicted_grid: np.ndarray) -> None:
    """Display input, target and predicted grids side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, grid, title in zip(axes, [input_grid, target_grid, predicted_grid],
                               ["Input", "Target", "Prediction"]):
        ax.imshow(grid, interpolation="nearest", cmap="tab20")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
