"""Script with plot method for visualization of relevance scores.
"""
import matplotlib.pyplot as plt
import torch


def plot_relevance_scores(x: torch.tensor, r: torch.tensor, name: str, config: dict) -> None:
    """Plots results from layer-wise relevance propagation next to original image.

    Method currently accepts only a batch size of one.

    Args:
        x: Original image.
        r: Relevance scores for original image.
        name: Image name.
        config: Dictionary holding configuration.

    Returns:
        None.

    """
    output_dir = config["output_dir"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    x = x[0].squeeze().permute(1, 2, 0).detach().cpu()
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min)
    axes[0].imshow(x)
    axes[0].set_axis_off()

    r_min = r.min()
    r_max = r.max()
    r = (r - r_min) / (r_max - r_min)
    axes[1].imshow(r, cmap="afmhot")
    axes[1].set_axis_off()

    fig.tight_layout()
    plt.savefig(f"{output_dir}/image_{name}.png")
    plt.close(fig)
