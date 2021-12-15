"""Script for qualitative evaluation of LRP heatmaps.
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True

output_dir = r"../../output"
fontsize = 30
dpi = 90


def load_next_results(index):
    x = np.load(f"{output_dir}/{index}_original.npy")
    r1 = np.load(f"{output_dir}/{index}_lrp.npy")
    r2 = np.load(f"{output_dir}/{index}_lrp_filter.npy")
    return x, r1, r2


def plot_results(x, r1, r2, index):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), tight_layout=True)

    axes[0].imshow(x)
    axes[0].set_axis_off()

    axes[1].imshow(r1, cmap="afmhot")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("$z^+$-rule", fontsize=fontsize)

    axes[2].imshow(r2, cmap="afmhot")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_xlabel("$z^+$-rule + relevance filter", fontsize=fontsize)

    plt.savefig(f"{output_dir}/result_{index}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    for index in range(69):
        x, r1, r2 = load_next_results(index=index)
        plot_results(x, r1, r2, index=index)


if __name__ == "__main__":
    main()
