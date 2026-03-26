import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="The name of .csv file")
args = parser.parse_args()
df = pd.read_csv(f"../logs/{args.file}.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for layer_idx in df["layer"].unique():
    ld = df[df["layer"] == layer_idx]
    label = f"Layer {layer_idx}"

    axes[0, 0].plot(ld["image"], ld["avg_weight"], label=label)
    axes[0, 1].plot(ld["image"], ld["avg_threshold"], label=label)
    axes[0, 2].plot(ld["image"], ld["avg_firing_rate"], label=label)
    axes[1, 0].plot(ld["image"], ld["max_weight"], label=label)
    axes[1, 1].plot(ld["image"], ld["active_neurons"], label=label)
    axes[1, 2].plot(ld["image"], ld["winners"], label=label)

axes[0, 0].set_title("Avg weight over training")
axes[0, 1].set_title("Avg threshold over training")
axes[0, 2].set_title("Avg firing rate over training")
axes[1, 0].set_title("Max weight over training")
axes[1, 1].set_title("Active neurons over training")
axes[1, 2].set_title("Winners")

for ax in axes.flat:
    ax.legend()
    ax.set_xlabel("Images seen")

plt.tight_layout()
plt.savefig(f"imgs/charts_{args.file}.png")
plt.show()


weights = np.loadtxt("layer0_weights.csv", delimiter=",")
# weights shape: (500, 784)

n_show = 100  # show first 100 neurons
fig, axes = plt.subplots(10, 10, figsize=(12, 12))

for idx, ax in enumerate(axes.flat):
    if idx < n_show:
        rf = weights[idx].reshape(28, 28)
        ax.imshow(rf, cmap="RdBu_r", vmin=0, vmax=weights.max())
    ax.axis("off")

plt.suptitle("Layer 0 receptive fields after training")
plt.tight_layout()
plt.savefig("receptive_fields.png")
plt.show()
