import matplotlib.pyplot as plt
import torch
import numpy.random as random
import json
from pathlib import Path

random.seed(1234)
root = Path(__file__).resolve().parents[2]
FIG_FOLDER = root.joinpath("reports/figures")


def plot_example_images():
    DATA_PATH = root.joinpath("data/processed/train.pt")
    JSON_PATH = root.joinpath("data/processed/class_dictionary.json")

    dataset = torch.load(DATA_PATH)
    with open(JSON_PATH, "r") as f:
        class_dict = json.load(f)

    N = len(dataset)
    D1, D2 = 3, 7
    idx = random.randint(0, N, D1 * D2)

    fig, axs = plt.subplots(D1, D2, figsize=(8, 4))

    axs = axs.flatten()
    for i, ax in enumerate(axs):
        ax.imshow(dataset[idx[i]][0].numpy().squeeze(), cmap="gray")

        ax.set_title(class_dict[str(dataset[idx[i]][1].item())], color="white")
        ax.axis("off")
    fig.patch.set_facecolor("black")
    plt.tight_layout()
    plt.savefig(FIG_FOLDER / "example_images.png", dpi=300)


if __name__ == "__main__":
    plot_example_images()
