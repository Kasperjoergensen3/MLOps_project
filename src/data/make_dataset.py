from pathlib import Path

from torchvision import transforms
from PIL import Image
import json

import torch

from matplotlib import pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[2] / "data"
IMG_SIZE = (224, 224)


def transform_image(image: Image):
    # Define a transformation to convert the image to tensor
    transform = transforms.ToTensor()

    if image.mode != "L":
        image = image.convert("L")

    # Apply the transformation to the image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    # Resize
    resize = transforms.Resize(IMG_SIZE, antialias=True)
    image_tensor = resize(image_tensor)
    return image_tensor


def build_dataset(mode="train", max_samples_per_class=float("inf")):
    """
    Build a PyTorch dataset from a given directory.

    Args:
    data_dir (str): Path to the data directory.
    transform (torchvision.transforms): Transform to apply to the images.

    Returns:
    torch.utils.data.Dataset: The dataset.
    """
    print(f"Building dataset with max_samples_per_class={max_samples_per_class}")

    raw_data_dir = DATA_PATH / "raw" / mode

    # intialise Tensor for storing images
    images = torch.Tensor().type(torch.float32)
    targets = torch.Tensor().type(torch.long)
    class_dictionary = get_class_dictionary()

    for i, class_name in class_dictionary.items():
        class_dir = raw_data_dir / class_name
        total_class_samples = 0
        for image_path in class_dir.iterdir():
            if total_class_samples >= max_samples_per_class:
                break
            image = Image.open(image_path)
            image = transform_image(image)
            target = torch.Tensor([i]).type(torch.long)
            images = torch.cat([images, image])
            targets = torch.cat([targets, target])
            total_class_samples += 1

    dataset = torch.utils.data.TensorDataset(images, targets)

    return dataset


def get_class_dictionary():
    raw_data_dir = DATA_PATH / "raw" / "train"
    classes = [f for f in sorted(raw_data_dir.iterdir()) if not f.name.startswith(".")]
    class_dictionary = {i: p.name for i, p in enumerate(classes)}
    return class_dictionary


if __name__ == "__main__":
    # build datasets
    train_dataset = build_dataset(mode="train")
    valid_dataset = build_dataset(mode="valid")
    test_dataset = build_dataset(mode="test")

    # save dataset
    torch.save(train_dataset, DATA_PATH / "processed" / "train.pt")
    torch.save(valid_dataset, DATA_PATH / "processed" / "valid.pt")
    torch.save(test_dataset, DATA_PATH / "processed" / "test.pt")

    # save class dictionary as json
    class_dictionary = get_class_dictionary()
    with open(DATA_PATH / "processed" / "class_dictionary.json", "w") as f:
        json.dump(class_dictionary, f)

    # print shape
    # print(train_dataset.tensors[0].shape)
    # print(train_dataset.tensors[1].shape)
    # print(valid_dataset.tensors[0].shape)
    # print(valid_dataset.tensors[1].shape)
    # print(test_dataset.tensors[0].shape)
    # print(test_dataset.tensors[1].shape)

    # plot some images
    # fig, ax = plt.subplots(1, 4)
    # for i in range(4):
    #     ax[i].imshow(train_dataset.tensors[0][i].squeeze(), cmap="gray")
    #     ax[i].set_title(f"Label: {train_dataset.tensors[1][i]}")
    # plt.show()
