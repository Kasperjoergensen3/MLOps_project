from pathlib import Path

from torchvision import transforms
from PIL import Image
import json

import torch

from matplotlib import pyplot as plt


DATA_PATH = Path(__file__).resolve().parents[2] / "data"
IMG_DIM = (256, 256)


def load_image(image_path):
    """
    Load an image from a given path and return it as a torch.Tensor.

    Args:
    image_path (str): Path to the image file.

    Returns:
    torch.Tensor: The image as a tensor.
    """
    # Define a transformation to convert the image to tensor
    transform = transforms.ToTensor()

    # Load the image using PIL (Python Imaging Library)
    image = Image.open(image_path)

    if image.mode != "L":
        image = image.convert("L")

    # Apply the transformation to the image
    image_tensor = transform(image)

    # add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def build_dataset(mode="train"):
    """
    Build a PyTorch dataset from a given directory.

    Args:
    data_dir (str): Path to the data directory.
    transform (torchvision.transforms): Transform to apply to the images.

    Returns:
    torch.utils.data.Dataset: The dataset.
    """
    raw_data_dir = DATA_PATH / "raw" / mode

    # intialise Tensor for storing images
    images = torch.Tensor().type(torch.float32)
    targets = torch.Tensor().type(torch.int32)
    class_dictionary = get_class_dictionary()

    for i, class_name in class_dictionary.items():
        print(i, class_name)
        class_dir = raw_data_dir / class_name
        for image_path in class_dir.iterdir():
            image = load_image(image_path)
            resize = transforms.Resize(IMG_DIM, antialias=True)
            image = resize(image)
            target = torch.Tensor([i]).type(torch.int32)
            images = torch.cat([images, image])
            targets = torch.cat([targets, target])

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
