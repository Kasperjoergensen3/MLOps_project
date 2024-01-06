from pathlib import Path
from torchvision import transforms
from PIL import Image

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


def build_dataset(mode="training"):
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
    for i, class_dir in enumerate(raw_data_dir.iterdir()):
        print(class_dir)
        for image_path in class_dir.iterdir():
            image = load_image(image_path)
            resize = transforms.Resize(IMG_DIM, antialias=True)
            image = resize(image)
            target = torch.Tensor([i])
            images = torch.cat([images, image])
            targets = torch.cat([targets, target])

    dataset = torch.utils.data.TensorDataset(images, targets)

    return dataset


if __name__ == "__main__":
    train_dataset = build_dataset(mode="training")
    test_dataset = build_dataset(mode="testing")
    # print shape
    print(train_dataset.tensors[0].shape)
    print(train_dataset.tensors[1].shape)
    print(test_dataset.tensors[0].shape)
    print(test_dataset.tensors[1].shape)

    # save dataset
    torch.save(train_dataset, DATA_PATH / "processed" / "train.pt")
    torch.save(test_dataset, DATA_PATH / "processed" / "test.pt")
