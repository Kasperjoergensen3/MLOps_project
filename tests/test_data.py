from pathlib import Path
import random
import pytest
import torch
from tests import _PATH_DATA_RAW
from src.data.make_dataset import (
    load_image,
    build_dataset,
    IMG_DIM,
    get_class_dictionary,
)
import os


@pytest.mark.skipif(not os.path.exists(_PATH_DATA_RAW), reason="Raw data dir not found")
@pytest.mark.parametrize("mode", ["train", "valid", "test"])
def test_load_image_on_random_files(mode):
    """
    Test load_image function on a random image from each class directory.
    """

    # Convert the string path to a Path object
    raw_data_dir = Path(_PATH_DATA_RAW) / mode

    for class_dir in raw_data_dir.iterdir():
        if class_dir.is_dir():
            # List all image files in the directory
            images = list(class_dir.glob("*"))
            if images:
                # Pick a random image
                random_image_path = random.choice(images)

                # Load the image
                image_tensor = load_image(str(random_image_path))

                # Check if the image is loaded correctly
                assert isinstance(image_tensor, torch.Tensor)
                assert (
                    len(image_tensor.shape) == 4
                )  # Make sure batch dimension is added
                assert (
                    image_tensor.shape[1] == 1
                )  # Make sure we only have one channel (grayscale)
            else:
                pytest.fail(f"No images found in {class_dir}")


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA_RAW), reason="Raw data dir not found")
def test_get_class_dictionary():
    expected_dict = {
        "0": "glioma",
        "1": "meningioma",
        "2": "no-tumor",
        "3": "pituitary",
    }
    class_dict = get_class_dictionary()

    # Convert keys to strings for comparison, as JSON keys are always strings
    class_dict = {str(key): value for key, value in class_dict.items()}

    assert class_dict == expected_dict


# Helper function for test_build_dataset
def count_images(raw_data_dir):
    """Utility function to count the number of image files in the directory."""
    count = 0
    for class_dir in raw_data_dir.iterdir():
        if class_dir.is_dir():
            count += len(list(class_dir.glob("*")))
    return count


@pytest.mark.skipif(not os.path.exists(_PATH_DATA_RAW), reason="Raw data dir not found")
@pytest.mark.parametrize("mode", ["train", "valid", "test"])
def test_build_dataset(mode):
    dataset = build_dataset(mode=mode)

    # Check if the returned object is a PyTorch Dataset
    assert isinstance(dataset, torch.utils.data.TensorDataset)

    # Count the number of images in the directory
    raw_data_dir = Path(_PATH_DATA_RAW) / mode
    image_count = count_images(raw_data_dir)

    # Check if the dataset size matches the number of images
    assert len(dataset) == image_count

    # check the shape of images and targets
    if len(dataset) > 0:
        images, targets = dataset.tensors
        assert (
            images.ndim == 4
        )  # Check for 4 dimensions (batch, channel, height, width)
        assert images.shape[1] == 1  # Check for single channel (grayscale)
        assert targets.ndim == 1  # Check for 1D target tensor

        # Make sure every image has been resized to the correct size
        for img in images:
            assert img.shape[1:] == IMG_DIM
