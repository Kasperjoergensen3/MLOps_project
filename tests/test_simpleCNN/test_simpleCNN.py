import pytest
import torch
from hydra import initialize, compose, initialize_config_dir
from src.models.SimpleCNN import SimpleCNN 
import os
import hydra
from tests import _NUM_CLASSES, _NUM_CHANNELS, _IMG_SIZE

@pytest.fixture(scope="module")
def simple_cnn_model():
    """
    Pytest fixture to initialize and return the SimpleCNN model.
    This fixture also asserts key configurations to ensure the model is initialized correctly.
    """
    print("Current Working Directory:", os.getcwd())
    with initialize(config_path="conf", version_base=None):
        cfg = compose(config_name="simpleCNN_test_config")

        # Asserting the presence of model and trainer configurations
        assert "model" in cfg, "Model configuration not found in cfg"
        assert "trainer" in cfg, "Trainer configuration not found in cfg"

        # Asserting specific configuration values
        assert cfg.model.num_classes == _NUM_CLASSES, f"Expected {_NUM_CLASSES} classes in model config, found {cfg.model.num_classes}"
        assert cfg.model.in_channels == _NUM_CHANNELS, f"Expected {_NUM_CHANNELS} input channel(s) in model config, found {cfg.model.in_channels}"

        return SimpleCNN(cfg)

def test_init(simple_cnn_model):
    """
    Test to ensure that the SimpleCNN model is initialized properly.
    """
    assert simple_cnn_model is not None, "Model initialization failed, resulting in None"

@pytest.fixture
def sample_batch():
    """
    Pytest fixture to create and return a sample batch of data and labels.
    """
    input_shape = (1, 1, _IMG_SIZE, _IMG_SIZE) 
    num_classes = _NUM_CHANNELS
    data = torch.rand(input_shape)
    labels = torch.randint(0, num_classes, (input_shape[0],))
    return data, labels

def test_training_step(simple_cnn_model, sample_batch):
    """
    Test the training step of the SimpleCNN model to ensure it computes loss correctly.
    """
    data, labels = sample_batch
    loss = simple_cnn_model.training_step((data, labels), batch_idx=0)
    assert loss is not None, "training_step did not compute loss, loss object is None"

def test_validation_step(simple_cnn_model, sample_batch):
    """
    Test the validation step of the SimpleCNN model to ensure it calculates loss and accuracy correctly.
    """
    data, labels = sample_batch
    result = simple_cnn_model.validation_step((data, labels), batch_idx=0)
    assert 'val_loss' in result, "validation_step did not return 'val_loss'"
    assert 'val_accuracy' in result, "validation_step did not return 'val_accuracy'"
    assert 0 <= result['val_accuracy'] <= 1, f"Invalid accuracy value: {result['val_accuracy']}"

def test_forward_pass(simple_cnn_model, sample_batch):
    """
    Test the forward pass of the SimpleCNN model.
    This test ensures that the model can process input data and produce output of the correct shape.
    """
    data, _ = sample_batch  # Labels are not needed for forward pass
    output = simple_cnn_model.forward(data)
    
    # Check if output is not None
    assert output is not None, "Forward pass returned None"

    # Check if the output shape is correct (batch_size, num_classes)
    expected_shape = (data.shape[0], _NUM_CLASSES)
    assert output.shape == expected_shape, f"Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}"
