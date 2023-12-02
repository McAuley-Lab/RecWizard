import pytest
import torch

from recwizard.utility import DeviceManager

# Fixtures
@pytest.fixture
def device_manager():
    DeviceManager.initialize()
    return DeviceManager

# Tests
def test_initialize_default_device(device_manager):
    assert device_manager.device == torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

@pytest.mark.skipif(torch.cuda.is_available() is False, reason="No CUDA Available")
def test_initialize_custom_device(device_manager):
    custom_device = torch.device('cuda:1')
    device_manager.initialize(custom_device)
    assert device_manager.device == custom_device

def test_copy_to_device_default(device_manager):
    data = torch.tensor([1, 2, 3])
    copied_data = device_manager.copy_to_device(data)
    assert copied_data.device == device_manager.device

@pytest.mark.skipif(torch.cuda.is_available() is False, reason="No CUDA Available")
def test_copy_to_device_custom(device_manager):
    data = torch.tensor([1, 2, 3])
    custom_device = torch.device('cuda:1')
    copied_data = device_manager.copy_to_device(data, custom_device)
    assert copied_data.device == custom_device

def test_copy_to_device_list(device_manager):
    data_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
    copied_data = device_manager.copy_to_device(data_list)
    for item in copied_data:
        assert item.device == device_manager.device

def test_copy_to_device_dict(device_manager):
    data_dict = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
    copied_data = device_manager.copy_to_device(data_dict)
    for key, value in copied_data.items():
        assert value.device == device_manager.device
