from typing import List, Tuple, Set, Dict
import torch

GLOBAL_DEVICE = torch.device('cuda:0')


class DeviceManager:
    """
    A static class that manages the device used for training and inference.
    """
    device = GLOBAL_DEVICE

    @classmethod
    def initialize(cls, device=None):
        if device is None:
            cls.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            cls.device = device

        import os
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        cls.device_cnt = 1

    @classmethod
    def copy_to_device(cls, val, device=None):
        """
        A helper function that copies a common data structure to a device.

        Args:
            val: The value to be copied.
            device: The device to copy to. If None, use the default device.
        """
        if device is None:
            device = cls.device

        if hasattr(val, "to"):
            return val.to(device)
        elif isinstance(val, List) or isinstance(val, Set) or isinstance(val, Tuple):
            return [cls.copy_to_device(x, device) for x in val]
        elif isinstance(val, Dict):
            return {k: cls.copy_to_device(v, device) for k, v in val.items()}
        else:
            return val
        # else:
        #     print(f"Unsupported type: {type(val)}")


def copy_to_device(val, device=None):
    return DeviceManager.copy_to_device(val, device)
