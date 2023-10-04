from collections import defaultdict
import torch

_instances = {}
_instanceNum = defaultdict(lambda: 0)


def Singleton(uid: str, instance: torch.nn.Module) -> torch.nn.Module:
    """
    A singleton wrapper for torch.nn.Module.
    Args:
        uid: the unique id used to identify the module instance.
        instance: the module instance.

    Returns:
        The singleton module instance.
    """
    if uid not in _instances:
        _instances[uid] = instance
    else:
        _instanceNum[uid] += 1
    return _instances[uid]
