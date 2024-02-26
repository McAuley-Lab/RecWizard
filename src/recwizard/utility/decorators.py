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


def WrapSingleInput(func):
    """
    Decorator for functions that takes either a single input or a batch of inputs, and returns the same.
    It wraps a single input in a batch before feeding it to the `func` and unwraps the result from the batched output.
    """

    def wrapper(self, raw_input, *args, **kwargs):
        isSingle = False
        if not isinstance(raw_input, List):
            raw_input = [raw_input]
            isSingle = True
        res = func(self, raw_input, *args, **kwargs)
        if isSingle:
            if isinstance(res, dict):
                for key in res:
                    res[key] = res[key][0]
            else:
                res = res[0]
        return res

    return wrapper
