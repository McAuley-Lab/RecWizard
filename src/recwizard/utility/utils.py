import copy
import inspect
import json
import os
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import random

from transformers import PreTrainedModel


# os.environ["TOKENIZERS_PARALLELISM"] = "false" # Not sure what's the effect of this

def sort_for_packed_sequence(lengths):
    """
    Sorts an array of lengths in descending order and returns the sorted lengths,
    the indices to sort, and the indices to retrieve the original order.

    Args:
        lengths: 1D array of lengths

    Returns:
        sorted_lengths: lengths in descending order
        sorted_idx: indices to sort
        rev: indices to retrieve original order
    """
    sorted_idx = torch.argsort(lengths, descending=True)  # idx to sort by length
    sorted_lengths = lengths[sorted_idx]
    rev = torch.argsort(sorted_idx)  # idx to retrieve original order

    return sorted_lengths, sorted_idx, rev


def init_deterministic(seed=42):
    """
    Initialize deterministic behavior for reproducibility.
    Args:
        seed: the seed to use for all random number generators.
    """
    # Set the random seed for Python's built-in random module
    import random
    random.seed(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)

    # Set the random seed for PyTorch CUDA operations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic behavior for CuDNN (if available)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def deterministic_seed(seed=42):
    # Store original states
    orig_pytorch_state = torch.random.get_rng_state()
    orig_cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    orig_np_state = np.random.get_state()
    orig_random_state = random.getstate()

    # Set new seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        yield
    finally:
        # Restore original states
        torch.random.set_rng_state(orig_pytorch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(orig_cuda_state)
        np.random.set_state(orig_np_state)
        random.setstate(orig_random_state)


def apply_func(func, *obj, **kwargs):
    """
    Apply func on n number of objects (each object has the same length k),
    and return function results stacked with length k.

    Note: Don't use obj as keyword

    Args:
        func (callable): function to exectute
        *obj: each obj as an single item or a list. should have the same length
        **kwargs: other arguments to pass to func

    Returns:
        (list): list of function results or a single function result
    """
    obj = list(obj)
    if isinstance(obj[0], List):
        for i in range(len(obj)):
            if obj[i] is None:
                obj[i] = [None] * len(obj[0])
        *res, = tuple(*zip(*[(func(*args, **kwargs),) for args in zip(*obj)]))
        if isinstance(res[0], tuple):
            res = list(zip(*res))
        return res
    else:
        return func(*obj, **kwargs)


def pad_and_stack(tensors, dtype=torch.long, pad_value=0):
    """
    Pad tensors of different sizes with zeros and stack them up.

    Args:

        tensors (list of torch.Tensor): List of tensors to be padded and stacked.
            Each tensor should have the same number of dimensions, but different
            sizes in the last dimension.
        dtype (torch.dtype, optional): Data type of the output tensor.
        pad_value (int, optional): The value to fill in the padded areas.

    Returns:
        padded_and_stacked (torch.Tensor): The padded and stacked tensor, with
            shape (num_tensors, max_size_dim1, max_size_dim2, ..., max_size_dimN),
            where N is the number of dimensions of the input tensors.
    """

    # Get the maximum size of each dimension across all tensors
    device = tensors[0].device
    max_sizes = [max(tensor.size(d) for tensor in tensors) for d in range(len(tensors[0].size()))]

    # Create a tensor filled with zeros, with size (num_tensors, max_size_dim1, max_size_dim2, ..., max_size_dimN)
    padded_and_stacked = pad_value * torch.ones((len(tensors), *max_sizes), dtype=dtype, device=device)

    # Pad each tensor with zeros and add it to the padded_and_stacked tensor
    for i, tensor in enumerate(tensors):
        slices = [slice(0, size) for size in tensor.size()]
        padded_and_stacked[(i,) + tuple(slices)] = tensor
    return padded_and_stacked


def batchify(batch):
    """
        convert a list of dict to a dict of list
    """
    assert isinstance(batch, list) and len(batch) > 0
    return {key: [ex[key] for ex in batch] for key in batch[0].keys()}


def unbatch(batch):
    """
        convert a dict of list to a list of dict
    """
    return [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]


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


def loadJsonFileFromDataset(dataset: str, file: str, dataset_org=None):
    """
    Load a json file from a huggingface dataset
    Args:
        dataset: the name of the dataset (e.g. "redial")
        file: the name of the file (e.g. "entity2id.json")
        dataset_org: the organization of the dataset. Default to `HF_ORG` from constants.py.

    Returns:
        (dict): the loaded json file
    """
    assert isinstance(file, str) and file.endswith('json')
    if dataset_org is None:
        from .constants import HF_ORG
        dataset_org = HF_ORG

    from huggingface_hub import hf_hub_download
    downloaded_file = hf_hub_download(f"{dataset_org}/{dataset}", file, repo_type="dataset")
    json_file = open(downloaded_file, 'r')
    return json.load(json_file)
