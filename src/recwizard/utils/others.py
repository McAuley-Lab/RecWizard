import json
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
import random


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


def load_json_file_from_dataset(dataset: str, file: str, dataset_org=None):
    """
    Load a json file from a huggingface dataset
    Args:
        dataset: the name of the dataset (e.g. "redial")
        file: the name of the file (e.g. "entity2id.json")
        dataset_org: the organization of the dataset. Default to `HF_ORG` from constants.py.

    Returns:
        (dict): the loaded json file
    """
    assert isinstance(file, str) and file.endswith("json")
    if dataset_org is None:
        from . import HF_ORG

        dataset_org = HF_ORG

    from huggingface_hub import hf_hub_download

    downloaded_file = hf_hub_download(f"{dataset_org}/{dataset}", file, repo_type="dataset")
    with open(downloaded_file, "r") as json_file:
        return json.load(json_file)
