import re
import sys
from typing import List

from datasets import load_dataset, Value, Sequence
import json

def print_dataset_info(dataset):
    # Print the features
    print("features:")
    for feature_name, feature in dataset.features.items():
        if True:
            print(f"  - name: {feature_name}")
        if isinstance(feature, Value):
            print(f"    dtype: {feature.dtype}")
        elif isinstance(feature, Sequence):
            print(f"    sequence: {feature.feature.dtype}")

    # Print the splits information
    print("splits:")
    for split_name, split_info in dataset.info.splits.items():
        print(f"  - name: {split_name}")
        print(f"    num_bytes: {split_info.num_bytes}")
        print(f"    num_examples: {split_info.num_examples}")

    # Print the download size and dataset size
    print(f"download_size: {dataset.info.download_size}")
    print(f"dataset_size: {dataset.info.dataset_size}")


if __name__ == "__main__":
    dataset = load_dataset(sys.argv[1], sys.argv[2])
    print_dataset_info(dataset["train"])

    print("An example of 'test' looks as follows.")
    print("```")
    print(json.dumps(dataset["test"][0], indent=2, sort_keys=True))
    print("```")
