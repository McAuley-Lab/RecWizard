import sys

from recwizard.modules.redial import RedialRecTokenizer

sys.path.append('../src')

import re
from unittest import TestCase
from datasets import load_dataset, load_dataset_builder, builder
from transformers import AutoTokenizer


class DatasetTests(TestCase):

    def test_redial(self):
        # dataset_builder = ReDIAL(base_path="./redial")
        # dataset_builder.download_and_prepare(base_path="./redial")
        #
        # raw_datasets = dataset_builder.as_dataset()
        datasets = load_dataset("redial", 'formatted', download_mode="force_redownload")
        for i in range(10):
            print(datasets['test'][i])


    def test_redial_unicrs(self):
        dataset = load_dataset("redial_unicrs", "compact")
        print(dataset["test"][0])

    def test_inspired_unicrs(self):
        dataset = load_dataset("inspired_unicrs")
        for i in range(10):
            print(dataset["test"][i])

    def test_redial_unrolled(self):
        dataset = load_dataset("redial_unicrs", "unrolled")
        for i in range(10):
            print(dataset["test"][i])
