import sys
sys.path.append('./src')
import os
import random
import csv
from typing import List, Dict, Union, Any, Optional

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import EvalLoopOutput

from recwizard.modules.redial import AutoRec, ReconstructionLoss
from recwizard.utility import init_deterministic, DeviceManager
from data_processor import RedialDataProcessor
from recwizard.modules.redial.params import autorec_params



def load_movies_merged(path="dataset/redial/movies_merged.csv"):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        id2index = {int(row[3]): int(row[0]) for row in reader if row[0] != "index"}
    return id2index


def process_rating(rating, binary_rating):
    if binary_rating:
        # return 1 for ratings >= 2.5, 0 for lower ratings (this gives 87% of liked on movielens-latest)
        # return 1 for ratings >= 2, 0 for lower ratings (this gives 94% of liked on movielens-latest)
        return float(float(rating) >= 2)
    # return a rating between 0 and 1
    return (float(rating) - .5) / 4.5
class DataCollatorForMovielens:
    """
    Data collator that pads the input data to the maximum length of the
    samples in a batch.
    """

    def __init__(self, train_ratings, random_noise=True, binary_rating=True, max_num_inputs=1e10):
        self.random_noise = random_noise
        self.binary_rating = binary_rating
        self.max_num_inputs = max_num_inputs
        self.id2index = load_movies_merged()
        self.n_movies = max(self.id2index.values()) + 1
        self.train_ratings = train_ratings


    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> (Dict[str, Union[Tensor, Any]], Tensor):
        """
        Pad the input sequences to the maximum length of the samples in a batch.
        """
        input, target = torch.zeros((len(batch), self.n_movies)), -torch.ones((len(batch), self.n_movies))
        for i, example in enumerate(batch):
            userId, movieIds, ratings = example["userId"], example["movieIds"], example["ratings"]
            train_movieIds, train_ratings = self.train_ratings[userId]
            if self.random_noise:
                # randomly chose a number of inputs to keep
                max_nb_inputs = min(self.max_num_inputs, len(train_ratings) - 1)
                n_inputs = random.randint(1, max(1, max_nb_inputs))
                # randomly chose the movies that will be in the input
                input_keys = random.sample(range(len(train_ratings)), n_inputs)
            else:
                input_keys = range(len(train_ratings))

            for key in input_keys:
                j = self.id2index[train_movieIds[key]]
                rating = train_ratings[key]
                input[i, j] = process_rating(rating, binary_rating=self.binary_rating)
            for movieId, rating in zip(movieIds, ratings):
                target[i, self.id2index[movieId]] = process_rating(rating,
                                                                                     binary_rating=self.binary_rating)

        return {
            "input": input,
            "target": target
        }

class DataCollatorForRedial:
    def __init__(self, subset, db2id, random_noise=True, max_num_inputs=1e10):
        self.subset = subset
        self.random_noise = random_noise
        self.max_num_inputs = max_num_inputs
        self.db2id = db2id
        self.n_movies = len(self.db2id)

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> (Dict[str, Union[Tensor, Any]], Tensor):
        if subset == "train" or self.random_noise:
            input = torch.zeros((len(batch), self.n_movies))
            target = -torch.ones((len(batch), self.n_movies))
            for i, example in enumerate(batch):
                movieIds, ratings = example["movieIds"], example["ratings"]
                movieIds = list(map(self.db2id.__getitem__, movieIds))
                indices = range(len(ratings))
                if self.random_noise:
                    # randomly chose a number of inputs to keep
                    max_nb_inputs = min(self.max_num_inputs, len(ratings) - 1)
                    n_inputs = random.randint(1, max(1, max_nb_inputs))
                    # randomly chose the movies that will be in the input
                    indices = random.sample(indices, n_inputs)
                # Create input
                for j in indices:
                    input[i, movieIds[j]] = ratings[j]
                # Create target
                for j in range(len(ratings)):
                    target[i, movieIds[j]] = ratings[j]
        else:
            input, target = [], []
            for example in batch:
                movieIds, ratings = example["movieIds"], example["ratings"]
                movieIds = list(map(self.db2id.__getitem__, movieIds))
                complete_input = [0] * self.n_movies
                for movieId, rating in zip(movieIds, ratings):
                    complete_input[movieId] = rating
                for movieId, rating in zip(movieIds, ratings):
                    # for each movie, zero out in the input and put target rating
                    input_tmp = complete_input[:]
                    input_tmp[movieId] = 0
                    target_tmp = [-1] * self.n_movies
                    target_tmp[movieId] = rating
                    input.append(input_tmp)
                    target.append(target_tmp)
            input = torch.tensor(input)
            target = torch.tensor(target)
        return {
            "input": input,
            "target": target
        }


class RedialTrainer(Trainer):

    def __init__(self, db2id, *pargs, **kwargs):
        super().__init__(*pargs, **kwargs)
        self.db2id = db2id
        self.criterion = ReconstructionLoss()

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=DataCollatorForRedial(subset="eval", random_noise=False, db2id=self.db2id),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        eval_sampler = self._get_eval_sampler(self.eval_dataset)

        return DataLoader(
            self.eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=DataCollatorForRedial(subset="eval", random_noise=False, db2id=self.db2id),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=DataCollatorForRedial(subset="train", random_noise=True, db2id=self.db2id),
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.model.eval()
        losses = []
        for batch in tqdm(dataloader):
            batch = DeviceManager.copy_to_device(batch)
            input, target = batch["input"], batch["target"]
            # compute output and loss
            with torch.no_grad():
                outputs = self.model(input)
                loss = self.criterion(outputs, target)
                losses.append(loss.item())

        num_samples = self.criterion.nb_observed_targets
        metrics = {
            f"{metric_key_prefix}_loss": np.sqrt(self.criterion.normalize_loss_reset(np.mean(losses)))
        }
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=num_samples)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--random_noise', '-r', action='store_true', help='Enable random_noise loading')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loading')
    parser.add_argument('--max_num_inputs', type=int, default=1e10, help='ax number of inputs (for random_noise loading mode)')
    parser.add_argument('--cache_dir', type=str, default=".cache")
    parser.add_argument('--data', choices=['movielens', 'db', 'db_pretrain'], default="db_pretrain", help='Choose different data for training')
    parser.add_argument('--output_dir', type=str, default="save/redial/autorec")
    parser.add_argument('--mloutput_dir', type=str, default="save/redial/autorec")
    parser.add_argument('--checkpoint', type=str, help='select checkpoint to load')

    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = 0 if args.debug else args.num_workers
    num_epochs = 1 if args.debug else args.num_epochs
    output_dir = "tmp" if args.debug else args.output_dir
    init_deterministic(args.seed)


    criterion = ReconstructionLoss()


    def compute_loss(model, inputs, return_outputs=False):
        input, target = inputs["input"], inputs["target"]
        outputs = model(input)
        loss = criterion(outputs, target)
        return (loss, outputs) if return_outputs else loss


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        loss = criterion(torch.tensor(logits), labels)
        return {"loss": loss.item()}



    subsets = ["train", "validation", "test"]

    if args.data in ['movielens', 'db_pretrain']:
        print("Train autorec on movielens.")

        # pre-train on movielens
        datasets = load_dataset("dataset/movielens", name="autorec")
        if args.debug:
            for subset in datasets:
                datasets[subset] = datasets[subset].select(range(1000))
        print("Collecting train ratings")
        train_ratings = {ex['userId']: (ex['movieIds'], ex['ratings']) for ex in tqdm(datasets["train"])}
        print("Selecting data from userId in training set")
        for subset in subsets[1:]:
            datasets[subset] = datasets[subset].filter(lambda x: x["userId"] in train_ratings)
        for subset in subsets:
            print(f"{subset}: {len(datasets[subset])} rows")

        data_collator = DataCollatorForMovielens(train_ratings)

        model = AutoRec(n_movies=data_collator.n_movies, **autorec_params)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   num_train_epochs=num_epochs,
                                   dataloader_num_workers=num_workers,
                                   learning_rate=args.lr,
                                   evaluation_strategy="epoch",
                                   save_strategy="epoch",
                                   remove_unused_columns=False,
                                   per_device_train_batch_size=args.batch_size,
                                   per_device_eval_batch_size=args.batch_size,
                                   load_best_model_at_end=True,
                                   label_names=["target"],
                                   lr_scheduler_type="constant",
                                   report_to='none'
                                   ),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
        )

        trainer.compute_loss = compute_loss
        trainer.train()
        model.save_pretrained(os.path.join(output_dir, 'pretrained_model_best'))
        print(trainer.evaluate(datasets["test"], metric_key_prefix="test"))

    if args.data in ['db', 'db_pretrain']:
        print("Train autorc on ReDIAL")
        datasets = load_dataset("dataset/redial", name="autorec")
        for subset in subsets:
            print(f"{subset}: {len(datasets[subset])} rows")
        dp = RedialDataProcessor()

        model = AutoRec(n_movies=dp.n_redial_movies,**autorec_params)

        model.load_checkpoint(os.path.join(args.mloutput_dir, "pretrained_model_best", "pytorch_model.bin"))

        trainer = RedialTrainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir,
                                   num_train_epochs=num_epochs,
                                   dataloader_num_workers=num_workers,
                                   learning_rate=args.lr / 10,
                                   # we find the finetuning learning rate should be 1/10 of pretraining
                                   evaluation_strategy="epoch",
                                   save_strategy="epoch",
                                   remove_unused_columns=False,
                                   per_device_train_batch_size=args.batch_size,
                                   per_device_eval_batch_size=args.batch_size,
                                   load_best_model_at_end=True,
                                   label_names=["target"],
                                   lr_scheduler_type="constant",
                                   report_to='none'
                                   ),
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            # compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            db2id=dp.db2id
        )
        trainer.compute_loss = compute_loss
        trainer.train()
        print(trainer.evaluate(datasets["test"], metric_key_prefix="test"))
        model.save_pretrained(os.path.join(output_dir, 'db_model_best'))
