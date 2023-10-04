import os
from typing import List, Dict, Union, Any

import numpy as np
import sklearn
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

from recwizard.modules.redial.hrnn_for_classification import HRNNForClassification, RedialSentimentAnalysisLoss
from recwizard.utility import init_deterministic, pad_and_stack, DeviceManager
from data_processor import RedialDataProcessor
from recwizard.modules.redial.params import sentiment_analysis_params


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class ReDialDataCollatorForSA:
    """
    Data collator that pads the input data to the maximum length of the
    samples in a batch.
    """

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[Tensor, Any]]:
        """
        Pad the input sequences to the maximum length of the samples in a batch.
        """
        res =  {
            "input_ids": pad_and_stack([ex["input_ids"] for ex in batch], pad_value=self.pad_token_id),
            "attention_mask": pad_and_stack([ex["attention_mask"] for ex in batch]),
            "movie_occurrences": pad_and_stack([ex["movie_occurrences"] for ex in batch]),
            "senders": pad_and_stack([ex["senders"] for ex in batch]),
            "labels": torch.stack([ex["labels"] for ex in batch]),
            # "lengths": pad_and_stack([ex["lengths"] for ex in batch]),
            "conversation_lengths": torch.tensor([ex["conversation_lengths"] for ex in batch])
        }
        return res



class RedialSATrainer(Trainer):
    def __init__(self, criterion,**kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = DeviceManager.copy_to_device(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.criterion(outputs, labels)
        return (loss, outputs) if return_outputs else loss


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):

        self.model.eval()

        total = 0
        correct = 0
        losses = []
        matrix_size = 18
        Iconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        Rconfusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
        Iconfusion_matrix_no_disagreement = np.zeros((matrix_size, matrix_size), dtype=int)
        Rconfusion_matrix_no_disagreement = np.zeros((matrix_size, matrix_size), dtype=int)
        wrongs = 0
        wrongs_with_disagreement = 0
        for batch in tqdm(dataloader):
            # compute output and loss
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop("labels")
            with torch.no_grad():
                output = self.model(**batch)
                loss = self.criterion(output, target)
            output = list(output.values())
            losses.append(loss.item())
            target = target.cpu().data.numpy()
            Isugg = (output[0] > 0.5).squeeze(1).cpu().long()
            # get the arg max for the categorical output
            Iseen = torch.max(output[1], 1)[1].squeeze().cpu()
            Iliked = torch.max(output[2], 1)[1].squeeze().cpu()
            Rsugg = (output[3] > 0.5).squeeze(1).cpu().long()
            Rseen = torch.max(output[4], 1)[1].squeeze().cpu()
            Rliked = torch.max(output[5], 1)[1].squeeze().cpu()

            # increment number of wrong predictions (either seeker and recommender)
            wrongs += np.sum(1 * (Iliked.data.numpy() != target[:, 2]) + 1 * (Rliked.data.numpy() != target[:, 5]))
            # increment number of wrong predictions where the targets disagree (ambiguous dialogue or careless worker)
            wrongs_with_disagreement += np.sum(
                1 * (Iliked.data.numpy() != target[:, 2]) * (target[:, 2] != target[:, 5])
                + 1 * (Rliked.data.numpy() != target[:, 5]) * (target[:, 2] != target[:, 5]))
            total += target.shape[0]

            if metric_key_prefix == "test":
                # Cartesian product of all three different targets
                Iclass = Isugg + 2 * Iseen + 6 * Iliked
                Rclass = Rsugg + 2 * Rseen + 6 * Rliked
                Itargetclass = target[:, 0] + 2 * target[:, 1] + 6 * target[:, 2]
                Rtargetclass = target[:, 3] + 2 * target[:, 4] + 6 * target[:, 5]
                # marks examples where the targets agree
                filter_no_disagreement = target[:, 2] == target[:, 5]

                correct += (Iclass.data == torch.LongTensor(Itargetclass)).cpu().sum()
                # increment confusion matrices
                Iconfusion_matrix += sklearn.metrics.confusion_matrix(Itargetclass, Iclass.data.numpy(),
                                                                      labels=np.arange(18))
                Rconfusion_matrix += sklearn.metrics.confusion_matrix(Rtargetclass, Rclass.data.numpy(),
                                                                      labels=np.arange(18))
                # increment confusion matrices only taking the examples where the two workers agree
                Iconfusion_matrix_no_disagreement += sklearn.metrics.confusion_matrix(
                    Itargetclass[filter_no_disagreement],
                    Iclass.data.numpy()[filter_no_disagreement],
                    labels=np.arange(18))
                Rconfusion_matrix_no_disagreement += sklearn.metrics.confusion_matrix(
                    Rtargetclass[filter_no_disagreement],
                    Rclass.data.numpy()[filter_no_disagreement],
                    labels=np.arange(18))
        if metric_key_prefix == "test":
            # the reshape modelizes a block matrix.
            # then we sum the blocks to obtain marginal matrices.
            # confusion matrix for suggested/mentioned label
            Isugg_marginal = Iconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(0, 2))
            Rsugg_marginal = Rconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(0, 2))
            # confusion matrix that ignores the suggested/mentioned label
            I_marginal = Iconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(1, 3))
            R_marginal = Rconfusion_matrix.reshape(9, 2, 9, 2).sum(axis=(1, 3))
            # confusion matrix for the seen/not seen/did not say label
            Iseen_marginal = I_marginal.reshape(3, 3, 3, 3).sum(axis=(0, 2))
            Rseen_marginal = R_marginal.reshape(3, 3, 3, 3).sum(axis=(0, 2))
            # confusion matrix for the liked/disliked/did not say label
            Iliked_marginal = I_marginal.reshape(3, 3, 3, 3).sum(axis=(1, 3))
            Rliked_marginal = R_marginal.reshape(3, 3, 3, 3).sum(axis=(1, 3))
            Iliked_marginal_no_disagreement = Iconfusion_matrix_no_disagreement.reshape(3, 3, 2, 3, 3, 2) \
                .sum(axis=(1, 2, 4, 5))
            Rliked_marginal_no_disagreement = Rconfusion_matrix_no_disagreement.reshape(3, 3, 2, 3, 3, 2) \
                .sum(axis=(1, 2, 4, 5))
            print("marginals")
            print(I_marginal)
            print(R_marginal)
            print("Suggested marginals")
            print(Isugg_marginal)
            print(Rsugg_marginal)
            print("Seen marginals")
            print(Iseen_marginal)
            print(Rseen_marginal)
            print("Liked marginals")
            print(Iliked_marginal)
            print(Rliked_marginal)
            print("Liked marginals, excluding targets with disagreements")
            print(Iliked_marginal_no_disagreement)
            print(Rliked_marginal_no_disagreement)
        print("{} wrong answers for {} liked labels, for {} of those there was a disagreement between workers"
              .format(wrongs, total*2, wrongs_with_disagreement))
        avg_loss = float(np.mean(losses))
        print("{} loss : {}".format(metric_key_prefix, avg_loss ))
        metrics = {
            f'{metric_key_prefix}_loss': avg_loss,
        }
        output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=total)
        return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--debug', '-d',  action='store_true', help='Enable debug mode')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--pretrained_model', type=str, default='princeton-nlp/unsup-simcse-roberta-base',
                        help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default="save/redial/sa")

    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = 0 if args.debug else args.num_workers
    num_epochs = 1 if args.debug else args.num_epochs
    output_dir = "tmp" if args.debug else args.output_dir
    init_deterministic(args.seed)

    # load huggingface datsets and pre-process
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    pad_token_id = tokenizer.pad_token_id
    raw_datasets = load_dataset("dataset/redial", "SA")
    if args.debug:
        for subset in ["train", "validation", "test"]:
            raw_datasets[subset] = raw_datasets[subset].select(range(10))
    preprocessor = RedialDataProcessor(tokenizer=tokenizer)
    tokenized_datasets = raw_datasets.map(preprocessor.map_redial_for_sentiment_analysis)
    tokenized_datasets.set_format("torch")

    model = HRNNForClassification(**sentiment_analysis_params).to(device)
    trainer = RedialSATrainer(
        model=model,
        optimizers=(torch.optim.Adam(model.parameters(), lr=args.lr), None),
        args=TrainingArguments(output_dir=output_dir,
                               save_strategy="epoch",
                               evaluation_strategy="epoch",
                               num_train_epochs=num_epochs,
                               remove_unused_columns=False,
                               dataloader_num_workers=num_workers,
                               per_device_train_batch_size=batch_size,
                               load_best_model_at_end=True,
                               report_to='none'
                               ),
        criterion=RedialSentimentAnalysisLoss(class_weight={"liked": [1. / 5, 1. / 80, 1. / 15]},
                                              use_targets="suggested seen liked").to(device),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=ReDialDataCollatorForSA(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()
    trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    trainer.save_model(os.path.join(output_dir, 'model_best'))
