import logging
import os
from collections import defaultdict
from typing import List, Dict, Union, Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_utils import EvalLoopOutput


from data_processor import RedialDataProcessor
from recwizard.pipelines.expansion import ExpansionPipeline, ExpansionConfig

from recwizard.modules.redial import RedialGen, RedialGenConfig, params, RedialRec, RedialRecConfig, RedialGenTokenizer, \
    RedialRecTokenizer
from recwizard.utility import init_deterministic, pad_and_stack, loadJsonFileFromDataset
from recwizard.pipelines.switch_decode import SwitchDecodeConfig, SwitchDecodePipeline

class ReDialDataCollatorForRec:
    """
    Data collator that pads the input data to the maximum length of the
    samples in a batch.
    """

    def __init__(self, pad_token_id, return_raw_text=False):
        self.pad_token_id = pad_token_id
        self.return_raw_text = return_raw_text

    def __call__(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Pad the input sequences to the maximum length of the samples in a batch.
        """

        res =  {
            "input": {
                "movieIds": [ex["movieIds"] for ex in batch],
                "dialogue": pad_and_stack([ex["dialogue"] for ex in batch]),
                "input_ids": pad_and_stack([ex["input_ids"] for ex in batch], pad_value=self.pad_token_id),
                "attention_mask": pad_and_stack([ex["attention_mask"] for ex in batch]),
                "movie_occurrences": [ex["movie_occurrences"] for ex in batch],
                "senders": pad_and_stack([ex["senders"] for ex in batch]),
                "lengths": pad_and_stack([ex["lengths"] for ex in batch]),
                "conversation_lengths": torch.tensor([ex["conversation_lengths"] for ex in batch])
            },
            "target": pad_and_stack([ex["target"] for ex in batch]),
        }
        if self.return_raw_text:
            res["input"]["raw_texts"] = [ex["raw_texts"] for ex in batch]
        return res


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
class RecommenderTrainer(Trainer):
    def __init__(self, criterion, num_vocab, **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion
        self.num_vocab = num_vocab


    def compute_loss(self, model, batch, return_outputs=False):
        input, target = batch["input"], batch["target"]
        outputs = model(**input)
        batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
        # indices of recommender's utterances(< batch * max_conv_len)
        idx = torch.nonzero((input["senders"].view(-1) == -1).data).squeeze()
        # select recommender's utterances for the loss
        outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
        target = target.view(-1, max_seq_length).index_select(0, idx)

        loss = self.criterion(outputs.view(-1, vocab_size), target.view(-1))

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
        losses = []
        recalls = defaultdict(list)
        for batch in tqdm(dataloader):
            input, target = batch["input"], batch["target"]
            input = {k: v.to(device) if type(v) is torch.Tensor
            else [x.to(device) for x in v] for k, v in input.items()}
            target = target.to(device)
            # compute output and loss
            with torch.no_grad():

                outputs = self.model(**input)

                batch_size, max_conv_length, max_seq_length, vocab_size = outputs.data.shape
                # indices of recommender's utterances(< batch * max_conv_len)
                idx = torch.nonzero((input["senders"].view(-1) == -1).data).squeeze()
                # select recommender's utterances for the loss
                outputs = outputs.view(-1, max_seq_length, vocab_size).index_select(0, idx)
                target = target.view(-1, max_seq_length).index_select(0, idx)

                loss = self.criterion(outputs.view(-1, vocab_size), target.view(-1))
                losses.append(loss.item())

            # NOTE: one movie occurrence appears multiple times in batch["target"]
            # should only evaluate the first token
            recommendation_idx = (target.view(-1) >= self.num_vocab) & (
                target
                != torch.cat(
                    [
                        torch.zeros(target.shape[0], 1, dtype=torch.long).to(device) - 1,
                        target[:, :-1],
                    ],
                    dim=1
                )
            ).view(-1)
            if recommendation_idx.sum() == 0:
                continue
            recommendation_target = target.view(-1)[recommendation_idx] - self.num_vocab
            recommendation_outputs = outputs.view(-1, vocab_size)[recommendation_idx][:, self.num_vocab:]
            pred, pred_idx = torch.topk(recommendation_outputs, k=100, dim=1)
            def hit(true, pred_list):
                return int(true in pred_list)
            for b in range(recommendation_target.shape[0]):
                recalls['1'].append(hit(recommendation_target[b].item(), pred_idx[b][:1].tolist()))
                recalls['10'].append(hit(recommendation_target[b].item(), pred_idx[b][:10].tolist()))
                recalls['50'].append(hit(recommendation_target[b].item(), pred_idx[b][:50].tolist()))

        metrics = {
            f"recall@{n}": np.mean(recalls[n]) for n in ['1', '10', '50']
        }
        metrics[f"{metric_key_prefix}_loss"] =  np.mean(losses)
        output = EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(recalls['1']))
        return output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Argument Parser')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--pretrained_model', type=str, default='princeton-nlp/unsup-simcse-roberta-base',
                        help='Pretrained model name')
    parser.add_argument('--output_dir', type=str, default="save/redial/rec")
    parser.add_argument('--autorec_model', type=str, default="save/redial/autorec/db_model_best/pytorch_model.bin")
    parser.add_argument('--sa_model', type=str, default="save/redial/sa/model_best/pytorch_model.bin")

    args = parser.parse_args()
    batch_size = args.batch_size
    num_workers = 0 if args.debug else args.num_workers
    num_epochs = 1 if args.debug else args.num_epochs
    output_dir = "tmp" if args.debug else args.output_dir

    init_deterministic(args.seed)

    # load huggingface datsets and pre-process
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    pad_token_id = tokenizer.pad_token_id
    raw_datasets = load_dataset("dataset/redial", "rec")
    dp = RedialDataProcessor(tokenizer=tokenizer, tokenize_for_rnn=True)
    if args.debug:
        for subset in ["train", "validation", "test"]:
            raw_datasets[subset] = raw_datasets[subset].select(range(10))
    tokenized_datasets = raw_datasets.map(dp.map_redial_for_recommender, load_from_cache_file=False)
    tokenized_datasets.set_format("torch")

    vocab = loadJsonFileFromDataset('dataset/redial', 'vocab.json')
    gen_module = RedialGen(
        RedialGenConfig(
            hrnn_params=params.hrnn_params,
            decoder_params=params.decoder_params,
            vocab_size=len(vocab),
            n_movies=6924
        ),
        word_embedding=dp.get_task_embedding()
    )
    sa_params = params.sentiment_analysis_params
    sa_params.update(
        {
            "return_liked_probability": True,
            "multiple_items_per_example": True,
        }
    )
    rec_module = RedialRec(
        RedialRecConfig(
            sa_params=sa_params,
            autorec_params=params.autorec_params
        ),
        n_movies=6924,
        recommend_new_movies=False
    )
    rec_module.load_state_dict(torch.load(args.autorec_model), strict=True,
                               allow_unexpected=True,
                               LOAD_IGNORES=('sentiment_analysis',),
                               LOAD_MAPPINGS={
                                   '': 'recommender.'
                               })
    rec_module.load_state_dict(torch.load(args.sa_model), strict=True,
                               allow_unexpected=True,
                               LOAD_IGNORES=('recommender',),
                               LOAD_MAPPINGS={
                                   '': 'sentiment_analysis.'
                               })
    gen_tokenizer = RedialGenTokenizer.load_from_dataset('redial')
    rec_tokenizer = RedialRecTokenizer.load_from_dataset('redial')
    model = ExpansionPipeline(config=ExpansionConfig(),
                              gen_module=gen_module,
                              gen_tokenizer=gen_tokenizer,
                              rec_tokenizer=rec_tokenizer,
                              rec_module=rec_module,
                              )

    criterion = torch.nn.NLLLoss()
    trainer = RecommenderTrainer(
        num_vocab=len(dp.vocab),
        model=model,
        optimizers=(torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr), None),
        args=TrainingArguments(output_dir=output_dir,
                               save_strategy="epoch",
                               evaluation_strategy="epoch",
                               num_train_epochs=num_epochs,
                               remove_unused_columns=False,
                               dataloader_num_workers=num_workers,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               load_best_model_at_end=True,
                               # load_best_model_at_end requires the save and eval strategy to match
                               label_names=["target"],
                               lr_scheduler_type="linear",
                               report_to='none'
                               ),
        criterion=criterion,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=ReDialDataCollatorForRec(pad_token_id=tokenizer.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
    print(trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test"))
    trainer.save_model(os.path.join(output_dir, 'model_best'))
