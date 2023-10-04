import sys
sys.path.append("./")
sys.path.append("./src")

from recwizard.tokenizer_utils import BaseTokenizer
import math
from transformers.trainer_utils import EvalLoopOutput

from evaluator import RecEvaluator

from recwizard.modules.unicrs import UnicrsRecConfig, UnicrsRec, UnicrsRecTokenizer
from data_processor import UnicrsDataProcessor, KGDataLoader
from transformers import Trainer, TrainingArguments, GPT2Config, AutoConfig

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

from recwizard.utility import DeviceManager, batchify, pad_and_stack, init_deterministic


class CRSDataCollatorForRec:
    def __init__(self, tokenizer: BaseTokenizer, debug=False):
        self.debug = debug
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch = batchify(batch)
        encodings = self.tokenizer(batch['messages'], add_special_tokens=True)

        return {
            **encodings,
            'rec_labels': torch.as_tensor(batch["rec"]),
        }


class PromptTrainer(Trainer):
    def __init__(self, evaulator, **kwargs):
        super().__init__(**kwargs)
        self.evaluator = evaulator

    def compute_loss(self, model, batch, return_outputs=False):
        batch = DeviceManager.copy_to_device(batch)
        outputs = model(**batch)
        loss = outputs.rec_loss

        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only=None,
            ignore_keys=None,
            metric_key_prefix="eval",
    ):
        losses = []
        for batch in dataloader:
            with torch.no_grad():
                loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
                losses.append(float(loss))
                self.evaluator.evaluate(outputs.rec_logits, labels=batch['rec_labels'])

        metrics = self.evaluator.report(reset_metric=True)
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=metrics['count'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save/unicrs_pre/redial', help="Where to store the final model.")
    parser.add_argument("--debug", "-d", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default="redial_unicrs", help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--entity_max_length", type=int, default=32, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    # model
    parser.add_argument("--model", type=str, default="microsoft/DialoGPT-small",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str, default="roberta-base")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
    parser.add_argument("--num_training_steps", type=int, default=-1,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=1389)
    parser.add_argument("--fp16", action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    num_workers = 0 if args.debug else args.num_workers
    num_epochs = 2 if args.debug else args.num_train_epochs
    output_dir = "tmp" if args.debug else args.output_dir

    init_deterministic()
    dp = UnicrsDataProcessor(args.dataset)
    model_config = GPT2Config.from_pretrained(args.model)
    text_encoder_config = AutoConfig.from_pretrained(args.text_encoder)

    tokenizer = UnicrsRecTokenizer.load_from_dataset(args.dataset)

    datasets = load_dataset(os.path.join("dataset", args.dataset), 'unrolled')
    if args.debug:
        for subset in ["train", "validation", "test"]:
            datasets[subset] = datasets[subset].select(range(10))

    datasets = datasets.map(dp.prepare_data_for_pretrain, batched=True, load_from_cache_file=False, remove_columns=["recNames"])

    kg_info = KGDataLoader.get_entity_kg_info(args.dataset)
    edge_index = kg_info.pop('edge_index')
    edge_type = kg_info.pop('edge_type')
    item_ids = kg_info.pop('item_ids')
    kgprompt_config = {
        'hidden_size': model_config.n_embd,
        'token_hidden_size': text_encoder_config.hidden_size,
        'n_head': model_config.n_head,
        'n_layer': model_config.n_layer,
        'n_block': 2,
        # "dataset": args.dataset,
        'num_bases': args.num_bases,
        'kg_info': kg_info,
        # newly added
        "n_prefix_rec": 10,
        "num_tokens": len(tokenizer.tokenizers[1])
    }


    model = UnicrsRec(
        config=UnicrsRecConfig(
            pretrained_model=args.model,
            kgprompt_config=kgprompt_config,
            num_tokens = len(tokenizer.tokenizers[0]),
            pad_token_id=tokenizer.tokenizers[0].pad_token_id
        ),
        edge_index=edge_index,
        edge_type=edge_type,
        use_rec_prefix=False
    )

    model = model.to(DeviceManager.device)

    # optim
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]


    num_update_steps_per_epoch = math.ceil(len(datasets["train"]) /
                                           args.per_device_train_batch_size /
                                           args.gradient_accumulation_steps
                                           )

    if args.num_training_steps == -1:
        num_training_steps = num_epochs * num_update_steps_per_epoch
    else:
        num_training_steps = args.num_training_steps


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, num_training_steps)

    trainer = PromptTrainer(
        model=model,
        optimizers=(optimizer, lr_scheduler),
        args=TrainingArguments(output_dir=output_dir,
                               save_strategy="epoch",
                               evaluation_strategy="epoch",
                               num_train_epochs=num_epochs,
                               max_steps=num_training_steps,
                               remove_unused_columns=False,
                               dataloader_num_workers=num_workers,
                               per_device_train_batch_size=args.per_device_train_batch_size,
                               per_device_eval_batch_size=args.per_device_eval_batch_size,
                               load_best_model_at_end=True,
                               # load_best_model_at_end requires the save and eval strategy to match
                               label_names=["target"],
                               lr_scheduler_type="linear",
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               max_grad_norm=args.max_grad_norm,
                               metric_for_best_model="eval_loss",
                               report_to="none",  # disable wandb,
                               ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CRSDataCollatorForRec(tokenizer),
        evaulator=RecEvaluator(),
    )
    trainer.train()
    print(trainer.evaluate(datasets["test"], metric_key_prefix="test"))
    model.save_pretrained(os.path.join(output_dir, 'model_best'))
