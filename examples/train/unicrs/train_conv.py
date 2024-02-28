import sys

sys.path.append("./")
sys.path.append("./src")
import time
from typing import Optional, List

import math
from tqdm import tqdm
from transformers.trainer_utils import EvalLoopOutput

from evaluator import ConvEvaluator
from data_processor import UnicrsDataProcessor, KGDataLoader
from recwizard.modules.unicrs import UnicrsGenConfig, UnicrsGen, UnicrsRecConfig, UnicrsGenTokenizer
from recwizard.tokenizer_utils import BaseTokenizer
from recwizard.utils import batchify, init_deterministic

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import logging

from recwizard.utils import STATE_DICT_FILE, DeviceManager


class CRSDataCollatorForConv:
    def __init__(self, tokenizer: BaseTokenizer, debug=False, use_amp=False):
        self.debug = debug
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = 8 if use_amp else None

    def __call__(self, data_batch):
        batch = batchify(data_batch)
        encodings = self.tokenizer(batch["context"], add_special_tokens=False)
        if sum(map(len, batch["resp"])) == 0:  # training
            labels = encodings["context"]["input_ids"].clone()
        else:  # generation
            labels = self.tokenizer(batch["resp"], add_special_tokens=False)["context"]["input_ids"]
        return {**encodings, "labels": labels}


class ConvTrainer(Trainer):
    def __init__(self, gen_dataset, tokenizer: BaseTokenizer, gen_file=None, debug=False, **kwargs):
        super().__init__(**kwargs)
        self.debug = debug
        if gen_file is None:
            gen_file = f'gen_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'
        os.makedirs("unicrs/log", exist_ok=True)
        self.tokenizer = tokenizer
        self.evaluator = ConvEvaluator(
            tokenizer=self.tokenizer, log_file_path=os.path.join("unicrs/log", f"{gen_file}.jsonl")
        )
        self.valid_gen_dataloader = self.get_gen_dataloader(gen_dataset["validation"])
        self.test_gen_dataloader = self.get_gen_dataloader(gen_dataset["test"])

    def get_gen_dataloader(self, gen_dataset):
        return DataLoader(
            gen_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,
        )

    def train(self, *args, **kwargs):
        return super().train(*args, **kwargs)

    def compute_loss(self, model, batch, return_outputs=False):
        batch = DeviceManager.copy_to_device(batch)
        outputs = model(**batch)
        loss = outputs.conv_loss

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        losses = []
        for batch in tqdm(dataloader):
            loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
            losses.append(float(loss))

        self.tokenizer.padding_side = "left"
        if metric_key_prefix == "eval":
            dataloader2 = self.valid_gen_dataloader
        else:
            dataloader2 = self.test_gen_dataloader
        for batch in tqdm(dataloader2):
            batch = DeviceManager.copy_to_device(batch)
            gen_seqs = self.model.generate(**batch)
            gen_resp_ids = []
            context_len = torch.sum(batch["context"]["attention_mask"], dim=1)
            for gen_seq, length in zip(gen_seqs, context_len):
                gen_seq = [token_id for token_id in gen_seq if token_id != self.tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq[length:])
            self.evaluator.evaluate(gen_resp_ids, batch["labels"], log=True)

        metrics = self.evaluator.report()
        self.evaluator.reset_metric()
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses)
        self.tokenizer.padding_side = "right"
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=metrics["sent_cnt"])

    @torch.no_grad()
    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        self.tokenizer.padding_side = "left"
        for batch in tqdm(dataloader):
            batch = DeviceManager.copy_to_device(batch)
            gen_seqs = self.model.generate(**batch)
            gen_resp_ids = []
            context_len = torch.sum(batch["context"]["attention_mask"], dim=1)
            for gen_seq, length in zip(gen_seqs, context_len):
                gen_seq = [token_id for token_id in gen_seq if token_id != self.tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq[length:])
            self.evaluator.evaluate(gen_resp_ids, batch["labels"], log=True)

        metrics = self.evaluator.report()
        self.evaluator.reset_metric()
        self.tokenizer.padding_side = "right"
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=len(dataloader))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir", type=str, default="save/unicrs_conv/redial", help="Where to store the final model."
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default="redial_unicrs", help="A file containing all data.")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--resp_max_length", type=int, default=183)
    parser.add_argument("--entity_max_length", type=int, default=32, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    # model
    parser.add_argument("--pretrained_model", type=str, default="save/unicrs_pre/redial/model_best")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-small",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--text_encoder", type=str, default="roberta-base")
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=-1,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (per device).")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--num_warmup_steps", type=int, default=6345)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--run_infer", action="store_true")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    num_workers = 0 if args.debug else args.num_workers
    num_epochs = min(2, args.num_train_epochs) if args.debug else args.num_train_epochs
    output_dir = "tmp/conv" if args.debug else args.output_dir
    init_deterministic()

    dp = UnicrsDataProcessor(args.dataset)
    kg_info = KGDataLoader.get_entity_kg_info(args.dataset)
    edge_index = kg_info.pop("edge_index")
    edge_type = kg_info.pop("edge_type")

    tokenizer = UnicrsGenTokenizer.load_from_dataset(dataset=args.dataset)

    datasets = load_dataset(os.path.join("dataset", args.dataset)).filter(
        lambda x: x["messages"][-1].startswith("System:")
    )
    if args.debug:
        for subset in datasets:
            datasets[subset] = datasets[subset].select(range(100))

    conv_dataset = datasets.map(dp.prepare_data_for_conv, fn_kwargs={"gen": False}, load_from_cache_file=False)
    gen_dataset = datasets.map(dp.prepare_data_for_conv, fn_kwargs={"gen": True}, load_from_cache_file=False)

    kgprompt_config = UnicrsRecConfig.from_pretrained(args.pretrained_model).kgprompt_config
    kgprompt_config.update(
        {
            "n_prefix_rec": None,
            "n_prefix_conv": 20,
        }
    )

    model = UnicrsGen(
        config=UnicrsGenConfig(
            pretrained_model=args.model,
            kgprompt_config=kgprompt_config,
            num_tokens=len(tokenizer.tokenizers[0]),
            pad_token_id=tokenizer.tokenizers[0].pad_token_id,
            max_gen_len=50,
        ),
        edge_index=edge_index,
        edge_type=edge_type,
    )
    model.load_checkpoint(os.path.join(args.pretrained_model, STATE_DICT_FILE), strict=False)
    model = model.to(DeviceManager.device)

    # optim
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    num_update_steps_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size / args.gradient_accumulation_steps)

    if args.num_training_steps == -1:
        num_training_steps = num_epochs * num_update_steps_per_epoch
    else:
        num_training_steps = args.num_training_steps

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, num_training_steps)
    data_collator = CRSDataCollatorForConv(
        tokenizer=tokenizer,
        use_amp=False,
        debug=args.debug,
    )
    trainer = ConvTrainer(
        model=model,
        optimizers=(optimizer, lr_scheduler),
        args=TrainingArguments(
            output_dir=output_dir,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            num_train_epochs=num_epochs,
            max_steps=num_training_steps,
            remove_unused_columns=False,
            dataloader_num_workers=num_workers,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            load_best_model_at_end=True,
            # load_best_model_at_end requires the save and eval strategy to match
            label_names=["target"],
            lr_scheduler_type="linear",
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            metric_for_best_model="eval_loss",
            report_to="none",  # disable wandb,
        ),
        train_dataset=conv_dataset["train"],
        eval_dataset=conv_dataset["validation"],
        data_collator=data_collator,
        gen_dataset=gen_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        debug=args.debug,
        tokenizer=tokenizer.tokenizers[0],
    )

    if num_epochs > 0:
        trainer.train()
        print(trainer.evaluate(conv_dataset["test"], metric_key_prefix="test"))
        trainer.save_model(os.path.join(output_dir, "model_best"))

    if args.run_infer:
        trainer.args.use_legacy_prediction_loop = True
        for subset in gen_dataset:
            logging.info(f"Generate for subset {subset}")
            trainer.evaluator.load_log_file("{}/{}_gen.jsonl".format(output_dir, subset))
            res = trainer.predict(gen_dataset[subset], metric_key_prefix=subset)
            logging.info(str(res.metrics))
