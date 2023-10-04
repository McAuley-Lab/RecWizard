import json
import sys
sys.path.append("./")
sys.path.append("./src")
import math
from evaluator import RecEvaluator
from data_processor import UnicrsDataProcessor, KGDataLoader
from pretrain import PromptTrainer, CRSDataCollatorForRec
from transformers import TrainingArguments

import argparse
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

from recwizard.modules.unicrs import UnicrsRec, UnicrsRecTokenizer
from recwizard.utility import STATE_DICT_FILE, DeviceManager, init_deterministic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save/unicrs_rec/redial', help="Where to store the final model.")
    parser.add_argument("--debug", "-d", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default="redial_unicrs", help="A file containing all data.")
    parser.add_argument("--gen_data", type=str, default='save/unicrs_conv/redial/{}_gen.jsonl',
                        help="A formatter for generated results.")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument("--context_max_length", type=int, default=200, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int, default=200)
    parser.add_argument("--entity_max_length", type=int, default=32, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str, default="microsoft/DialoGPT-small")
    parser.add_argument("--text_tokenizer", type=str, default="roberta-base")
    # model
    parser.add_argument("--pretrained_model", type=str, default="save/unicrs_pre/redial/model_best")

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
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int, default=530)
    parser.add_argument("--fp16", action='store_true')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    num_workers = 0 if args.debug else args.num_workers
    num_epochs = 2 if args.debug else args.num_train_epochs
    output_dir = "tmp/rec" if args.debug else args.output_dir

    init_deterministic()

    dp = UnicrsDataProcessor(args.dataset)
    kg_info = KGDataLoader.get_entity_kg_info(args.dataset)

    config = UnicrsRec.config_class.from_pretrained(args.pretrained_model)

    model = UnicrsRec(config=config, use_rec_prefix=True)
    model.load_checkpoint(os.path.join(args.pretrained_model, STATE_DICT_FILE), strict=False)
    model = model.to(DeviceManager.device)

    tokenizer = UnicrsRecTokenizer.load_from_dataset(dataset=args.dataset)

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

    datasets = load_dataset(os.path.join("dataset", args.dataset)).filter(lambda x: x['messages'][-1].startswith('System:'))
    if args.debug:
        for subset in datasets:
            datasets[subset] = datasets[subset].select(range(10))


    for subset in datasets:
        gen_data = []
        for line in open(args.gen_data.format(subset), 'r'):
            gen_data.append(json.loads(line))
        datasets[subset] = datasets[subset].map(dp.prepare_data_for_rec, batched=True, with_indices=True,
                                                fn_kwargs={"gen_data": gen_data}, remove_columns=["recNames"], load_from_cache_file=False)


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
                               report_to="none"  # disable wandb
                               ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CRSDataCollatorForRec(tokenizer),
        evaulator=RecEvaluator(item_ids=kg_info['item_ids'])
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    trainer.train()
    print(trainer.evaluate(datasets["test"], metric_key_prefix="test"))
    model.save_pretrained(os.path.join(output_dir, 'model_best'))
