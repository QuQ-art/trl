import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args():
    parser = argparse.ArgumentParser(description="SFT a GPT-2 student or teacher on the fixed Dolly pilot split.")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", default="data/dolly_pilot/train.jsonl")
    parser.add_argument("--eval_file", default="data/dolly_pilot/valid.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--save_only_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    return parser.parse_args()


def get_context_length(config):
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine the model context length")


def validate_args(args):
    if not torch.cuda.is_available():
        raise RuntimeError("GPT-2 v1 SFT requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")
    if args.max_length < 1 or args.learning_rate <= 0:
        raise ValueError("--max_length and --learning_rate must be positive")
    if args.max_steps == 0 or args.num_train_epochs <= 0:
        raise ValueError("--max_steps cannot be zero and --num_train_epochs must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must be in [0, 1)")
    if args.eval_steps < 1 or args.save_steps < 1:
        raise ValueError("--eval_steps and --save_steps must be at least 1")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save_steps must be a multiple of --eval_steps")
    for name in (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; use a new output directory")


def main():
    args = parse_args()
    validate_args(args)

    model_config = AutoConfig.from_pretrained(args.model_name)
    if model_config.model_type != "gpt2":
        raise ValueError(f"GPT-2 v1 requires model_type='gpt2', got {model_config.model_type!r}")
    context_length = get_context_length(model_config)
    if args.max_length > context_length:
        raise ValueError(f"--max_length={args.max_length} exceeds model context length {context_length}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ok = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype)
    # GPT2LMHeadModel's class name does not match Transformers' generic loss-name inference.
    # This is the same default selected by the warning, made explicit to keep formal logs clean.
    model.loss_type = "ForCausalLM"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    required_columns = {"prompt", "response"}
    for name, dataset in (("train", train_dataset), ("eval", eval_dataset)):
        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(f"{name} dataset is missing columns: {sorted(missing)}")

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    def tokenize_example(example):
        prompt = example["prompt"]
        response = example["response"].strip()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
        full = tokenizer(
            prompt + response + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        labels = full["input_ids"].copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
            "has_supervised_tokens": any(label != -100 for label in labels),
        }

    def tokenize_and_filter(dataset, split_name):
        original_size = len(dataset)
        dataset = dataset.map(
            tokenize_example,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing GPT-2 {split_name}",
        )
        dataset = dataset.filter(
            lambda example: example["has_supervised_tokens"],
            desc=f"Dropping GPT-2 {split_name} examples without response tokens",
        )
        dataset = dataset.remove_columns("has_supervised_tokens")
        print(f"{split_name}: kept {len(dataset)}, dropped {original_size - len(dataset)}")
        if len(dataset) == 0:
            raise ValueError(f"No supervised {split_name} examples remain")
        return dataset

    train_dataset = tokenize_and_filter(train_dataset, "train")
    eval_dataset = tokenize_and_filter(eval_dataset, "eval")

    def collate_fn(features):
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collate_fn,
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    run_config = {
        "experiment": "gpt2_v1_sft",
        "command_args": vars(args),
        "resolved": {
            "model_type": model_config.model_type,
            "model_context_length": context_length,
            "vocab_size": len(tokenizer),
            "loss_type": model.loss_type,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset),
            "effective_train_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            ),
            "bf16": bf16_ok,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "best_eval_loss": trainer.state.best_metric,
        },
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)
    print(f"SFT finished; best checkpoint={trainer.state.best_model_checkpoint}; saved to {output_dir}")


if __name__ == "__main__":
    main()
