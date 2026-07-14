import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Dolly pilot lengths for the GPT-2 v1 experiment.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--train_file", type=Path, required=True)
    parser.add_argument("--valid_file", type=Path, required=True)
    parser.add_argument("--test_file", type=Path, required=True)
    parser.add_argument("--train_prompt_limit", type=int, default=896)
    parser.add_argument("--eval_prompt_limit", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_eval_new_tokens", type=int, default=256)
    parser.add_argument("--sft_max_length", type=int, default=1024)
    parser.add_argument("--output_file", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("prompt") or "response" not in row:
                raise ValueError(f"{path}:{line_number} must contain a non-empty prompt and a response")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no examples")
    return rows


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_context_length(config):
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine the tokenizer model's context length")


def summarize_split(path, rows, tokenizer, limit, sft_max_length):
    token_lengths = [
        len(tokenizer.encode(row["prompt"], add_special_tokens=False, verbose=False)) for row in rows
    ]
    over_limit = [
        {
            "index": index,
            "prompt_tokens": length,
            "category": rows[index].get("category", ""),
        }
        for index, length in enumerate(token_lengths)
        if length > limit
    ]
    empty_prompt_indices = [index for index, length in enumerate(token_lengths) if length == 0]
    sft_without_supervision = []
    for index, row in enumerate(rows):
        full_ids = tokenizer.encode(
            row["prompt"] + row["response"].strip() + tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=sft_max_length,
            verbose=False,
        )
        if len(full_ids) <= min(token_lengths[index], len(full_ids)):
            sft_without_supervision.append(index)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "num_examples": len(rows),
        "prompt_limit": limit,
        "min_prompt_tokens": min(token_lengths),
        "max_prompt_tokens": max(token_lengths),
        "mean_prompt_tokens": sum(token_lengths) / len(token_lengths),
        "eligible_examples": len(rows) - len(over_limit) - len(empty_prompt_indices),
        "over_limit_examples": len(over_limit),
        "over_limit": over_limit,
        "empty_prompt_indices": empty_prompt_indices,
        "sft_max_length": sft_max_length,
        "sft_eligible_examples": len(rows) - len(sft_without_supervision),
        "sft_without_supervision_indices": sft_without_supervision,
    }


def main():
    args = parse_args()
    for name in (
        "train_prompt_limit",
        "eval_prompt_limit",
        "max_completion_length",
        "max_eval_new_tokens",
        "sft_max_length",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    config = AutoConfig.from_pretrained(args.tokenizer)
    context_length = get_context_length(config)
    if args.train_prompt_limit + args.max_completion_length > context_length:
        raise ValueError("train prompt + completion lengths exceed the model context length")
    if args.eval_prompt_limit + args.max_eval_new_tokens > context_length:
        raise ValueError("eval prompt + generated lengths exceed the model context length")

    paths = {
        "train": args.train_file.resolve(),
        "valid": args.valid_file.resolve(),
        "test": args.test_file.resolve(),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")

    splits = {
        "train": summarize_split(
            paths["train"],
            load_jsonl(paths["train"]),
            tokenizer,
            args.train_prompt_limit,
            args.sft_max_length,
        ),
        "valid": summarize_split(
            paths["valid"],
            load_jsonl(paths["valid"]),
            tokenizer,
            args.eval_prompt_limit,
            args.sft_max_length,
        ),
        "test": summarize_split(
            paths["test"],
            load_jsonl(paths["test"]),
            tokenizer,
            args.eval_prompt_limit,
            args.sft_max_length,
        ),
    }
    report = {
        "experiment": "gpt2_v1_model_swap",
        "tokenizer": args.tokenizer,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "model_type": config.model_type,
        "model_context_length": context_length,
        "limits": {
            "train_prompt": args.train_prompt_limit,
            "train_completion": args.max_completion_length,
            "eval_prompt": args.eval_prompt_limit,
            "eval_generation": args.max_eval_new_tokens,
            "sft_max_length": args.sft_max_length,
        },
        "splits": splits,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    for split_name, split in splits.items():
        print(
            f"{split_name}: {split['num_examples']} examples; "
            f"eligible={split['eligible_examples']}; over_limit={split['over_limit_examples']}; "
            f"max_prompt_tokens={split['max_prompt_tokens']}; "
            f"sft_eligible={split['sft_eligible_examples']}"
        )
    print(f"Audit saved to {args.output_file}")


if __name__ == "__main__":
    main()
