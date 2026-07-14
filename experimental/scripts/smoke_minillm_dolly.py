import argparse
import sys
from pathlib import Path

# Always exercise the TRL checkout that belongs to this project, even when a
# different checkout has been installed in editable mode in the environment.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "trl"))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig


def build_prompt(example):
    instruction = example["instruction"].strip()
    context = example["context"].strip()

    if context:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            "### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        )

def ensure_prompt(x):
    if "prompt" in x and x["prompt"]:
        return {"prompt": x["prompt"]}
    return {"prompt": build_prompt(x)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/databricks-dolly-15k/databricks-dolly-15k.jsonl",
    )
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    ds = load_dataset("json", data_files=args.data_path, split="train")
    ds = ds.shuffle(seed=42).select(range(args.num_samples))
    ds = ds.map(ensure_prompt)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cfg = MiniLLMConfig(
        output_dir=f"outputs/dolly_pilot_minillm_alpha{args.alpha}",
        max_steps=args.max_steps,
        learning_rate=1e-6,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        num_generations=2,

        max_completion_length=64,

        temperature=1.0,
        kd_temperature=1.0,
        teacher_mixin_alpha=args.alpha,

        top_p=1.0,
        top_k=0,
        min_p=None,
        repetition_penalty=1.0,
        use_vllm=False,

        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,

        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        report_to=[],
    )

    trainer = MiniLLMTrainer(
        model=args.model,
        teacher_model=args.teacher,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.train()
    print(f"Smoke test finished. alpha={args.alpha}")


if __name__ == "__main__":
    main()
