import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "trl"))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train MiniLLM on the Dolly pilot split.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--teacher", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)

    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--loss_type", choices=["grpo", "dapo"], default="grpo")
    parser.add_argument("--single_step_decomposition", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--length_normalization", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_only_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    return parser.parse_args()


def validate_args(args):
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0.0, 1.0]")
    if args.alpha > 0.0 and args.temperature != args.kd_temperature:
        raise ValueError("Teacher-mixed sampling requires --temperature == --kd_temperature")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup_steps must be in [0, max_steps)")
    if args.max_prompt_length < 1 or args.max_completion_length < 1:
        raise ValueError("Prompt and completion lengths must be at least 1")
    if args.per_device_train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Training batch size and gradient accumulation must be at least 1")
    if args.generation_batch_size < 1 or args.num_generations < 1 or args.num_iterations < 1:
        raise ValueError("Generation batch size, num_generations, and num_iterations must be at least 1")
    if args.generation_batch_size % args.num_generations != 0:
        raise ValueError("--generation_batch_size must be divisible by --num_generations")
    if args.generation_batch_size % args.per_device_train_batch_size != 0:
        raise ValueError("--generation_batch_size must be divisible by --per_device_train_batch_size")
    effective_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.generation_batch_size % effective_train_batch_size != 0:
        raise ValueError(
            "--generation_batch_size must be divisible by the effective train batch size "
            "(per-device batch size * gradient accumulation)"
        )
    if not 0.0 < args.epsilon < 1.0:
        raise ValueError("--epsilon must be in (0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("MiniLLM training requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This experiment must use exactly one visible GPU. Set CUDA_VISIBLE_DEVICES to the Blackwell GPU."
        )

    output_dir = Path(args.output_dir)
    existing_files = [] if not output_dir.exists() else list(output_dir.iterdir())
    blocking_files = [path for path in existing_files if path.name != "run_config.json"]
    if blocking_files and args.resume_from_checkpoint is None:
        raise FileExistsError(
            f"{output_dir} is not empty. Use a new output directory or pass --resume_from_checkpoint."
        )


def load_train_dataset(path, tokenizer, max_prompt_length, max_train_samples):
    dataset = load_dataset("json", data_files=path, split="train")
    if "prompt" not in dataset.column_names:
        raise ValueError(f"{path} must contain a 'prompt' column")

    dataset = dataset.map(
        lambda example: {
            "prompt_tokens": len(tokenizer.encode(example["prompt"], add_special_tokens=False)),
        },
        desc="Measuring prompt lengths",
    )
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda example: 0 < example["prompt_tokens"] <= max_prompt_length,
        desc="Dropping empty or overlong prompts",
    )
    dataset = dataset.remove_columns("prompt_tokens")
    print(f"train: kept {len(dataset)} examples, dropped {original_size - len(dataset)} overlong/empty prompts")

    if max_train_samples is not None:
        dataset = dataset.select(range(min(max_train_samples, len(dataset))))
        print(f"train smoke subset: {len(dataset)} examples")
    return dataset


def save_run_config(args, config, train_size):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "command_args": vars(args),
        "resolved": {
            "train_size": train_size,
            "effective_train_batch_size": (
                args.per_device_train_batch_size * args.gradient_accumulation_steps
            ),
            "rollout_size": args.generation_batch_size,
            "mini_batches_per_rollout": args.generation_batch_size
            // (args.per_device_train_batch_size * args.gradient_accumulation_steps),
            "optimizer_steps_per_rollout": (
                args.generation_batch_size
                // (args.per_device_train_batch_size * args.gradient_accumulation_steps)
                * args.num_iterations
            ),
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "bf16": config.bf16,
            "fp16": config.fp16,
        },
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    validate_args(args)

    bf16_ok = torch.cuda.is_bf16_supported()
    dtype_name = "bfloat16" if bf16_ok else "float16"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_train_dataset(
        args.train_file,
        tokenizer,
        args.max_prompt_length,
        args.max_train_samples,
    )

    config = MiniLLMConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        model_init_kwargs={"dtype": dtype_name},
        teacher_model_init_kwargs={"dtype": dtype_name},
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=args.generation_batch_size,
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        loss_type=args.loss_type,
        importance_sampling_level="token",
        single_step_decomposition=args.single_step_decomposition,
        length_normalization=args.length_normalization,
        rkl_advantage=True,
        gamma=0.0,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        kd_temperature=args.kd_temperature,
        teacher_mixin_alpha=args.alpha,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=None,
        repetition_penalty=args.repetition_penalty,
        use_vllm=False,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
    )
    save_run_config(args, config, len(train_dataset))

    trainer = MiniLLMTrainer(
        model=args.model,
        teacher_model=args.teacher,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"MiniLLM finished. alpha={args.alpha}; saved to {args.output_dir}")


if __name__ == "__main__":
    main()
