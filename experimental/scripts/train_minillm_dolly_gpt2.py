import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "trl"))

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


def build_parser(*, smoke_defaults=False):
    parser = argparse.ArgumentParser(description="Train the GPT-2 v1 MiniLLM model-swap experiment.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--max_steps", type=int, default=3 if smoke_defaults else 1000)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", default="constant_with_warmup")
    parser.add_argument("--warmup_steps", type=int, default=0 if smoke_defaults else 20)
    parser.add_argument("--max_prompt_length", type=int, default=896)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--max_train_samples", type=int, default=16 if smoke_defaults else None)
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
    parser.add_argument("--logging_steps", type=int, default=1 if smoke_defaults else 10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--save_only_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser


def get_context_length(config):
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Could not determine the model context length")


def validate_args(args):
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")
    if args.alpha > 0.0 and args.temperature != args.kd_temperature:
        raise ValueError("teacher-mixed sampling requires equal temperature and kd_temperature")
    if args.max_steps < 1:
        raise ValueError("--max_steps must be at least 1")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup_steps must be in [0, max_steps)")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive")
    for name in (
        "max_prompt_length",
        "max_completion_length",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "generation_batch_size",
        "num_generations",
        "num_iterations",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1")
    if args.max_train_samples is not None and args.max_train_samples < 1:
        raise ValueError("--max_train_samples must be at least 1")
    if args.generation_batch_size % args.num_generations != 0:
        raise ValueError("--generation_batch_size must be divisible by --num_generations")
    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.generation_batch_size % effective_batch_size != 0:
        raise ValueError("--generation_batch_size must be divisible by the effective train batch size")
    if not 0.0 < args.epsilon < 1.0:
        raise ValueError("--epsilon must be in (0, 1)")
    if not torch.cuda.is_available():
        raise RuntimeError("GPT-2 MiniLLM training requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one GPU with CUDA_VISIBLE_DEVICES=0")

    output_dir = Path(args.output_dir)
    existing = [] if not output_dir.exists() else list(output_dir.iterdir())
    blocking = [path for path in existing if path.name != "run_config.json"]
    if blocking and args.resume_from_checkpoint is None:
        raise FileExistsError(
            f"{output_dir} is not empty; use a new directory or pass --resume_from_checkpoint"
        )


def validate_models_and_tokenizers(args):
    student_config = AutoConfig.from_pretrained(args.model)
    teacher_config = AutoConfig.from_pretrained(args.teacher)
    if student_config.model_type != "gpt2" or teacher_config.model_type != "gpt2":
        raise ValueError("GPT-2 v1 requires GPT-2 student and teacher checkpoints")
    student_context = get_context_length(student_config)
    teacher_context = get_context_length(teacher_config)
    required_context = args.max_prompt_length + args.max_completion_length
    if required_context > student_context or required_context > teacher_context:
        raise ValueError(
            f"prompt + completion length {required_context} exceeds student/teacher context "
            f"({student_context}/{teacher_context})"
        )
    if student_config.vocab_size != teacher_config.vocab_size:
        raise ValueError("Student and teacher vocab sizes differ")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("Student and teacher tokenizers do not have identical vocabularies")
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("GPT-2 tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, student_config, teacher_config, student_context, teacher_context


def load_train_dataset(args, tokenizer):
    dataset = load_dataset("json", data_files=args.train_file, split="train")
    if "prompt" not in dataset.column_names:
        raise ValueError(f"{args.train_file} must contain a prompt column")

    def measure(example, index):
        return {
            "source_index": index,
            "prompt_tokens": len(
                tokenizer.encode(example["prompt"], add_special_tokens=False, verbose=False)
            ),
        }

    dataset = dataset.map(measure, with_indices=True, desc="Measuring GPT-2 prompt lengths")
    original_size = len(dataset)
    dataset = dataset.filter(
        lambda example: 0 < example["prompt_tokens"] <= args.max_prompt_length,
        desc="Filtering GPT-2 prompts to the frozen context limit",
    )
    eligible_indices = list(dataset["source_index"])
    dropped_indices = sorted(set(range(original_size)) - set(eligible_indices))
    dataset = dataset.remove_columns(["source_index", "prompt_tokens"])
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("No training examples remain after prompt filtering")
    print(
        f"train: original={original_size}; eligible={len(eligible_indices)}; "
        f"selected={len(dataset)}; dropped={len(dropped_indices)}"
    )
    return dataset, original_size, eligible_indices, dropped_indices


def run(args):
    validate_args(args)
    tokenizer, student_config, teacher_config, student_context, teacher_context = (
        validate_models_and_tokenizers(args)
    )
    train_dataset, original_size, eligible_indices, dropped_indices = load_train_dataset(args, tokenizer)

    bf16_ok = torch.cuda.is_bf16_supported()
    dtype_name = "bfloat16" if bf16_ok else "float16"
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment": "gpt2_v1_minillm",
        "command_args": vars(args),
        "resolved": {
            "student_model_type": student_config.model_type,
            "teacher_model_type": teacher_config.model_type,
            "student_context_length": student_context,
            "teacher_context_length": teacher_context,
            "vocab_size": len(tokenizer),
            "train_original_size": original_size,
            "train_eligible_size": len(eligible_indices),
            "train_selected_size": len(train_dataset),
            "eligible_source_indices": eligible_indices,
            "dropped_source_indices": dropped_indices,
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
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    trainer = MiniLLMTrainer(
        model=args.model,
        teacher_model=args.teacher,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"GPT-2 MiniLLM finished; alpha={args.alpha}; saved to {output_dir}")


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
