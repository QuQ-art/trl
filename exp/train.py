import os
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from safetensors import safe_open
from transformers import AutoTokenizer
from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig


def _prefer_train_split(files: list[Path]) -> list[str]:
    train_files = sorted(str(p) for p in files if "train" in p.name.lower())
    if train_files:
        return train_files
    return sorted(str(p) for p in files)


def validate_local_model(model_path: str, model_name: str):
    """
    尽早检查本地 safetensors 分片是否可读，避免在 Trainer 初始化深处才报错。
    """
    model_dir = Path(os.path.expanduser(model_path)).resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"{model_name} model directory does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise ValueError(f"{model_name} model path must be a directory: {model_dir}")

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_names = sorted(set(index_data.get("weight_map", {}).values()))
    else:
        shard_names = sorted(p.name for p in model_dir.glob("*.safetensors"))

    if not shard_names:
        print(f"[WARN] No safetensors shards found under {model_dir}, skip weight validation.")
        return

    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"{model_name} shard is missing: {shard_path}")

        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                _ = len(f.keys())
        except Exception as e:
            raise ValueError(
                f"{model_name} shard is invalid: {shard_path}. "
                "The file is likely incomplete or corrupted, please re-download this model."
            ) from e


def load_local_tldr(data_dir: str):
    """
    从本地文件夹读取 trl-lib/tldr 数据。

    适配这种情况：
    hf download trl-lib/tldr --repo-type dataset --local-dir /home/xxf/Distill/data-tldr

    目录下面通常会有 parquet 文件。
    """
    data_dir = Path(os.path.expanduser(data_dir)).resolve()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    if not data_dir.is_dir():
        raise ValueError(f"data_dir must be a directory: {data_dir}")

    parquet_candidates = sorted(p for p in data_dir.rglob("*.parquet"))
    jsonl_candidates = sorted(p for p in data_dir.rglob("*.jsonl"))
    json_candidates = sorted(
        p
        for p in data_dir.rglob("*.json")
        if p.name not in {"dataset_infos.json"}
    )

    parquet_files = _prefer_train_split(parquet_candidates)
    jsonl_files = _prefer_train_split(jsonl_candidates)
    json_files = _prefer_train_split(json_candidates)

    if parquet_files:
        print(f"[INFO] Loading parquet files from {data_dir}")
        dataset = load_dataset("parquet", data_files=parquet_files, split="train")
    elif jsonl_files:
        print(f"[INFO] Loading jsonl files from {data_dir}")
        dataset = load_dataset("json", data_files=jsonl_files, split="train")
    elif json_files:
        print(f"[INFO] Loading json files from {data_dir}")
        dataset = load_dataset("json", data_files=json_files, split="train")
    else:
        raise FileNotFoundError(
            f"No parquet/jsonl/json files found under {data_dir}"
        )

    if "prompt" not in dataset.column_names:
        raise ValueError(
            f"Dataset must contain a 'prompt' column, got columns: {dataset.column_names}"
        )

    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="~/Distill/data-tldr",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="~/models/Qwen3.5-0.8B-Base",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="~/models/Qwen3.5-2B",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/minillm-qwen-tldr-demo",
    )

    # demo 默认只跑一点点，先验证流程
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=20)

    # batch 参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=2)

    # 长度参数
    parser.add_argument("--max_completion_length", type=int, default=1024)

    # 优化参数
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    data_path = os.path.expanduser(args.data_path)
    student_model = os.path.expanduser(args.student_model)
    teacher_model = os.path.expanduser(args.teacher_model)

    print("=" * 80)
    print(f"[INFO] data_path     = {data_path}")
    print(f"[INFO] student_model = {student_model}")
    print(f"[INFO] teacher_model = {teacher_model}")
    print(f"[INFO] output_dir    = {args.output_dir}")
    print("=" * 80)

    validate_local_model(student_model, "Student")
    validate_local_model(teacher_model, "Teacher")

    train_dataset = load_local_tldr(data_path)

    print("[INFO] Raw dataset:")
    print(train_dataset)
    print("[INFO] Columns:", train_dataset.column_names)
    print("[INFO] First sample:")
    print(train_dataset[0])

    if args.max_samples is not None and args.max_samples > 0:
        n = min(args.max_samples, len(train_dataset))
        train_dataset = train_dataset.select(range(n))
        print(f"[INFO] Use first {n} samples for demo.")

    tokenizer = AutoTokenizer.from_pretrained(
        student_model,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # 简单检查 student / teacher tokenizer 是否大致一致
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_model,
        trust_remote_code=True,
    )

    if len(tokenizer) != len(teacher_tokenizer):
        raise ValueError(
            f"Student and teacher tokenizer size mismatch: "
            f"{len(tokenizer)} vs {len(teacher_tokenizer)}. "
            "MiniLLM token-level KD normally requires aligned tokenizers."
        )

    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(
            "Student and teacher tokenizers are not token-id aligned. "
            "MiniLLM token-level KD requires the same vocabulary mapping."
        )

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    if bf16_ok:
        dtype = "bfloat16"
    elif torch.cuda.is_available():
        dtype = "float16"
    else:
        dtype = "float32"

    print(f"[INFO] bf16 = {bf16_ok}")
    print(f"[INFO] dtype = {dtype}")

    training_args = MiniLLMConfig(
        output_dir=args.output_dir,

        # training
        seed=args.seed,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,

        # generation
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        top_p=1.0,
        top_k=0,

        # MiniLLM loss
        rkl_advantage=True,
        single_step_decomposition=True,
        kd_temperature=1.0,
        gamma=0.0,
        length_normalization=True,

        # memory
        gradient_checkpointing=True,
        use_cache=False,

        # precision
        bf16=bf16_ok,
        fp16=torch.cuda.is_available() and not bf16_ok,
        tf32=True,

        # logging / saving
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=2,
        report_to="none",

        # model loading kwargs
        model_init_kwargs={
            "dtype": dtype,
            "trust_remote_code": True,
        },
        teacher_model_init_kwargs={
            "dtype": dtype,
            "trust_remote_code": True,
        },
    )

    trainer = MiniLLMTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("=" * 80)
    print(f"[DONE] Saved student model to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
