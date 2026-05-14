import os

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig


TERMINATION_TOKENS = ("<|im_end|>", "<|endoftext|>")

# =========================
# Experiment identity
# =========================

EXPERIMENT_NAME = "big_qwen17b_teacher4b_len128_teacher_kv_offload"
TEACHER_KV_MODE = "offload"  # one of: none / gpu / offload

# =========================
# Model / data config
# =========================

STUDENT_REPO = "Qwen/Qwen3-1.7B"
TEACHER_REPO = "Qwen/Qwen3-4B"
DATASET_NAME = "trl-lib/tldr"
MAX_COMPLETION_LENGTH = 128

# =========================
# Runtime config
# =========================
# Default is a quick 20-step resource check.
# To reproduce longer runs: RUN_MODE=full python <script.py>
# To override step count directly: MAX_STEPS=100 python <script.py>

RUN_MODE = os.environ.get("RUN_MODE", "quick").lower()
DO_EVAL = os.environ.get("DO_EVAL", "0") == "1"

if RUN_MODE == "quick":
    DEFAULT_MAX_STEPS = 20
    DEFAULT_WARMUP_STEPS = 5
    DEFAULT_LOGGING_STEPS = 5
    DEFAULT_SAVE_STEPS = 20
    DEFAULT_EVAL_STEPS = 20
    DEFAULT_EVAL_SPLIT = "validation[:50]"
elif RUN_MODE == "full":
    DEFAULT_MAX_STEPS = 200
    DEFAULT_WARMUP_STEPS = 20
    DEFAULT_LOGGING_STEPS = 10
    DEFAULT_SAVE_STEPS = 100
    DEFAULT_EVAL_STEPS = 100
    DEFAULT_EVAL_SPLIT = "validation[:200]"
else:
    raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}. Use quick or full.")

MAX_STEPS = int(os.environ.get("MAX_STEPS", str(DEFAULT_MAX_STEPS)))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", str(DEFAULT_WARMUP_STEPS)))
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", str(DEFAULT_LOGGING_STEPS)))
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", str(DEFAULT_SAVE_STEPS)))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
EVAL_SPLIT = os.environ.get("EVAL_SPLIT", DEFAULT_EVAL_SPLIT)

OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"outputs/minillm_{EXPERIMENT_NAME}_steps{MAX_STEPS}",
)


def resolve_local_model_path(repo_id: str) -> str:
    """Resolve a model from the local Hugging Face cache only."""
    print("=" * 80)
    print(f"Resolving local model path: {repo_id}")
    print("=" * 80)

    local_path = snapshot_download(
        repo_id=repo_id,
        local_files_only=True,
    )

    print(f"{repo_id} local path:")
    print(local_path)
    print("=" * 80)
    return local_path


def format_tldr_prompt(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Summarize the following Reddit post into a concise TL;DR.\n\n"
                    f"{example['prompt']}"
                ),
            }
        ]
    }


def get_generation_token_ids(tokenizer):
    eos_token_ids = []

    for token in TERMINATION_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            eos_token_ids.append(token_id)

    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(tokenizer.eos_token_id)

    return list(dict.fromkeys(eos_token_ids))


def print_tokenizer_info(tokenizer, eos_token_ids):
    print("=" * 80)
    print("Tokenizer information")
    print("=" * 80)
    print("padding_side:", tokenizer.padding_side)
    print("pad_token:", tokenizer.pad_token)
    print("pad_token_id:", tokenizer.pad_token_id)
    print("eos_token:", tokenizer.eos_token)
    print("eos_token_id:", tokenizer.eos_token_id)

    for token in TERMINATION_TOKENS:
        print(f"{token} -> {tokenizer.convert_tokens_to_ids(token)}")

    print("final eos_token_ids:", eos_token_ids)
    print("=" * 80)


def build_config(output_dir, eos_token_ids, tokenizer):
    config_kwargs = dict(
        output_dir=output_dir,

        # Basic training settings
        seed=42,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-6,
        warmup_steps=WARMUP_STEPS,

        # Logging / saving
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",

        # Memory-related settings
        bf16=True,
        gradient_checkpointing=True,

        # MiniLLM-specific settings
        rkl_advantage=True,
        single_step_decomposition=True,
        kd_temperature=1.0,
        gamma=0.0,
        length_normalization=True,

        # Rollout generation settings
        num_generations=1,
        max_completion_length=MAX_COMPLETION_LENGTH,
        chat_template_kwargs={"enable_thinking": False},
        generation_kwargs={
            "eos_token_id": eos_token_ids,
            "pad_token_id": tokenizer.pad_token_id,
        },

        # Disable evaluation by default for faster resource checks.
        eval_strategy="steps" if DO_EVAL else "no",
    )

    if DO_EVAL:
        config_kwargs["eval_steps"] = EVAL_STEPS

    return MiniLLMConfig(**config_kwargs)


def main():
    # Keep the experiment self-contained and avoid stale environment variables.
    os.environ.pop("MINILLM_TEACHER_KV_OFFLOAD", None)
    os.environ["MINILLM_TEACHER_KV_MODE"] = TEACHER_KV_MODE

    print("=" * 80)
    print(f"RUN_MODE: {RUN_MODE}")
    print(f"EXPERIMENT_NAME: {EXPERIMENT_NAME}")
    print(f"TEACHER_KV_MODE: {TEACHER_KV_MODE}")
    print(f"Student repo: {STUDENT_REPO}")
    print(f"Teacher repo: {TEACHER_REPO}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Max completion length: {MAX_COMPLETION_LENGTH}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Do eval: {DO_EVAL}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
    print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"))
    print("MINILLM_TEACHER_KV_MODE:", os.environ.get("MINILLM_TEACHER_KV_MODE"))
    print("=" * 80)

    student_path = resolve_local_model_path(STUDENT_REPO)
    teacher_path = resolve_local_model_path(TEACHER_REPO)

    tokenizer = AutoTokenizer.from_pretrained(
        student_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_token_ids = get_generation_token_ids(tokenizer)
    print_tokenizer_info(tokenizer, eos_token_ids)

    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token_id is not None
    assert len(eos_token_ids) > 0

    training_args = build_config(
        output_dir=OUTPUT_DIR,
        eos_token_ids=eos_token_ids,
        tokenizer=tokenizer,
    )

    print("=" * 80)
    print("Loading dataset")
    print("=" * 80)

    train_dataset = load_dataset(DATASET_NAME, split="train").map(format_tldr_prompt)
    eval_dataset = None
    if DO_EVAL:
        eval_dataset = load_dataset(DATASET_NAME, split=EVAL_SPLIT).map(format_tldr_prompt)

    print("train_dataset:", train_dataset)
    print("eval_dataset:", eval_dataset)

    print("=" * 80)
    print("Sample formatted prompt")
    print("=" * 80)
    print(train_dataset[0]["prompt"])

    trainer = MiniLLMTrainer(
        model=student_path,
        teacher_model=teacher_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")

    print("=" * 80)
    print("Training finished")
    print(f"Final model saved to: {OUTPUT_DIR}/final")
    print("=" * 80)


if __name__ == "__main__":
    main()
