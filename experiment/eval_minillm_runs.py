import gc
import math
import os
import time
from datetime import datetime
from pathlib import Path


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ROOT_DIR = Path("/home/xxf/Distill")
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT_DIR / ".hf_cache/datasets"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


RUNS_DIR = ROOT_DIR / "Qwen3.5-0.8B-Base-MiniLLM/runs"
STUDENT_BASELINE = ROOT_DIR / "models/Qwen3.5-0.8B-Base"
TEACHER_BASELINE = ROOT_DIR / "models/Qwen3.5-2B"
DATASET_PATH = ROOT_DIR / "data-tldr"
RESULT_PATH = ROOT_DIR / "Qwen3.5-0.8B-Base-MiniLLM/student_eval_results.md"

EVAL_SAMPLES = 1000
BATCH_SIZE = 8
MAX_LENGTH = 2048


def find_eval_targets():
    targets = [
        {
            "run": "baseline",
            "checkpoint": "student_untrained",
            "path": STUDENT_BASELINE,
        },
        {
            "run": "baseline",
            "checkpoint": "teacher",
            "path": TEACHER_BASELINE,
        },
    ]
    checkpoints = []
    for checkpoint in RUNS_DIR.glob("*/checkpoint-*"):
        if (checkpoint / "model.safetensors").is_file() or (checkpoint / "pytorch_model.bin").is_file():
            checkpoints.append(checkpoint)
    for checkpoint in sorted(checkpoints, key=lambda path: (path.parent.name, int(path.name.split("-")[-1]))):
        targets.append(
            {
                "run": checkpoint.parent.name,
                "checkpoint": checkpoint.name,
                "path": checkpoint,
            }
        )
    return targets


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(rows):
    lines = [
        "# MiniLLM Student-Only Eval Results",
        "",
        f"- Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Device: physical cuda:2 via `CUDA_VISIBLE_DEVICES=2`",
        f"- Dataset: `{DATASET_PATH}`, split `train`, first {EVAL_SAMPLES} samples",
        f"- Metric: completion NLL/perplexity under the student only; no teacher model is loaded.",
        "",
        "|run|checkpoint|completion_nll|completion_ppl|tokens|runtime_sec|tokens_per_second|error|",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "|{run}|{checkpoint}|{completion_nll}|{completion_ppl}|{tokens}|{runtime_sec}|"
            "{tokens_per_second}|{error}|".format(**row)
        )
    RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing_rows():
    if not RESULT_PATH.is_file():
        return []
    rows = []
    for line in RESULT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or line.startswith("|run|"):
            continue
        values = [value.replace("\\|", "|") for value in line[1:-1].split("|")]
        if len(values) != 8:
            continue
        rows.append(
            {
                "run": values[0],
                "checkpoint": values[1],
                "completion_nll": values[2],
                "completion_ppl": values[3],
                "tokens": values[4],
                "runtime_sec": values[5],
                "tokens_per_second": values[6],
                "error": values[7],
            }
        )
    return rows


def build_completion_batch(tokenizer, examples):
    prompt_texts = [example["prompt"] for example in examples]
    full_texts = [example["prompt"] + example["completion"] for example in examples]

    prompt_inputs = tokenizer(prompt_texts, add_special_tokens=False)
    full_inputs = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    for row_idx, prompt_ids in enumerate(prompt_inputs["input_ids"]):
        prompt_len = min(len(prompt_ids), labels.shape[1])
        labels[row_idx, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100

    return {
        "input_ids": full_inputs["input_ids"],
        "attention_mask": full_inputs["attention_mask"],
        "labels": labels,
    }


def batch_completion_nll(model, batch):
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    labels = batch["labels"].to(model.device)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels != -100
    if not valid_mask.any():
        return 0.0, 0

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(shift_labels)
    total_loss = token_losses[valid_mask].sum().item()
    total_tokens = valid_mask.sum().item()
    return total_loss, total_tokens


def evaluate_model(model_path, dataset):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": "cuda:0"},
    )
    model.eval()

    started = time.time()
    total_loss = 0.0
    total_tokens = 0
    try:
        for start in range(0, len(dataset), BATCH_SIZE):
            examples = [dataset[idx] for idx in range(start, min(start + BATCH_SIZE, len(dataset)))]
            batch = build_completion_batch(tokenizer, examples)
            batch_loss, batch_tokens = batch_completion_nll(model, batch)
            total_loss += batch_loss
            total_tokens += batch_tokens
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    runtime = time.time() - started
    nll = total_loss / total_tokens
    return {
        "completion_nll": nll,
        "completion_ppl": math.exp(nll),
        "tokens": total_tokens,
        "runtime_sec": runtime,
        "tokens_per_second": total_tokens / runtime,
    }


def main():
    dataset = load_dataset(str(DATASET_PATH), split="train").select(range(EVAL_SAMPLES))
    targets = find_eval_targets()

    rows = load_existing_rows()
    done = {(row["run"], row["checkpoint"]) for row in rows if not row["error"]}
    for index, target in enumerate(targets, start=1):
        key = (target["run"], target["checkpoint"])
        if key in done:
            print(f"[{index}/{len(targets)}] skipping {target['path']}")
            continue

        print(f"[{index}/{len(targets)}] evaluating {target['path']}")
        try:
            metrics = evaluate_model(target["path"], dataset)
            row = {
                "run": target["run"],
                "checkpoint": target["checkpoint"],
                "completion_nll": fmt(metrics["completion_nll"]),
                "completion_ppl": fmt(metrics["completion_ppl"]),
                "tokens": fmt(metrics["tokens"]),
                "runtime_sec": fmt(metrics["runtime_sec"]),
                "tokens_per_second": fmt(metrics["tokens_per_second"]),
                "error": "",
            }
        except Exception as error:
            row = {
                "run": target["run"],
                "checkpoint": target["checkpoint"],
                "completion_nll": "",
                "completion_ppl": "",
                "tokens": "",
                "runtime_sec": "",
                "tokens_per_second": "",
                "error": fmt(f"{type(error).__name__}: {error}"),
            }
        rows.append(row)
        write_markdown(rows)

    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
