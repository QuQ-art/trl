import json
import random
from pathlib import Path
from datasets import load_dataset


DATA_PATH = "data/databricks-dolly-15k/databricks-dolly-15k.jsonl"
OUT_DIR = Path("data/dolly_pilot")
SEED = 42

N_TRAIN = 1000
N_VALID = 200
N_TEST = 200


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


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.shuffle(seed=SEED)

    total = N_TRAIN + N_VALID + N_TEST
    ds = ds.select(range(total))

    rows = []
    for ex in ds:
        rows.append(
            {
                "prompt": build_prompt(ex),
                "response": ex["response"].strip(),
                "category": ex.get("category", ""),
                "instruction": ex["instruction"],
                "context": ex["context"],
            }
        )

    train = rows[:N_TRAIN]
    valid = rows[N_TRAIN:N_TRAIN + N_VALID]
    test = rows[N_TRAIN + N_VALID:]

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "valid.jsonl", valid)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    print(f"train: {len(train)}")
    print(f"valid: {len(valid)}")
    print(f"test:  {len(test)}")
    print(f"saved to {OUT_DIR}")


if __name__ == "__main__":
    main()