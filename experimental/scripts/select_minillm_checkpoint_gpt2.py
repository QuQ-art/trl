import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = PROJECT_ROOT / "trl" / "experimental" / "scripts" / "eval_dolly_generation_gpt2.py"
PROTOCOL_NAME = "dolly_eval_gpt2_compatible_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Select a GPT-2 MiniLLM checkpoint by valid Rouge-L.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_file", type=Path, default=Path("data/dolly_pilot/valid.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_step(path):
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"Invalid checkpoint directory name: {path.name}") from error


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_input_length < 1 or args.max_new_tokens < 1:
        raise ValueError("batch and length arguments must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1")

    run_dir = args.run_dir.resolve()
    data_file = args.data_file.resolve()
    output_dir = args.output_dir.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* directories found in {run_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        checkpoint_output = output_dir / f"checkpoint-{step}"
        metrics_path = checkpoint_output / "metrics.json"
        if metrics_path.exists() and not args.force:
            with metrics_path.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            expected = {
                "model": str(checkpoint),
                "data_file": str(data_file),
                "evaluation_protocol": PROTOCOL_NAME,
            }
            mismatches = {
                key: (existing.get(key), value)
                for key, value in expected.items()
                if existing.get(key) != value
            }
            generation = existing.get("generation", {})
            if generation.get("max_input_length") != args.max_input_length:
                mismatches["max_input_length"] = (
                    generation.get("max_input_length"),
                    args.max_input_length,
                )
            if generation.get("max_new_tokens") != args.max_new_tokens:
                mismatches["max_new_tokens"] = (
                    generation.get("max_new_tokens"),
                    args.max_new_tokens,
                )
            if mismatches:
                raise ValueError(f"{metrics_path} protocol mismatch: {mismatches}; pass --force")
        if args.force or not metrics_path.exists():
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--model",
                str(checkpoint),
                "--data_file",
                str(data_file),
                "--output_dir",
                str(checkpoint_output),
                "--batch_size",
                str(args.batch_size),
                "--max_input_length",
                str(args.max_input_length),
                "--max_new_tokens",
                str(args.max_new_tokens),
                "--qualitative_per_category",
                "0",
                "--device",
                "cuda",
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        with metrics_path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        results.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "rouge_l": metrics["rouge_l"],
                "classification_accuracy": metrics["classification_accuracy"],
                "input_truncated_examples": metrics["input_truncated_examples"],
                "completion_truncated_examples": metrics["completion_truncated_examples"],
            }
        )

    best = max(results, key=lambda result: result["rouge_l"])
    summary = {
        "selection_metric": f"valid mean Rouge-L F1 under {PROTOCOL_NAME}",
        "evaluation_protocol": PROTOCOL_NAME,
        "run_dir": str(run_dir),
        "data_file": str(data_file),
        "num_checkpoints": len(results),
        "best": best,
        "checkpoints": results,
    }
    with (output_dir / "checkpoint_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "best_checkpoint.txt").open("w", encoding="utf-8") as handle:
        handle.write(best["checkpoint"] + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
