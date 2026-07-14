import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = PROJECT_ROOT / "trl" / "experimental" / "scripts" / "eval_dolly_generation.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Select a MiniLLM checkpoint by Dolly valid Rouge-L.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--data_file", type=Path, default=Path("data/dolly_pilot/valid.jsonl"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def checkpoint_step(path):
    return int(path.name.removeprefix("checkpoint-"))


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    data_file = args.data_file.resolve()
    output_dir = args.output_dir.resolve()
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
            with open(metrics_path, encoding="utf-8") as f:
                existing_metrics = json.load(f)
            expected_model = str(checkpoint)
            expected_data_file = str(data_file)
            if (
                existing_metrics.get("model") != expected_model
                or existing_metrics.get("data_file") != expected_data_file
            ):
                raise ValueError(f"{metrics_path} belongs to a different model or dataset; pass --force")
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
                "--qualitative_per_category",
                "0",
                "--device",
                "cuda",
            ]
            if args.max_samples is not None:
                command.extend(["--max_samples", str(args.max_samples)])
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        results.append(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "rouge_l": metrics["rouge_l"],
                "classification_accuracy": metrics["classification_accuracy"],
                "completion_truncated_examples": metrics["completion_truncated_examples"],
            }
        )

    best = max(results, key=lambda result: result["rouge_l"])
    summary = {
        "selection_metric": "valid mean Rouge-L F1 under dolly_eval_v1",
        "run_dir": str(run_dir),
        "data_file": str(data_file),
        "num_checkpoints": len(results),
        "best": best,
        "checkpoints": results,
    }
    with open(output_dir / "checkpoint_selection.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_dir / "best_checkpoint.txt", "w", encoding="utf-8") as f:
        f.write(best["checkpoint"] + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
