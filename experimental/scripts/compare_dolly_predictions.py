import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Paired comparison of two Dolly prediction files.")
    parser.add_argument("--baseline_predictions", type=Path, required=True)
    parser.add_argument("--candidate_predictions", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tie_tolerance", type=float, default=1e-12)
    return parser.parse_args()


def load_predictions(path):
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "index" not in row or "rouge_l" not in row:
                raise ValueError(f"{path}:{line_number} must contain index and rouge_l")
            index = int(row["index"])
            if index in predictions:
                raise ValueError(f"{path} contains duplicate index {index}")
            predictions[index] = row
    if not predictions:
        raise ValueError(f"{path} contains no predictions")
    return predictions


def main():
    args = parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be at least 1")
    if args.tie_tolerance < 0:
        raise ValueError("--tie_tolerance cannot be negative")

    baseline = load_predictions(args.baseline_predictions)
    candidate = load_predictions(args.candidate_predictions)
    if set(baseline) != set(candidate):
        raise ValueError("Prediction files do not contain identical example indices")

    indices = sorted(baseline)
    for index in indices:
        if baseline[index].get("reference") != candidate[index].get("reference"):
            raise ValueError(f"Reference mismatch at index {index}")
    baseline_scores = np.asarray([baseline[index]["rouge_l"] for index in indices], dtype=np.float64)
    candidate_scores = np.asarray([candidate[index]["rouge_l"] for index in indices], dtype=np.float64)
    deltas = candidate_scores - baseline_scores

    rng = np.random.default_rng(args.seed)
    bootstrap_means = np.empty(args.bootstrap_samples, dtype=np.float64)
    chunk_size = 1000
    for start in range(0, args.bootstrap_samples, chunk_size):
        end = min(start + chunk_size, args.bootstrap_samples)
        sampled_indices = rng.integers(0, len(deltas), size=(end - start, len(deltas)))
        bootstrap_means[start:end] = deltas[sampled_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])

    wins = int(np.sum(deltas > args.tie_tolerance))
    losses = int(np.sum(deltas < -args.tie_tolerance))
    ties = len(deltas) - wins - losses

    classification_indices = [
        index
        for index in indices
        if baseline[index].get("classification_match") is not None
        and candidate[index].get("classification_match") is not None
    ]
    classification = None
    if classification_indices:
        baseline_only = sum(
            bool(baseline[index]["classification_match"])
            and not bool(candidate[index]["classification_match"])
            for index in classification_indices
        )
        candidate_only = sum(
            bool(candidate[index]["classification_match"])
            and not bool(baseline[index]["classification_match"])
            for index in classification_indices
        )
        both_correct = sum(
            bool(candidate[index]["classification_match"])
            and bool(baseline[index]["classification_match"])
            for index in classification_indices
        )
        classification = {
            "num_examples": len(classification_indices),
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "both_incorrect": len(classification_indices) - both_correct - baseline_only - candidate_only,
        }

    report = {
        "comparison": "candidate minus baseline",
        "baseline_predictions": str(args.baseline_predictions),
        "candidate_predictions": str(args.candidate_predictions),
        "num_examples": len(indices),
        "baseline_mean_rouge_l": float(baseline_scores.mean()),
        "candidate_mean_rouge_l": float(candidate_scores.mean()),
        "mean_delta_rouge_l": float(deltas.mean()),
        "mean_delta_rouge_l_points": float(deltas.mean() * 100.0),
        "paired_bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "confidence_level": 0.95,
            "lower_rouge_l": float(lower),
            "upper_rouge_l": float(upper),
            "lower_rouge_l_points": float(lower * 100.0),
            "upper_rouge_l_points": float(upper * 100.0),
        },
        "win_tie_loss": {"candidate_wins": wins, "ties": ties, "candidate_losses": losses},
        "classification_paired": classification,
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
