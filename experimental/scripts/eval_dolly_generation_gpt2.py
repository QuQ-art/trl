import argparse
import json
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


EVAL_PROTOCOL = {
    "name": "dolly_eval_gpt2_compatible_v1",
    "do_sample": False,
    "max_input_length": 768,
    "max_new_tokens": 256,
    "truncation_strategy": "middle_keep_head_and_tail",
    "stop_string": "\n\n###",
    "seed": 42,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-2 or locked Qwen checkpoints under the common 1024-token protocol."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=EVAL_PROTOCOL["max_input_length"])
    parser.add_argument("--max_new_tokens", type=int, default=EVAL_PROTOCOL["max_new_tokens"])
    parser.add_argument("--stop_string", default=EVAL_PROTOCOL["stop_string"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=EVAL_PROTOCOL["seed"])
    parser.add_argument("--qualitative_per_category", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("prompt") or "response" not in row:
                raise ValueError(f"{path}:{line_number} must contain a non-empty prompt and a response")
            rows.append(row)
    return rows


def normalize_answer(text):
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def mean(values):
    return sum(values) / len(values) if values else None


def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cuda":
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device("cpu")


def get_dtype(device):
    if device.type == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def get_context_length(config):
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def get_eos_token_ids(model, tokenizer):
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return set(), None
    if isinstance(eos_token_id, int):
        return {eos_token_id}, eos_token_id
    return set(eos_token_id), eos_token_id


def middle_truncate(token_ids, max_length):
    if len(token_ids) <= max_length:
        return token_ids
    head_length = max_length // 2
    tail_length = max_length - head_length
    return token_ids[:head_length] + token_ids[-tail_length:]


def encode_prompts(prompts, tokenizer, max_input_length, device):
    full_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False, verbose=False) for prompt in prompts
    ]
    token_ids = [middle_truncate(ids, max_input_length) for ids in full_token_ids]
    width = max(len(ids) for ids in token_ids)
    padded_ids = []
    attention_masks = []
    for ids in token_ids:
        padding = width - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * padding + ids)
        attention_masks.append([0] * padding + [1] * len(ids))
    encoded = {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long, device=device),
    }
    return encoded, [len(ids) for ids in full_token_ids]


def trim_after_eos(token_ids, eos_token_ids):
    for index, token_id in enumerate(token_ids):
        if token_id in eos_token_ids:
            return token_ids[: index + 1]
    return token_ids


def trim_at_stop_string(text, stop_string):
    if not stop_string or stop_string not in text:
        return text.strip(), False
    return text.split(stop_string, maxsplit=1)[0].strip(), True


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_input_length < 1 or args.max_new_tokens < 1:
        raise ValueError("batch and length arguments must be at least 1")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be at least 1")
    if args.qualitative_per_category < 0:
        raise ValueError("--qualitative_per_category cannot be negative")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{output_dir} is not empty; pass --force to overwrite evaluation files")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(args.data_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No evaluation examples were loaded")

    set_seed(args.seed)
    device = select_device(args.device)
    dtype = get_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    context_length = get_context_length(model.config)
    if context_length is not None and args.max_input_length + args.max_new_tokens > context_length:
        raise ValueError(
            f"evaluation length {args.max_input_length + args.max_new_tokens} exceeds "
            f"model context length {context_length}"
        )
    print(f"Loading {args.model} on {device} with dtype={dtype}; context={context_length}")

    eos_token_ids, generation_eos_token_id = get_eos_token_ids(model, tokenizer)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    predictions = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Generating"):
        batch = rows[start : start + args.batch_size]
        prompts = [row["prompt"] for row in batch]
        encoded, prompt_token_lengths = encode_prompts(
            prompts, tokenizer, args.max_input_length, device
        )
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=generation_eos_token_id,
                stop_strings=[args.stop_string] if args.stop_string else None,
                tokenizer=tokenizer,
                use_cache=True,
            )

        prompt_width = encoded["input_ids"].shape[1]
        completion_ids = generated[:, prompt_width:].cpu().tolist()
        for offset, (row, token_ids, prompt_tokens) in enumerate(
            zip(batch, completion_ids, prompt_token_lengths, strict=True)
        ):
            token_ids = trim_after_eos(token_ids, eos_token_ids)
            raw_prediction = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            prediction, stopped_at_boundary = trim_at_stop_string(raw_prediction, args.stop_string)
            reference = row["response"].strip()
            rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
            ended_with_eos = bool(token_ids) and token_ids[-1] in eos_token_ids
            completion_truncated = (
                len(token_ids) >= args.max_new_tokens and not ended_with_eos and not stopped_at_boundary
            )
            category = row.get("category", "")
            classification_match = None
            if category == "classification":
                classification_match = normalize_answer(prediction) == normalize_answer(reference)
            predictions.append(
                {
                    "index": start + offset,
                    "category": category,
                    "instruction": row.get("instruction", ""),
                    "context": row.get("context", ""),
                    "prompt": row["prompt"],
                    "reference": reference,
                    "raw_prediction": raw_prediction,
                    "prediction": prediction,
                    "rouge_l": rouge_l,
                    "classification_match": classification_match,
                    "prompt_tokens": prompt_tokens,
                    "input_truncated": prompt_tokens > args.max_input_length,
                    "completion_tokens": len(token_ids),
                    "scored_completion_tokens": len(
                        tokenizer.encode(prediction, add_special_tokens=False)
                    ),
                    "ended_with_eos": ended_with_eos,
                    "stopped_at_boundary": stopped_at_boundary,
                    "completion_truncated": completion_truncated,
                }
            )

    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    category_rouge = defaultdict(list)
    for prediction in predictions:
        category_rouge[prediction["category"]].append(prediction["rouge_l"])
    classification_rows = [
        prediction for prediction in predictions if prediction["classification_match"] is not None
    ]
    uses_locked_protocol = (
        args.max_input_length == EVAL_PROTOCOL["max_input_length"]
        and args.max_new_tokens == EVAL_PROTOCOL["max_new_tokens"]
        and args.stop_string == EVAL_PROTOCOL["stop_string"]
        and args.seed == EVAL_PROTOCOL["seed"]
    )
    metrics = {
        "evaluation_protocol": EVAL_PROTOCOL["name"] if uses_locked_protocol else "custom",
        "model": args.model,
        "data_file": args.data_file,
        "num_examples": len(predictions),
        "model_context_length": context_length,
        "generation": {
            "do_sample": False,
            "max_input_length": args.max_input_length,
            "max_new_tokens": args.max_new_tokens,
            "truncation_strategy": EVAL_PROTOCOL["truncation_strategy"],
            "stop_string": args.stop_string,
            "seed": args.seed,
        },
        "rouge_l": mean([prediction["rouge_l"] for prediction in predictions]),
        "rouge_l_by_category": {
            category: mean(scores) for category, scores in sorted(category_rouge.items())
        },
        "classification_accuracy": mean(
            [float(prediction["classification_match"]) for prediction in classification_rows]
        ),
        "classification_examples": len(classification_rows),
        "average_completion_tokens": mean(
            [prediction["completion_tokens"] for prediction in predictions]
        ),
        "average_scored_completion_tokens": mean(
            [prediction["scored_completion_tokens"] for prediction in predictions]
        ),
        "empty_predictions": sum(not prediction["prediction"] for prediction in predictions),
        "input_truncated_examples": sum(prediction["input_truncated"] for prediction in predictions),
        "ended_with_eos_examples": sum(prediction["ended_with_eos"] for prediction in predictions),
        "stopped_at_boundary_examples": sum(
            prediction["stopped_at_boundary"] for prediction in predictions
        ),
        "completion_truncated_examples": sum(
            prediction["completion_truncated"] for prediction in predictions
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    qualitative_samples = []
    category_counts = defaultdict(int)
    for prediction in predictions:
        category = prediction["category"]
        if category_counts[category] < args.qualitative_per_category:
            qualitative_samples.append(prediction)
            category_counts[category] += 1
    with (output_dir / "qualitative_samples.json").open("w", encoding="utf-8") as handle:
        json.dump(qualitative_samples, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Predictions saved to {prediction_path}")


if __name__ == "__main__":
    main()
