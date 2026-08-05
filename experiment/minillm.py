
"""Train MiniLLM with optional vLLM student generation and teacher scoring.

For a complete local launch example, use `examples/scripts/run_minillm_vllm.sh`.
"""

from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config, get_quantization_config
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer


@dataclass
class MiniLLMScriptArguments(ScriptArguments):
    """Arguments used only by the MiniLLM example script."""

    teacher_model_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Teacher model ID or path."},
    )


if __name__ == "__main__":
    parser = TrlParser((MiniLLMScriptArguments, MiniLLMConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    if model_args.model_name_or_path is None:
        raise ValueError("model_name_or_path must be provided.")
    if script_args.dataset_name is None:
        raise ValueError("dataset_name must be provided.")
    if script_args.teacher_model_name_or_path is None:
        raise ValueError("teacher_model_name_or_path must be provided.")

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = {
        "revision": model_args.model_revision,
        "attn_implementation": model_args.attn_implementation,
        "dtype": dtype,
    }
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    if not training_args.use_vllm_teacher:
        training_args.teacher_model_init_kwargs = {
            "revision": model_args.model_revision,
            "attn_implementation": model_args.attn_implementation,
            "dtype": model_args.dtype,
        }

    processing_class = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        padding_side="left",
        trust_remote_code=training_args.trust_remote_code,
    )
    if processing_class.pad_token is None:
        processing_class.pad_token = processing_class.eos_token

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        streaming=script_args.dataset_streaming,
    )
    train_dataset = dataset[script_args.dataset_train_split]
    if "prompt" not in train_dataset.column_names:
        raise ValueError(
            f"MiniLLM requires a 'prompt' column, but the dataset contains {train_dataset.column_names}."
        )

    eval_dataset = None
    if training_args.eval_strategy != "no":
        if script_args.dataset_test_split not in dataset:
            raise ValueError(
                f"Evaluation split {script_args.dataset_test_split!r} was not found. Available splits: {list(dataset)}"
            )
        eval_dataset = dataset[script_args.dataset_test_split]

    trainer = MiniLLMTrainer(
        model=model_args.model_name_or_path,
        teacher_model=script_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing_class,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
