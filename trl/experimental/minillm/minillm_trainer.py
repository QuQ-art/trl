# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed one:\minillm_trainer_changes.md an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from transformers.cache_utils import DynamicCache
from datasets import Dataset, IterableDataset
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from transformers.utils import is_peft_available

from ...models import prepare_deepspeed
from ...trainer.grpo_trainer import GRPOTrainer, RewardFunc, RolloutFunc
from ...trainer.utils import disable_dropout_in_model, get_config_model_id
from ..utils import empty_cache
from .minillm_config import MiniLLMConfig


if is_peft_available():
    from peft import PeftConfig


def dummy_reward_func(completions: list, **kwargs):
    # placeholder reward function when no reward function is provided
    return [1.0 for _ in completions]


class MiniLLMTrainer(GRPOTrainer):
    """
    Trainer for the Knowledge Distillation of Language Models (MiniLLM) method. This algorithm was initially proposed
    in the paper [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543).

    Example:

    ```python
    from datasets import load_dataset
    from trl.experimental.minillm import MiniLLMTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = MiniLLMTrainer(
        model="Qwen/Qwen3-0.6B",
        teacher_model="Qwen/Qwen3-1.7B",
        train_dataset=dataset,
    )
    trainer.train()
    ```

    Args:
        model (`str | PreTrainedModel`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        teacher_model (`PreTrainedModel | nn.Module | str`):
            Teacher model used for knowledge distillation. Instantiated similarly to `model`.
        reward_funcs (`RewardFunc | list[RewardFunc]`, *optional*):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`experimental.minillm.MiniLLMConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Dataset | IterableDataset]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        reward_processing_classes ([`~transformers.PreTrainedTokenizerBase`] or `list[PreTrainedTokenizerBase]`, *optional*):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using
            [`~transformers.AutoTokenizer.from_pretrained`]. For elements in `reward_funcs` that are custom reward
            functions (not [`~transformers.PreTrainedModel`]), the corresponding entries in `reward_processing_classes`
            are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
        rollout_func (`RolloutFunc`, *optional*):
            Function to use for generating completions. It must take prompts, args, and processing_class as parameters
            and return a dict with `"prompt_ids"`, `"completion_ids"`, and `"logprobs"` fields. Any other fields that
            are forwarded to the reward functions. This feature is experimental and may change or be removed at any
            time without prior notice.
    """

    _tag_names = ["trl", "minillm"]
    _name = "MiniLLM"
    _paper = {
        "title": "MiniLLM: Knowledge Distillation of Large Language Models",
        "id": "2306.08543",
        # docstyle-ignore
        "citation": textwrap.dedent("""\
            @inproceedings{
                gu2024minillm,
                title={{MiniLLM: Knowledge Distillation of Large Language Models}},
                author={Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
                booktitle={The Twelfth International Conference on Learning Representations},
                year={2024},
                url={https://openreview.net/forum?id=5h0qf7IBZZ}
            }"""),
    }

    def __init__(
        self,
        model: str | PreTrainedModel,
        teacher_model: PreTrainedModel | nn.Module | str,
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: MiniLLMConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
    ):
        if reward_funcs is None:
            reward_funcs = [dummy_reward_func]

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = MiniLLMConfig(f"{model_name}-MiniLLM")

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model,
            reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            rollout_func=rollout_func,
        )

        if args.teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        elif not isinstance(teacher_model, str):
            raise ValueError(
                "You passed teacher_model_init_kwargs to the MiniLLMConfig, but your teacher_model is already instantiated."
            )
        else:
            teacher_model_init_kwargs = args.teacher_model_init_kwargs
            teacher_model_init_kwargs["dtype"] = (
                teacher_model_init_kwargs["dtype"]
                if teacher_model_init_kwargs["dtype"] in ["auto", None]
                else getattr(torch, teacher_model_init_kwargs["dtype"])
            )

        if isinstance(teacher_model, str):
            teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model, **teacher_model_init_kwargs)

        # Disable dropout in the model
        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        if self.is_deepspeed_enabled:
            self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)

        self.temperature = args.temperature
        self.kd_temperature = args.kd_temperature
        self.single_step_decomposition = args.single_step_decomposition
        self.rkl_advantage = args.rkl_advantage
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization

        # Teacher-side KV cache mode for MiniLLM experiments.
        #
        # - "none": keep the original MiniLLM behavior and compute teacher logits with use_cache=False.
        # - "gpu": compute teacher logits with KV cache enabled and keep the cache on GPU.
        # - "offload": compute teacher logits with KV cache enabled and offload the cache to CPU.
        self.teacher_kv_mode = os.environ.get("MINILLM_TEACHER_KV_MODE", "none").lower()

        if self.teacher_kv_mode not in {"none", "gpu", "offload"}:
            raise ValueError(
                "MINILLM_TEACHER_KV_MODE must be one of: none, gpu, offload. "
                f"Got: {self.teacher_kv_mode}"
            )

        self.teacher_kv_enabled = self.teacher_kv_mode in {"gpu", "offload"}
        self.teacher_kv_offload = self.teacher_kv_mode == "offload"
        self._teacher_kv_debug_printed = False

    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "batchmean",
    ):
        """
        Compute the MiniLLM loss for knowledge distillation using F.kl_div. See Eq. (1) of
        https://huggingface.co/papers/2306.08543 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """
        reg_loss = F.kl_div(
            teacher_log_probs, student_log_probs, reduction="none", log_target=True
        )  # (batch_size, sequence_length)

        # Masking
        if mask is not None:
            reg_loss = reg_loss[mask]

        # Apply reduction
        if reduction == "batchmean":
            return reg_loss.sum() / mask.sum() if mask is not None else reg_loss.sum() / reg_loss.size(0)
        elif reduction == "sum":
            return reg_loss.sum()
        elif reduction == "mean":
            return reg_loss.mean()
        else:
            return reg_loss

    def _compute_advantage(
        self,
        student_log_probs_on_labels: torch.Tensor,
        teacher_log_probs_on_labels: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute the advantage for Reverse KL Divergence.

        Mostly following [this
        implementation](https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/minillm/minillm/losses.py#L37-L49).

        $$ \text{rewards}_t = \text{teacher\_log\_probs\_on\_labels}_t - \text{student\_log\_probs\_on\_labels}_t $$

        If length normalization is enabled:

        $$ \text{lengths}_t = \sum_{i=t}^{T} \gamma^{i-t} $$

        $$ \text{advantages}_t = \frac{\sum_{i=t}^{T} \gamma^{i-t} R_i}{\text{lengths}_t} $$

        Otherwise:

        $$ \text{advantages}_t = \sum_{i=t}^{T} \gamma^{i-t} R_i $$

        Args:
            student_log_probs_on_labels: Log probabilities of the student model on the labels.
                Shape: (batch_size, sequence_length)
            teacher_log_probs_on_labels: Log probabilities of the teacher model on the labels.
                Shape: (batch_size, sequence_length)
            mask: Optional mask to apply to the log probabilities. Shape: (batch_size, sequence_length)
        Returns:
            advantage: Computed advantage. Shape: (batch_size, sequence_length)
        """
        response_length = student_log_probs_on_labels.size(1)
        if mask is None:
            mask = torch.ones_like(student_log_probs_on_labels)
        mask = mask.float()
        student_log_probs_on_labels = student_log_probs_on_labels * mask
        teacher_log_probs_on_labels = teacher_log_probs_on_labels * mask

        rewards = teacher_log_probs_on_labels - student_log_probs_on_labels  # (batch_size, sequence_length)

        if self.gamma > 0.0:
            gamma_pow = torch.pow(self.gamma, torch.arange(response_length, device=rewards.device))

            advantages = rewards * gamma_pow
            advantages = advantages.flip(1).cumsum(dim=1).flip(1)

            if self.length_normalization:
                mask = torch.where(mask < 0.5, 1e-4, mask)
                lengths = mask * gamma_pow
                lengths = lengths.flip(1).cumsum(dim=1).flip(1)
                advantages = advantages / lengths
        else:
            advantages = rewards

        return advantages

    def _get_teacher_config(self):
        if hasattr(self.teacher_model, "config"):
            return self.teacher_model.config
        if hasattr(self.teacher_model, "module") and hasattr(self.teacher_model.module, "config"):
            return self.teacher_model.module.config
        raise AttributeError("Cannot find config from teacher_model.")

    def _teacher_logits_with_kv_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: int,
    ) -> torch.Tensor:
        """
        Compute teacher logits for completion tokens using teacher-side KV cache.

        This helper is used when ``MINILLM_TEACHER_KV_MODE`` is set to either
        ``"gpu"`` or ``"offload"``:

        - ``"gpu"``: keep the KV cache on GPU with ``DynamicCache(offloading=False)``.
        - ``"offload"``: offload the KV cache to CPU with ``DynamicCache(offloading=True)``.

        It replaces the original full-sequence teacher forward with a prompt
        prefill followed by token-by-token cached forward passes.

        Returned shape:
            [batch_size, completion_length, vocab_size]
        """
        self.teacher_model.eval()

        _, seq_len = input_ids.shape
        completion_length = seq_len - prompt_lengths

        if completion_length <= 0:
            raise ValueError(
                f"Invalid completion_length={completion_length}. "
                f"seq_len={seq_len}, prompt_lengths={prompt_lengths}"
            )

        teacher_config = self._get_teacher_config()

        past_key_values = DynamicCache(
            config=teacher_config,
            offloading=self.teacher_kv_offload,
        )

        if not self._teacher_kv_debug_printed:
            print("=" * 80)
            print("[TEACHER_KV] enabled")
            print("[TEACHER_KV] mode:", self.teacher_kv_mode)
            print("[TEACHER_KV] cache class:", past_key_values.__class__.__name__)
            print("[TEACHER_KV] offloading:", getattr(past_key_values, "offloading", None))
            print("[TEACHER_KV] input_ids shape:", tuple(input_ids.shape))
            print("[TEACHER_KV] prompt_lengths:", prompt_lengths)
            print("[TEACHER_KV] completion_length:", completion_length)
            print("=" * 80)
            self._teacher_kv_debug_printed = True

        logits_for_completion = []

        # 1. Prefill prompt in one forward.
        # The last prompt position predicts the first completion token.
        prompt_input_ids = input_ids[:, :prompt_lengths]
        prompt_attention_mask = attention_mask[:, :prompt_lengths]

        outputs = self.teacher_model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

        past_key_values = outputs.past_key_values
        logits_for_completion.append(outputs.logits[:, -1:, :])

        # 2. Feed completion tokens one by one.
        # At each position, the output logits predict the next token.
        # We only need positions up to seq_len - 2.
        for pos in range(prompt_lengths, seq_len - 1):
            current_input_ids = input_ids[:, pos : pos + 1]
            current_attention_mask = attention_mask[:, : pos + 1]

            outputs = self.teacher_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values
            logits_for_completion.append(outputs.logits[:, -1:, :])

        teacher_logits = torch.cat(logits_for_completion, dim=1)

        if teacher_logits.shape[1] != completion_length:
            raise RuntimeError(
                f"teacher_logits length mismatch: got {teacher_logits.shape[1]}, "
                f"expected {completion_length}"
            )

        return teacher_logits

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
        prompt_lengths = inputs["prompt_ids"].shape[1]
        shifted_labels = input_ids[:, prompt_lengths:]

        # Compute student output.
        # Student remains unchanged: full forward, no KV cache.
        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]

        # Compute teacher output.
        self.teacher_model.eval()
        with torch.no_grad():
            if self.teacher_kv_enabled:
                teacher_logits = self._teacher_logits_with_kv_cache(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_lengths=prompt_lengths,
                )
            else:
                teacher_outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        # Compute log probabilities for student and teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs_on_labels = torch.gather(
            student_log_probs,
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)

        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs,
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)

        mask = shifted_labels != -100

        if self.rkl_advantage:
            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            )

            inputs["advantages"] = inputs["advantages"].unsqueeze(1) + reverse_kl_advantage

        # Compute GRPO loss on verifiable reward
        loss = self._compute_loss(model, inputs)

        # Compute MiniLLM single-step decomposition loss
        if self.single_step_decomposition:
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
            )

            loss += single_step_decomposition_loss

        empty_cache()

        return (loss, student_outputs) if return_outputs else loss