# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import textwrap

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
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
from ...trainer.utils import disable_dropout_in_model, get_config_model_id, pad
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

        if args.teacher_mixin_alpha > 0.0 and args.temperature != args.kd_temperature:
            raise ValueError("teacher-mixed sampling requires temperature == kd_temperature.")
        # 是否必须？

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

        if args.teacher_mixin_alpha > 0.0:
            student_config = (
                self.model.config.text_config if hasattr(self.model.config, "text_config") else self.model.config
            )
            teacher_config = (
                teacher_model.config.text_config
                if hasattr(teacher_model.config, "text_config")
                else teacher_model.config
            )
            student_vocab_size = getattr(student_config, "vocab_size", None)
            teacher_vocab_size = getattr(teacher_config, "vocab_size", None)
            if (
                student_vocab_size is not None
                and teacher_vocab_size is not None
                and student_vocab_size != teacher_vocab_size
            ):
                raise ValueError("Teacher and student vocab sizes must match for teacher-mixed sampling.")

        # if self.is_deepspeed_enabled:
        #     self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        # else:
        #     self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model  = teacher_model.to("cuda:0") # 临时！！！

        self.temperature = args.temperature
        self.kd_temperature = args.kd_temperature
        self.teacher_mixin_alpha = args.teacher_mixin_alpha
        self.teacher_mixin_importance_clip = args.teacher_mixin_importance_clip
        self.single_step_decomposition = args.single_step_decomposition
        self.rkl_advantage = args.rkl_advantage
        assert self.rkl_advantage, "As for miniLLM trainer, rkl_advantage must be true"
        self.gamma = args.gamma
        self.length_normalization = args.length_normalization

        # 是否必须？
        self._last_teacher_mixed_student_logps = None
        self._last_teacher_mixed_logps = None

    def _single_step_decomposition_loss(
        self,
        student_log_probs: torch.Tensor,
        teacher_log_probs: torch.Tensor,
        mask: torch.Tensor | None = None,
        importance_weights: torch.Tensor | None = None,
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
        reg_loss = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)

        if importance_weights is not None:
            reg_loss = reg_loss * importance_weights

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
        r"""Compute the future-only advantage for Reverse KL Divergence.

        Mostly following [this
        implementation](https://github.com/microsoft/LMOps/blob/e210d2c026b9958617887762400778ace81172e6/minillm/minillm/losses.py#L37-L49).

        $$ \text{rewards}_t = \text{teacher\_log\_probs\_on\_labels}_t - \text{student\_log\_probs\_on\_labels}_t $$

        The current-token reward is handled by the single-step term, so the long-term MiniLLM advantage only uses
        future rewards.

        When `gamma > 0`, we compute the discounted future return:

        $$ \text{advantages}_t = \sum_{i=t+1}^{T} \gamma^{i-t-1} R_i $$

        Otherwise, we use the undiscounted future return:

        $$ \text{advantages}_t = \sum_{i=t+1}^{T} R_i $$

        If length normalization is enabled, the future return is divided by the corresponding discounted or
        undiscounted future length.

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

        if self.gamma > 0.0: # 需要检查为什么这么改
            advantages = torch.zeros_like(rewards)
            lengths = torch.zeros_like(rewards) if self.length_normalization else None
            next_advantage = torch.zeros(rewards.size(0), device=rewards.device, dtype=rewards.dtype)
            next_length = torch.zeros_like(next_advantage)

            for t in range(response_length - 1, -1, -1):
                advantages[:, t] = next_advantage
                next_advantage = rewards[:, t] + self.gamma * next_advantage

                if self.length_normalization:
                    lengths[:, t] = next_length
                    next_length = mask[:, t] + self.gamma * next_length

            if self.length_normalization:
                advantages = torch.where(
                    lengths > 0, advantages / lengths.clamp_min(1.0), torch.zeros_like(advantages)
                )
        else:
            advantages = rewards.flip(1).cumsum(dim=1).flip(1) - rewards

            if self.length_normalization:
                lengths = mask.flip(1).cumsum(dim=1).flip(1) - mask
                advantages = torch.where(
                    lengths > 0, advantages / lengths.clamp_min(1.0), torch.zeros_like(advantages)
                )

        return advantages * mask

    def _validate_teacher_mixed_rollout(self, images, multimodal_fields):
        if self.rollout_func is not None:
            raise ValueError("teacher-mixed rollout is incompatible with rollout_func.")
        if self.use_vllm:
            raise ValueError("teacher-mixed rollout currently requires use_vllm=False.")
        if self.use_transformers_paged:
            raise ValueError(
                "teacher-mixed rollout does not support use_transformers_paged. Use the standard transformers generation path instead."
            )
        if images is not None or multimodal_fields:
            raise ValueError("teacher-mixed rollout currently supports text-only MiniLLM rollouts.")
        if self.tools:
            raise ValueError("teacher-mixed rollout does not support tool-augmented MiniLLM rollouts.")
        if self.top_p != 1.0 or self.top_k != 0 or self.min_p is not None or self.repetition_penalty != 1.0:
            raise ValueError(
                "teacher-mixed rollout requires top_p=1.0, top_k=0, min_p=None, and repetition_penalty=1.0."
            )
        if self.args.generation_kwargs is not None:
            raise ValueError("teacher-mixed rollout does not support custom generation_kwargs.")
        if self.temperature <= 0.0:
            raise ValueError("teacher-mixed rollout requires stochastic sampling.")
        if self.temperature != self.kd_temperature:
            raise ValueError("teacher-mixed rollout requires temperature == kd_temperature.")

    def _prepare_teacher_mixed_generation_inputs(self, prompt_ids):
        device = self.accelerator.device
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("MiniLLM teacher-mixed rollout requires either a pad token or an EOS token.")

        eos_token_ids = self.generation_config.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = self._tokenizer.eos_token_id
        if eos_token_ids is None:
            eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        else:
            eos_token_ids = list(eos_token_ids)
        eos_token_ids = torch.tensor(eos_token_ids, device=device, dtype=torch.long)

        prompt_tensors = [torch.tensor(ids, dtype=torch.long) for ids in prompt_ids]
        input_ids = pad(prompt_tensors, padding_value=pad_token_id, padding_side="left").to(device=device)
        attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left").to(
            device=device
        )
        return pad_token_id, eos_token_ids, input_ids, attention_mask

    @staticmethod
    def _prepare_teacher_mixed_position_ids(model, input_ids, model_kwargs):
        if "position_ids" not in set(inspect.signature(model.forward).parameters):
            return
        model_kwargs["position_ids"] = model._prepare_position_ids_for_generation(input_ids, model_kwargs)

    def _teacher_mixed_generate_single_turn_full_prefix(self, prompt_ids):
        # Test/debug helper kept to compare against the cache rollout implementation.
        device = self.accelerator.device
        pad_token_id, eos_token_ids, input_ids, attention_mask = self._prepare_teacher_mixed_generation_inputs(
            prompt_ids
        )

        completion_ids = [[] for _ in prompt_ids]
        sampled_student_logps = [[] for _ in prompt_ids]
        sampled_mixed_logps = [[] for _ in prompt_ids]
        unfinished = torch.ones(len(prompt_ids), dtype=torch.bool, device=device)

        student_model = self.model_wrapped if self.model_wrapped is not None else self.model
        student_was_training = student_model.training
        teacher_was_training = self.teacher_model.training
        student_model.eval()
        self.teacher_model.eval()

        try:
            with torch.no_grad():
                for _ in range(self.max_completion_length):
                    student_logits = student_model(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                    ).logits[:, -1, :]
                    teacher_logits = self.teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
                    ).logits[:, -1, :]
                    if student_logits.shape[-1] != teacher_logits.shape[-1]:
                        raise ValueError("Teacher and student vocab sizes must match for teacher-mixed sampling.")

                    student_log_probs = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
                    teacher_log_probs = F.log_softmax(teacher_logits.float() / self.temperature, dim=-1)
                    student_probs = student_log_probs.exp()
                    teacher_probs = teacher_log_probs.exp()
                    mixed_probs = (
                        1.0 - self.teacher_mixin_alpha
                    ) * student_probs + self.teacher_mixin_alpha * teacher_probs
                    next_token_ids = torch.multinomial(mixed_probs, 1).squeeze(1)
                    mixed_log_probs = torch.log(mixed_probs.clamp_min(torch.finfo(mixed_probs.dtype).tiny))
                    step_student_logps = torch.gather(
                        student_log_probs, dim=-1, index=next_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    step_mixed_logps = torch.gather(
                        mixed_log_probs, dim=-1, index=next_token_ids.unsqueeze(-1)
                    ).squeeze(-1)

                    was_unfinished = unfinished.clone()
                    for idx, is_unfinished in enumerate(was_unfinished.tolist()):
                        if is_unfinished:
                            completion_ids[idx].append(int(next_token_ids[idx].item()))
                            sampled_student_logps[idx].append(float(step_student_logps[idx].item()))
                            sampled_mixed_logps[idx].append(float(step_mixed_logps[idx].item()))

                    active_token_ids = torch.where(
                        was_unfinished, next_token_ids, torch.full_like(next_token_ids, pad_token_id)
                    )
                    input_ids = torch.cat([input_ids, active_token_ids.unsqueeze(1)], dim=1)
                    attention_mask = torch.cat([attention_mask, was_unfinished.long().unsqueeze(1)], dim=1)

                    generated_eos = torch.zeros_like(was_unfinished)
                    if len(eos_token_ids) > 0:
                        generated_eos = (next_token_ids.unsqueeze(1) == eos_token_ids.unsqueeze(0)).any(dim=1)
                    unfinished = was_unfinished & ~generated_eos

                    if not unfinished.any():
                        break
        finally:
            if student_was_training:
                student_model.train()
            if teacher_was_training:
                self.teacher_model.train()

        self._last_teacher_mixed_student_logps = sampled_student_logps
        self._last_teacher_mixed_logps = sampled_mixed_logps
        return completion_ids, None

    def _teacher_mixed_generate_single_turn_with_cache(self, prompt_ids):
        device = self.accelerator.device
        pad_token_id, eos_token_ids, input_ids, attention_mask = self._prepare_teacher_mixed_generation_inputs(
            prompt_ids
        )

        completion_ids = [[] for _ in prompt_ids]
        sampled_student_logps = [[] for _ in prompt_ids]
        sampled_mixed_logps = [[] for _ in prompt_ids]
        unfinished = torch.ones(len(prompt_ids), dtype=torch.bool, device=device)

        student_model = self.model_wrapped if self.model_wrapped is not None else self.model
        teacher_model = self.teacher_model
        student_was_training = student_model.training
        teacher_was_training = teacher_model.training
        student_model.eval()
        teacher_model.eval()

        try:
            student_kwargs = {
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            teacher_kwargs = {
                "attention_mask": attention_mask.clone(),
                "use_cache": True,
            }
            self._prepare_teacher_mixed_position_ids(student_model, input_ids, student_kwargs)
            self._prepare_teacher_mixed_position_ids(teacher_model, input_ids, teacher_kwargs)

            with torch.no_grad():
                student_inputs = student_model.prepare_inputs_for_generation(
                    input_ids, is_first_iteration=True, **student_kwargs
                )
                teacher_inputs = teacher_model.prepare_inputs_for_generation(
                    input_ids, is_first_iteration=True, **teacher_kwargs
                )
                student_outputs = student_model(**student_inputs)
                # print(f"[DDDBUG] teacher dtype{next(teacher_model.parameters()).dtype}")
                print(f"[DDDBUG] input shape {teacher_inputs['input_ids'].shape}")
                teacher_outputs = teacher_model(**teacher_inputs)
                pkv = teacher_outputs.past_key_values
                print(f"[DDDBUG] teacher logits dtype: {teacher_outputs.logits.dtype}")
                if student_outputs.past_key_values is None or teacher_outputs.past_key_values is None:
                    raise ValueError(
                        "teacher-mixed rollout requires models that return past_key_values when use_cache=True."
                    )
                student_logits = student_outputs.logits[:, -1, :]
                teacher_logits = teacher_outputs.logits[:, -1, :]

                for _ in range(self.max_completion_length):
                    if student_logits.shape[-1] != teacher_logits.shape[-1]:
                        raise ValueError("Teacher and student vocab sizes must match for teacher-mixed sampling.")

                    student_log_probs = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
                    teacher_log_probs = F.log_softmax(teacher_logits.float() / self.temperature, dim=-1)
                    student_probs = student_log_probs.exp()
                    teacher_probs = teacher_log_probs.exp()
                    mixed_probs = (
                        1.0 - self.teacher_mixin_alpha
                    ) * student_probs + self.teacher_mixin_alpha * teacher_probs
                    next_token_ids = torch.multinomial(mixed_probs, 1).squeeze(1)
                    mixed_log_probs = torch.log(mixed_probs.clamp_min(torch.finfo(mixed_probs.dtype).tiny))
                    step_student_logps = torch.gather(
                        student_log_probs, dim=-1, index=next_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    step_mixed_logps = torch.gather(
                        mixed_log_probs, dim=-1, index=next_token_ids.unsqueeze(-1)
                    ).squeeze(-1)

                    was_unfinished = unfinished.clone()
                    for idx, is_unfinished in enumerate(was_unfinished.tolist()):
                        if is_unfinished:
                            completion_ids[idx].append(int(next_token_ids[idx].item()))
                            sampled_student_logps[idx].append(float(step_student_logps[idx].item()))
                            sampled_mixed_logps[idx].append(float(step_mixed_logps[idx].item()))

                    active_token_ids = torch.where(
                        was_unfinished, next_token_ids, torch.full_like(next_token_ids, pad_token_id)
                    )

                    generated_eos = torch.zeros_like(was_unfinished)
                    if len(eos_token_ids) > 0:
                        generated_eos = (next_token_ids.unsqueeze(1) == eos_token_ids.unsqueeze(0)).any(dim=1)
                    unfinished = was_unfinished & ~generated_eos

                    input_ids = torch.cat([input_ids, active_token_ids.unsqueeze(1)], dim=1)
                    attention_mask = torch.cat([attention_mask, was_unfinished.long().unsqueeze(1)], dim=1)
                    student_kwargs["past_key_values"] = student_outputs.past_key_values
                    teacher_kwargs["past_key_values"] = teacher_outputs.past_key_values
                    student_kwargs["attention_mask"] = attention_mask
                    teacher_kwargs["attention_mask"] = attention_mask.clone()
                    self._prepare_teacher_mixed_position_ids(student_model, input_ids, student_kwargs)
                    self._prepare_teacher_mixed_position_ids(teacher_model, input_ids, teacher_kwargs)

                    if not unfinished.any():
                        break

                    student_inputs = student_model.prepare_inputs_for_generation(
                        input_ids,
                        next_sequence_length=1,
                        **student_kwargs,
                    )
                    teacher_inputs = teacher_model.prepare_inputs_for_generation(
                        input_ids,
                        next_sequence_length=1,
                        **teacher_kwargs,
                    )
                    student_outputs = student_model(**student_inputs)
                    teacher_outputs = teacher_model(**teacher_inputs)
                    student_logits = student_outputs.logits[:, -1, :]
                    teacher_logits = teacher_outputs.logits[:, -1, :]
        finally:
            if student_was_training:
                student_model.train()
            if teacher_was_training:
                teacher_model.train()

        self._last_teacher_mixed_student_logps = sampled_student_logps
        self._last_teacher_mixed_logps = sampled_mixed_logps

        print(
            f"[DDDBUG] alloc {torch.cuda.memory_allocated() / 1024**3} reserved {torch.cuda.memory_reserved() / 1024**3}"
        )
        return completion_ids, None

    def _teacher_mixed_generate_single_turn(self, prompt_ids, images=None, multimodal_fields=None):
        multimodal_fields = {} if multimodal_fields is None else multimodal_fields
        self._validate_teacher_mixed_rollout(images, multimodal_fields)
        return self._teacher_mixed_generate_single_turn_with_cache(prompt_ids)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        if self.teacher_mixin_alpha == 0.0:
            return super()._generate_single_turn(prompt_ids, images, multimodal_fields)
        return self._teacher_mixed_generate_single_turn(prompt_ids, images, multimodal_fields)

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        if self.teacher_mixin_alpha == 0.0:
            return output

        teacher_mixed_student_logps = self._last_teacher_mixed_student_logps
        teacher_mixed_logps = self._last_teacher_mixed_logps
        self._last_teacher_mixed_student_logps = None
        self._last_teacher_mixed_logps = None
        if teacher_mixed_student_logps is None or teacher_mixed_logps is None:
            raise RuntimeError("MiniLLM teacher-mixed rollout did not record sampling log-probabilities.")

        device = self.accelerator.device
        teacher_mixed_student_logps = [torch.tensor(logps) for logps in teacher_mixed_student_logps]
        teacher_mixed_logps = [torch.tensor(logps) for logps in teacher_mixed_logps]
        teacher_mixed_student_logps = pad(
            teacher_mixed_student_logps,
            padding_value=0.0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        teacher_mixed_logps = pad(
            teacher_mixed_logps,
            padding_value=0.0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        teacher_mixed_importance_weights = torch.exp(teacher_mixed_student_logps - teacher_mixed_logps)
        output["teacher_mixed_logps"] = teacher_mixed_logps
        output["old_per_token_logps"] = teacher_mixed_logps
        output["teacher_mixed_importance_weights"] = teacher_mixed_importance_weights
        if output["teacher_mixed_logps"].shape != output["completion_mask"].shape:
            raise RuntimeError("teacher_mixed_logps and completion_mask must have the same shape.")
        if output["old_per_token_logps"].shape != output["completion_mask"].shape:
            raise RuntimeError("old_per_token_logps and completion_mask must have the same shape.")
        return output

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
        attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)

        # Compute student output
        student_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Compute teacher output in eval mode
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Slice the logits for the generated tokens using the inputs["prompts"] lengths
        prompt_lengths = inputs["prompt_ids"].shape[1]
        student_logits = student_outputs.logits[:, prompt_lengths - 1 : -1, :]
        teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            raise ValueError("Teacher and student vocab sizes must match for MiniLLM distillation.")
        shifted_labels = input_ids[:, prompt_lengths:]

        # Apply temperature scaling
        student_logits = student_logits / self.kd_temperature
        teacher_logits = teacher_logits / self.kd_temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        student_log_probs_on_labels = torch.gather(
            student_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)
        teacher_log_probs_on_labels = torch.gather(
            teacher_log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = inputs["completion_mask"].bool()
        if "tool_mask" in inputs:
            mask = mask & inputs["tool_mask"].bool()
        assert mask.shape == student_log_probs_on_labels.shape

        teacher_mixed_logps = inputs.get("teacher_mixed_logps")
        if teacher_mixed_logps is not None:
            assert teacher_mixed_logps.shape == mask.shape
            teacher_mixed_importance_weights = torch.exp(student_log_probs_on_labels.detach() - teacher_mixed_logps)
        else:
            teacher_mixed_importance_weights = inputs.get("teacher_mixed_importance_weights")
            if teacher_mixed_importance_weights is not None:
                assert teacher_mixed_importance_weights.shape == mask.shape
        if teacher_mixed_importance_weights is not None:
            if self.teacher_mixin_importance_clip is not None:
                teacher_mixed_importance_weights = teacher_mixed_importance_weights.clamp(
                    max=self.teacher_mixin_importance_clip
                )
            teacher_mixed_importance_weights = teacher_mixed_importance_weights * mask.float()
        
        if self.rkl_advantage:
            reverse_kl_advantage = self._compute_advantage(
                student_log_probs_on_labels=student_log_probs_on_labels,
                teacher_log_probs_on_labels=teacher_log_probs_on_labels,
                mask=mask,
            )
        
        inputs["advantages"] = reverse_kl_advantage.detach()

        if teacher_mixed_logps is not None and "old_per_token_logps" not in inputs:
            inputs["old_per_token_logps"] = teacher_mixed_logps

        # Compute the GRPO surrogate on the OPD advantages.
        loss = self._compute_loss(model, inputs)

        # Compute loss
        if self.single_step_decomposition:
            single_step_decomposition_loss = self._single_step_decomposition_loss(
                student_log_probs=student_log_probs,
                teacher_log_probs=teacher_log_probs,
                mask=mask,
                importance_weights=teacher_mixed_importance_weights,
            )

            loss += single_step_decomposition_loss

        # Empty cache
        empty_cache()

        # Return loss
        return (loss, student_outputs) if return_outputs else loss
