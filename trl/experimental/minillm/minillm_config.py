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

from dataclasses import dataclass, field
from typing import Any

from ...trainer.base_config import _BaseConfig
from ...trainer.grpo_config import GRPOConfig


@dataclass
class MiniLLMConfig(GRPOConfig):
    """
    Configuration class for [`MiniLLMTrainer`].

    This class includes only the parameters that are specific to MiniLLM training. For a full list of training
    arguments, please refer to the [`~transformers.TrainingArguments`] and [`GRPOConfig`] documentation.

    Args:
        teacher_model_init_kwargs (`dict[str, Any]`, *optional*):
            Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the teacher model
            from a string. Only used when `use_vllm_teacher=False`.
        use_vllm_teacher (`bool`, *optional*, defaults to `False`):
            Whether to use an automatically managed vLLM server for teacher scoring. `use_vllm` retains its
            [`~trl.GRPOConfig`] meaning and controls only student completion generation.
        teacher_vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.2`):
            GPU memory utilization for the teacher vLLM engine automatically managed by the trainer when
            `use_vllm_teacher=True`.
        teacher_vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout in seconds for starting the automatically managed teacher vLLM engine.
        disable_dropout (`bool`, *optional*, defaults to `True`):
            Whether to disable dropout in the model.
        rkl_advantage (`bool`, *optional*, defaults to `True`):
            Whether to add the reverse KL advantage to the reward advantage.
        single_step_decomposition (`bool`, *optional*, defaults to `True`):
            Whether to use single-step decomposition for the KL divergence computation. Automatically disabled when
            `use_vllm_teacher=True`, because vLLM returns sampled-token log probabilities rather than a dense
            vocabulary distribution.
        kd_temperature (`float`, *optional*, defaults to `1.0`):
            Temperature for knowledge distillation. Higher temperatures produce softer probability distributions over
            classes.
        gamma (`float`, *optional*, defaults to `0.0`):
            Discount factor for future rewards in reinforcement learning.
        length_normalization (`bool`, *optional*, defaults to `True`):
            Whether to apply length normalization to the rewards.
    """

    _VALID_DICT_FIELDS = GRPOConfig._VALID_DICT_FIELDS + ["teacher_model_init_kwargs"]

    teacher_model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments to pass to `AutoModelForCausalLM.from_pretrained` when instantiating the "
            "teacher model from a string."
        },
    )
    use_vllm_teacher: bool = field(
        default=False,
        metadata={"help": "Whether to use an automatically managed vLLM server for teacher scoring."},
    )
    teacher_vllm_gpu_memory_utilization: float = field(
        default=0.2,
        metadata={"help": "GPU memory utilization for the automatically managed teacher vLLM engine."},
    )
    teacher_vllm_server_timeout: float = field(
        default=240.0,
        metadata={"help": "Total timeout in seconds for starting the automatically managed teacher vLLM engine."},
    )
    vllm_enable_sleep_mode: bool = field(
        default=True,
        metadata={
            "help": "Enable sleep mode for the colocated student vLLM engine. MiniLLM enables it by default to "
            "leave memory for the training model and teacher vLLM engine."
        },
    )
    vllm_max_model_length: int | None = field(
        default=8192,
        metadata={"help": "Context window used by both the student and teacher vLLM engines."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropouts in `model`."},
    )
    rkl_advantage: bool = field(
        default=True,
        metadata={"help": "Whether to add the reverse KL advantage to the reward advantage."},
    )
    single_step_decomposition: bool = field(
        default=True,
        metadata={"help": "Whether to use single-step decomposition for the KL divergence computation."},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for knowledge distillation. Higher temperatures produce softer probability "
            "distributions over classes."
        },
    )
    gamma: float = field(
        default=0.0,
        metadata={"help": "Discount factor for future rewards in reinforcement learning."},
    )
    length_normalization: bool = field(
        default=True,
        metadata={"help": "Whether to apply length normalization to the rewards."},
    )

    def __post_init__(self):
        # We do not use the post_init of GRPOConfig because:
        # 1. num_generations can be < 2 in MiniLLMConfig. Scale_rewards must be set to "none" to avoid nan.
        _BaseConfig.__post_init__(self)

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)
        if self.num_generations == 1:
            self.scale_rewards = "none"

        if self.use_vllm_teacher:
            # vLLM returns the teacher probability of the sampled token, not the full-vocabulary distribution needed
            # by the single-step decomposition. The reverse-KL advantage remains exact.
            self.single_step_decomposition = False
            if not self.rkl_advantage:
                raise ValueError(
                    "use_vllm_teacher=True requires rkl_advantage=True because the teacher vLLM engine returns "
                    "sampled-token log probabilities for the reverse-KL advantage."
                )
            if self.teacher_model_init_kwargs is not None:
                raise ValueError(
                    "teacher_model_init_kwargs cannot be used with use_vllm_teacher=True because the teacher is "
                    "loaded by the automatically managed vLLM engine."
                )
        if not 0 < self.teacher_vllm_gpu_memory_utilization <= 1:
            raise ValueError("teacher_vllm_gpu_memory_utilization must be in the range (0, 1].")
        if self.teacher_vllm_server_timeout <= 0:
            raise ValueError("teacher_vllm_server_timeout must be positive.")

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")
