# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import copy

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from trl.experimental.gkd import GKDConfig, GKDTrainer

from ..testing_utils import is_bf16_supported, require_torch_accelerator


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("trl-internal-testing/small-Qwen3ForCausalLM")


@pytest.fixture
def models():
    config = AutoConfig.from_pretrained("trl-internal-testing/small-Qwen3ForCausalLM")
    config.hidden_size = 16
    config.intermediate_size = 32
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 8
    config.layer_types = ["full_attention"]
    torch.manual_seed(42)
    return [AutoModelForCausalLM.from_config(config) for _ in range(4)]


def prompt_dataset(tokenizer, teacher_ids):
    # Prompt-only input intentionally has no supervised labels before generation.
    rows = []
    for index, teacher_id in enumerate(teacher_ids):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Count to {index + 1}."}], tokenize=False, add_generation_prompt=True
        )
        rows.append(
            {
                "prompt": prompt,
                "input_ids": tokenizer.encode(prompt, add_special_tokens=False),
                "teacher_id": teacher_id,
            }
        )
    return Dataset.from_list(rows)


def config_for_test(tmp_path, **kwargs):
    return GKDConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        lmbda=1.0,
        beta=1.0,
        temperature=1.0,
        max_length=128,
        max_new_tokens=2,
        max_steps=1,
        learning_rate=1e-4,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        bf16=False,
        fp16=False,
        **kwargs,
    )


class CaptureGradients(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        self.gradients = {
            name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None
        }


@require_torch_accelerator
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
@pytest.mark.parametrize(
    "bf16", [False, pytest.param(True, marks=pytest.mark.skipif(not is_bf16_supported(), reason="requires BF16"))]
)
def test_constant_route_matches_single_teacher(tmp_path, tokenizer, models, gradient_accumulation_steps, bf16):
    student, teacher, _, _ = models
    dataset = prompt_dataset(tokenizer, [0, 0])
    results = []
    for routed in [False, True]:
        callback = CaptureGradients()
        teachers = (
            {"teacher_models": {0: copy.deepcopy(teacher)}} if routed else {"teacher_model": copy.deepcopy(teacher)}
        )
        args = config_for_test(tmp_path / str(routed), gradient_accumulation_steps=gradient_accumulation_steps)
        args.bf16 = bf16
        trainer = GKDTrainer(
            model=copy.deepcopy(student),
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            **teachers,
        )
        trainer.generation_config.do_sample = False
        trainer.generation_kwargs["do_sample"] = False
        result = trainer.train()
        assert torch.isfinite(torch.tensor(result.training_loss))
        assert result.training_loss > 0
        assert any(torch.count_nonzero(grad) for grad in callback.gradients.values())
        results.append((result.training_loss, callback.gradients, copy.deepcopy(trainer.model.cpu().state_dict())))
        trainer.teacher_model.cpu()
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-6)
    for reference, actual in zip(results[0][1:], results[1][1:], strict=True):
        for name in reference:
            torch.testing.assert_close(reference[name], actual[name], rtol=1e-5, atol=1e-7)


@require_torch_accelerator
def test_routes_freeze_and_offload_teachers(tmp_path, tokenizer, models):
    student, *teacher_list = models
    teachers = dict(enumerate(teacher_list))
    before = [{name: p.detach().clone() for name, p in teacher.named_parameters()} for teacher in teacher_list]
    dataset = prompt_dataset(tokenizer, [0, 1, 2, 0, 1, 2])
    args = config_for_test(tmp_path, gradient_accumulation_steps=2)
    args.max_steps = 3
    trainer = GKDTrainer(
        model=student, teacher_models=teachers, args=args, train_dataset=dataset, processing_class=tokenizer
    )
    calls = []

    def observe_teacher(teacher_id):
        def observe(module, inputs, kwargs):
            assert trainer.teacher_model is module
            assert trainer._active_teacher_id == teacher_id
            assert not module.training and not torch.is_grad_enabled()
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
            assert all(p.device == kwargs["input_ids"].device for p in module.parameters())
            for other_id, other in teachers.items():
                if other_id != teacher_id:
                    assert all(p.device.type == "cpu" for p in other.parameters())
            sequence = kwargs["input_ids"][0].tolist()
            matching_rows = [row for row in dataset if sequence[: len(row["input_ids"])] == row["input_ids"]]
            assert len(matching_rows) == 1
            assert matching_rows[0]["teacher_id"] == teacher_id
            calls.append(teacher_id)

        return observe

    handles = [
        teacher.register_forward_pre_hook(observe_teacher(i), with_kwargs=True) for i, teacher in teachers.items()
    ]
    result = trainer.train()
    for handle in handles:
        handle.remove()
    assert sorted(calls) == [0, 0, 1, 1, 2, 2]
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert trainer.state.global_step == 3
    optimizer_parameters = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    for i, teacher in teachers.items():
        for name, parameter in teacher.named_parameters():
            assert id(parameter) not in optimizer_parameters
            torch.testing.assert_close(parameter.cpu(), before[i][name], rtol=0, atol=0)


@require_torch_accelerator
def test_accumulation_across_teachers_averages_gradients(tmp_path, tokenizer, models):
    student, teacher_a, teacher_b, _ = models
    row = prompt_dataset(tokenizer, [0])[0]
    gradients = []
    for mode in [0, 1, "mixed"]:
        callback = CaptureGradients()
        mixed = mode == "mixed"
        dataset = Dataset.from_list([{**row, "teacher_id": i} for i in ([0, 1] if mixed else [mode])])
        kwargs = (
            {"teacher_models": {0: copy.deepcopy(teacher_a), 1: copy.deepcopy(teacher_b)}}
            if mixed
            else {"teacher_model": copy.deepcopy([teacher_a, teacher_b][mode])}
        )
        trainer = GKDTrainer(
            model=copy.deepcopy(student),
            args=config_for_test(tmp_path / str(mode), gradient_accumulation_steps=2 if mixed else 1, max_grad_norm=0),
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            **kwargs,
        )
        trainer.generation_config.do_sample = False
        trainer.generation_kwargs["do_sample"] = False
        trainer.train()
        gradients.append(callback.gradients)
        trainer.model.cpu()
        trainer.teacher_model.cpu()
    for name in gradients[2]:
        expected = (gradients[0][name] + gradients[1][name]) / 2
        torch.testing.assert_close(gradients[2][name], expected, rtol=1e-4, atol=1e-7)


@require_torch_accelerator
@pytest.mark.parametrize("teacher_id", [7, 0.5, None])
def test_invalid_routes_fail(tmp_path, tokenizer, models, teacher_id):
    student, teacher, _, _ = models
    dataset = prompt_dataset(tokenizer, [teacher_id])
    trainer = GKDTrainer(
        model=student,
        teacher_models={0: teacher},
        args=config_for_test(tmp_path),
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    with pytest.raises(ValueError, match="teacher_id"):
        trainer.train()


@pytest.mark.parametrize("overrides", [{"lmbda": 0.5}, {"seq_kd": True}, {"per_device_train_batch_size": 2}])
def test_unsupported_mopd_configuration(tmp_path, overrides):
    args = config_for_test(tmp_path)
    for name, value in overrides.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match="MOPD requires"):
        GKDTrainer(teacher_models={0: "unused"}, args=args)
