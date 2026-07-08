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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer
from trl.trainer.grpo_trainer import GRPOTrainer

from ..testing_utils import TrlTestCase


class GenerationTestMixin:
    def prepare_inputs_for_generation(
        self, input_ids, next_sequence_length=None, past_key_values=None, attention_mask=None, **kwargs
    ):
        self.prepare_inputs_calls.append(
            {
                "next_sequence_length": next_sequence_length,
                "has_past_key_values": past_key_values is not None,
                "input_ids_length": input_ids.shape[1],
                "attention_mask_length": None if attention_mask is None else attention_mask.shape[1],
            }
        )
        if next_sequence_length is not None:
            input_ids = input_ids[:, -next_sequence_length:]

        model_inputs = {"input_ids": input_ids}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        model_inputs.update(kwargs)
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        attention_mask = model_kwargs.get("attention_mask")
        if attention_mask is None:
            position_ids = torch.arange(inputs_tensor.shape[1], dtype=torch.long, device=inputs_tensor.device)
            return position_ids.unsqueeze(0).expand(inputs_tensor.shape[0], -1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)


class ConstantLogitsModel(GenerationTestMixin, nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("step_logits", logits)
        self.prepare_inputs_calls = []

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, use_cache=False, **kwargs
    ):
        batch_size, sequence_length = input_ids.shape
        logits = self.step_logits.to(input_ids.device).expand(batch_size, sequence_length, -1).clone()
        outputs = {"logits": logits}
        if use_cache:
            outputs["past_key_values"] = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
        return SimpleNamespace(**outputs)


class LookupCacheModel(GenerationTestMixin, nn.Module):
    def __init__(self, logits_by_last_token: dict[int, torch.Tensor], vocab_size: int):
        super().__init__()
        logits_table = torch.zeros(vocab_size, vocab_size)
        for token_id, logits in logits_by_last_token.items():
            logits_table[token_id] = logits
        self.register_buffer("logits_table", logits_table)
        self.saw_past_key_values = False
        self.prepare_inputs_calls = []

    def forward(
        self, input_ids, attention_mask=None, past_key_values=None, position_ids=None, use_cache=False, **kwargs
    ):
        batch_size, sequence_length = input_ids.shape
        if past_key_values is None:
            if attention_mask is None:
                last_token_ids = input_ids[:, -1]
            else:
                positions = torch.arange(sequence_length, device=input_ids.device).expand(batch_size, -1)
                last_positions = (positions * attention_mask.long()).argmax(dim=1)
                last_token_ids = input_ids.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        else:
            self.saw_past_key_values = True
            if attention_mask is None:
                last_token_ids = input_ids[:, -1]
            else:
                last_token_ids = torch.where(attention_mask[:, -1].bool(), input_ids[:, -1], past_key_values)

        step_logits = self.logits_table.to(input_ids.device)[last_token_ids]
        logits = step_logits.unsqueeze(1).expand(batch_size, sequence_length, -1).clone()
        outputs = {"logits": logits}
        if use_cache:
            outputs["past_key_values"] = last_token_ids.clone()
        return SimpleNamespace(**outputs)


class DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")


def make_teacher_mixin_trainer(
    student_model,
    teacher_model,
    *,
    teacher_mixin_alpha=0.2,
    teacher_mixin_importance_clip=10.0,
    max_completion_length=2,
    temperature=1.0,
    kd_temperature=1.0,
    eos_token_id=2,
):
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_mixin_alpha = teacher_mixin_alpha
    trainer.teacher_mixin_importance_clip = teacher_mixin_importance_clip
    trainer.rollout_func = None
    trainer.use_vllm = False
    trainer.use_transformers_paged = False
    trainer.tools = []
    trainer.top_p = 1.0
    trainer.top_k = 0
    trainer.min_p = None
    trainer.repetition_penalty = 1.0
    trainer.args = SimpleNamespace(generation_kwargs=None)
    trainer.accelerator = DummyAccelerator()
    trainer._tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=eos_token_id)
    trainer.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
    trainer.max_completion_length = max_completion_length
    trainer.temperature = temperature
    trainer.kd_temperature = kd_temperature
    trainer.model = student_model
    trainer.model_wrapped = student_model
    trainer.teacher_model = teacher_model
    trainer._last_teacher_mixed_student_logps = None
    trainer._last_teacher_mixed_logps = None
    return trainer


def test_teacher_mixin_alpha_defaults_to_zero():
    assert MiniLLMConfig.__dataclass_fields__["teacher_mixin_alpha"].default == 0.0
    assert MiniLLMConfig.__dataclass_fields__["teacher_mixin_importance_clip"].default == 10.0


def test_minillm_trainer_rejects_external_reward_funcs():
    args = MiniLLMConfig(
        output_dir="/tmp/minillm-reward-check",
        bf16=False,
        report_to="none",
    )

    with pytest.raises(ValueError, match="pure OPD only; reward_funcs must be None"):
        MiniLLMTrainer(
            model="dummy-student",
            teacher_model="dummy-teacher",
            reward_funcs=[lambda completions, **kwargs: [0.0 for _ in completions]],
            args=args,
        )


def test_teacher_mixin_alpha_requires_matching_temperatures():
    args = MiniLLMConfig(
        output_dir="/tmp/minillm-temperature-check",
        temperature=0.8,
        kd_temperature=1.0,
        teacher_mixin_alpha=0.2,
        bf16=False,
        report_to="none",
    )

    with pytest.raises(ValueError, match="temperature == kd_temperature"):
        MiniLLMTrainer(model="dummy-student", teacher_model="dummy-teacher", args=args)


def test_teacher_mixed_generate_single_turn_alpha_zero_matches_student_probs(monkeypatch):
    captured_probs = []
    trainer = make_teacher_mixin_trainer(
        ConstantLogitsModel(torch.tensor([0.0, 8.0, -8.0])),
        ConstantLogitsModel(torch.tensor([0.0, -8.0, 8.0])),
        teacher_mixin_alpha=0.0,
        max_completion_length=1,
    )

    def fake_multinomial(probs, num_samples):
        captured_probs.append(probs.detach().clone())
        return torch.tensor([[1]], device=probs.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    trainer._teacher_mixed_generate_single_turn([[1]], None, {})

    expected_student_probs = F.log_softmax(torch.tensor([[0.0, 8.0, -8.0]]), dim=-1).exp()
    assert torch.allclose(captured_probs[0], expected_student_probs)


def test_teacher_mixed_generate_single_turn_alpha_one_matches_teacher_probs(monkeypatch):
    trainer = make_teacher_mixin_trainer(
        ConstantLogitsModel(torch.tensor([0.0, 8.0, -8.0])),
        ConstantLogitsModel(torch.tensor([0.0, -8.0, 8.0])),
        teacher_mixin_alpha=1.0,
        max_completion_length=1,
    )

    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([[2]], device=probs.device))

    completion_ids, _ = trainer._generate_single_turn([[1]], None, {})

    assert completion_ids == [[2]]
    expected_student_logp = F.log_softmax(torch.tensor([0.0, 8.0, -8.0]), dim=-1)[2]
    expected_teacher_logp = F.log_softmax(torch.tensor([0.0, -8.0, 8.0]), dim=-1)[2]
    assert torch.isclose(torch.tensor(trainer._last_teacher_mixed_student_logps[0][0]), expected_student_logp)
    assert torch.isclose(torch.tensor(trainer._last_teacher_mixed_logps[0][0]), expected_teacher_logp)


def test_teacher_mixed_generate_single_turn_mixed_probs_are_normalized(monkeypatch):
    captured_probs = []
    trainer = make_teacher_mixin_trainer(
        ConstantLogitsModel(torch.tensor([1.0, 0.5, -0.2])),
        ConstantLogitsModel(torch.tensor([-0.3, 0.2, 1.7])),
        teacher_mixin_alpha=0.25,
        max_completion_length=1,
    )

    def fake_multinomial(probs, num_samples):
        captured_probs.append(probs.detach().clone())
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    trainer._teacher_mixed_generate_single_turn([[1]], None, {})

    assert torch.allclose(captured_probs[0].sum(dim=-1), torch.ones(1))


def test_teacher_mixed_generate_single_turn_with_cache_matches_full_prefix():
    vocab_size = 5
    student_logits_by_last_token = {
        1: torch.tensor([-2.0, -2.0, 1.0, 0.5, -1.0]),
        2: torch.tensor([-2.0, -2.0, -2.0, 1.1, 0.8]),
        3: torch.tensor([-2.0, -2.0, 0.7, -2.0, 1.0]),
        4: torch.tensor([-2.0, -2.0, -2.0, -2.0, 2.0]),
    }
    teacher_logits_by_last_token = {
        1: torch.tensor([-2.0, -2.0, 0.6, 1.2, -1.0]),
        2: torch.tensor([-2.0, -2.0, -2.0, 0.4, 1.4]),
        3: torch.tensor([-2.0, -2.0, 1.1, -2.0, 0.5]),
        4: torch.tensor([-2.0, -2.0, -2.0, -2.0, 2.0]),
    }
    full_prefix_student = LookupCacheModel(student_logits_by_last_token, vocab_size=vocab_size)
    full_prefix_teacher = LookupCacheModel(teacher_logits_by_last_token, vocab_size=vocab_size)
    cache_student = LookupCacheModel(student_logits_by_last_token, vocab_size=vocab_size)
    cache_teacher = LookupCacheModel(teacher_logits_by_last_token, vocab_size=vocab_size)

    full_prefix_trainer = make_teacher_mixin_trainer(
        full_prefix_student,
        full_prefix_teacher,
        teacher_mixin_alpha=0.3,
        max_completion_length=3,
        eos_token_id=4,
    )
    cache_trainer = make_teacher_mixin_trainer(
        cache_student,
        cache_teacher,
        teacher_mixin_alpha=0.3,
        max_completion_length=3,
        eos_token_id=4,
    )

    torch.manual_seed(1234)
    full_prefix_completion_ids, _ = full_prefix_trainer._teacher_mixed_generate_single_turn_full_prefix([[1], [2]])
    full_prefix_student_logps = [list(logps) for logps in full_prefix_trainer._last_teacher_mixed_student_logps]
    full_prefix_logps = [list(logps) for logps in full_prefix_trainer._last_teacher_mixed_logps]

    torch.manual_seed(1234)
    cache_completion_ids, _ = cache_trainer._teacher_mixed_generate_single_turn([[1], [2]], None, {})
    cache_student_logps = [list(logps) for logps in cache_trainer._last_teacher_mixed_student_logps]
    cache_logps = [list(logps) for logps in cache_trainer._last_teacher_mixed_logps]

    assert full_prefix_completion_ids == cache_completion_ids
    assert cache_student.saw_past_key_values is True
    assert cache_teacher.saw_past_key_values is True
    assert cache_student.prepare_inputs_calls[0]["has_past_key_values"] is False
    assert cache_teacher.prepare_inputs_calls[0]["has_past_key_values"] is False
    assert cache_student.prepare_inputs_calls[1]["has_past_key_values"] is True
    assert cache_teacher.prepare_inputs_calls[1]["has_past_key_values"] is True
    assert cache_student.prepare_inputs_calls[0]["input_ids_length"] == cache_student.prepare_inputs_calls[0]["attention_mask_length"]
    assert cache_teacher.prepare_inputs_calls[0]["input_ids_length"] == cache_teacher.prepare_inputs_calls[0]["attention_mask_length"]
    assert cache_student.prepare_inputs_calls[1]["input_ids_length"] == cache_student.prepare_inputs_calls[1]["attention_mask_length"]
    assert cache_teacher.prepare_inputs_calls[1]["input_ids_length"] == cache_teacher.prepare_inputs_calls[1]["attention_mask_length"]
    assert cache_student.prepare_inputs_calls[1]["next_sequence_length"] == 1
    assert cache_teacher.prepare_inputs_calls[1]["next_sequence_length"] == 1
    assert len(full_prefix_student_logps) == len(cache_student_logps)
    for full_prefix_seq_logps, cache_seq_logps in zip(full_prefix_student_logps, cache_student_logps, strict=True):
        torch.testing.assert_close(torch.tensor(full_prefix_seq_logps), torch.tensor(cache_seq_logps))
    assert len(full_prefix_logps) == len(cache_logps)
    for full_prefix_seq_logps, cache_seq_logps in zip(full_prefix_logps, cache_logps, strict=True):
        torch.testing.assert_close(torch.tensor(full_prefix_seq_logps), torch.tensor(cache_seq_logps))


def test_teacher_mixed_generate_single_turn_records_eos_once_and_stops(monkeypatch):
    vocab_size = 5
    logits_by_last_token = {
        1: torch.tensor([-9.0, -9.0, -9.0, -9.0, 9.0]),
        2: torch.tensor([-9.0, -9.0, -9.0, 9.0, -9.0]),
        3: torch.tensor([-9.0, -9.0, -9.0, -9.0, 9.0]),
        4: torch.tensor([-9.0, -9.0, -9.0, -9.0, 9.0]),
    }
    trainer = make_teacher_mixin_trainer(
        LookupCacheModel(logits_by_last_token, vocab_size=vocab_size),
        LookupCacheModel(logits_by_last_token, vocab_size=vocab_size),
        teacher_mixin_alpha=0.5,
        max_completion_length=3,
        eos_token_id=4,
    )

    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: probs.argmax(dim=-1, keepdim=True))

    completion_ids, _ = trainer._teacher_mixed_generate_single_turn([[1], [2]], None, {})

    assert completion_ids == [[4], [3, 4]]
    assert [len(seq) for seq in completion_ids] == [len(seq) for seq in trainer._last_teacher_mixed_student_logps]
    assert [len(seq) for seq in completion_ids] == [len(seq) for seq in trainer._last_teacher_mixed_logps]
    assert len(trainer._last_teacher_mixed_student_logps[0]) == 1
    assert len(trainer._last_teacher_mixed_student_logps[1]) == 2
    assert len(trainer._last_teacher_mixed_logps[0]) == 1
    assert len(trainer._last_teacher_mixed_logps[1]) == 2


def test_generate_and_score_completions_uses_teacher_mixed_logps_as_behavior_logps(monkeypatch):
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_mixin_alpha = 0.2
    trainer.model = SimpleNamespace(training=True)
    trainer.accelerator = DummyAccelerator()
    trainer.pad_to_multiple_of = None
    trainer._last_teacher_mixed_student_logps = [[-0.2, -0.7]]
    trainer._last_teacher_mixed_logps = [[-0.4, -1.1]]

    def fake_generate_and_score_completions(self, inputs):
        return {"advantages": torch.tensor([0.0]), "completion_mask": torch.tensor([[1, 1]])}

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_generate_and_score_completions)

    output = trainer._generate_and_score_completions(inputs=[])

    expected_teacher_mixed_logps = torch.tensor([[-0.4, -1.1]])
    assert torch.allclose(output["teacher_mixed_logps"], expected_teacher_mixed_logps)
    assert torch.allclose(output["old_per_token_logps"], expected_teacher_mixed_logps)
    assert "teacher_mixed_importance_weights" not in output
    assert output["teacher_mixed_logps"].shape == output["completion_mask"].shape
    assert output["old_per_token_logps"].shape == output["completion_mask"].shape
    assert trainer._last_teacher_mixed_student_logps is None
    assert trainer._last_teacher_mixed_logps is None


def test_generate_single_turn_uses_student_only_rollout_during_eval(monkeypatch):
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_mixin_alpha = 0.2
    trainer.model = SimpleNamespace(training=False)
    calls = {"student": 0, "teacher_mixed": 0}

    def fake_student_generate_single_turn(self, prompt_ids, images, multimodal_fields):
        calls["student"] += 1
        return [[2]], [[-0.1]]

    def fake_teacher_mixed_generate_single_turn(self, prompt_ids, images, multimodal_fields):
        calls["teacher_mixed"] += 1
        return [[3]], None

    monkeypatch.setattr(GRPOTrainer, "_generate_single_turn", fake_student_generate_single_turn)
    monkeypatch.setattr(MiniLLMTrainer, "_teacher_mixed_generate_single_turn", fake_teacher_mixed_generate_single_turn)

    completion_ids, logps = trainer._generate_single_turn([[1]], None, {})

    assert completion_ids == [[2]]
    assert logps == [[-0.1]]
    assert calls == {"student": 1, "teacher_mixed": 0}


def test_generate_and_score_completions_ignores_teacher_mixed_logps_during_eval(monkeypatch):
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_mixin_alpha = 0.2
    trainer.model = SimpleNamespace(training=False)
    trainer._last_teacher_mixed_student_logps = [[-0.2]]
    trainer._last_teacher_mixed_logps = [[-0.4]]

    def fake_generate_and_score_completions(self, inputs):
        return {"completion_mask": torch.tensor([[1]]), "old_per_token_logps": torch.tensor([[-0.1]])}

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_generate_and_score_completions)

    output = trainer._generate_and_score_completions(inputs=[])

    assert "teacher_mixed_logps" not in output
    assert torch.allclose(output["old_per_token_logps"], torch.tensor([[-0.1]]))


def test_compute_advantage_is_future_only():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.gamma = 0.0
    trainer.length_normalization = False

    advantages = trainer._compute_advantage(
        student_log_probs_on_labels=torch.zeros((1, 3), requires_grad=True),
        teacher_log_probs_on_labels=torch.tensor([[1.0, 2.0, 3.0]]),
        mask=torch.tensor([[True, True, True]]),
    )

    assert torch.allclose(advantages, torch.tensor([[5.0, 3.0, 0.0]]))
    assert not advantages.requires_grad


def test_compute_loss_uses_completion_mask_without_weighting_single_step_kl():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, 2.0]))
    trainer.teacher_mixin_importance_clip = 2.0
    trainer.kd_temperature = 1.0
    trainer.rkl_advantage = False
    trainer.single_step_decomposition = True
    trainer.gamma = 0.0
    trainer.length_normalization = True
    captured = {}

    def fake_compute_loss(model, inputs):
        return torch.tensor(0.0)

    def fake_single_step_decomposition_loss(student_log_probs, teacher_log_probs, mask, reduction="batchmean"):
        captured["mask"] = mask.detach().clone()
        return torch.tensor(0.0)

    trainer._compute_loss = fake_compute_loss
    trainer._single_step_decomposition_loss = fake_single_step_decomposition_loss

    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 0]]),
        "completion_mask": torch.tensor([[1, 0]]),
        "advantages": torch.tensor([0.0]),
        "teacher_mixed_logps": torch.tensor([[-10.0, -0.9]]),
    }
    student_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, -2.0]))

    trainer.compute_loss(student_model, inputs)

    assert torch.equal(captured["mask"], torch.tensor([[True, False]]))


def test_compute_loss_sets_old_per_token_logps_from_teacher_mixed_logps_when_missing():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, 2.0]))
    trainer.teacher_mixin_importance_clip = 10.0
    trainer.kd_temperature = 1.0
    trainer.rkl_advantage = False
    trainer.single_step_decomposition = False
    trainer.gamma = 0.0
    trainer.length_normalization = True
    captured = {}

    def fake_compute_loss(model, inputs):
        captured["old_per_token_logps"] = inputs["old_per_token_logps"].detach().clone()
        return torch.tensor(0.0)

    trainer._compute_loss = fake_compute_loss

    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 2]]),
        "completion_mask": torch.tensor([[1, 1]]),
        "advantages": torch.tensor([0.0]),
        "teacher_mixed_logps": torch.tensor([[-0.7, -0.9]]),
    }
    student_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, -2.0]))

    trainer.compute_loss(student_model, inputs)

    assert torch.allclose(captured["old_per_token_logps"], inputs["teacher_mixed_logps"])


def test_single_step_decomposition_loss_is_not_teacher_mixed_importance_weighted():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    student_log_probs = F.log_softmax(torch.tensor([[[1.5, -0.5]]]), dim=-1)
    teacher_log_probs = F.log_softmax(torch.tensor([[[0.5, 0.5]]]), dim=-1)
    mask = torch.tensor([[True]])

    loss = trainer._single_step_decomposition_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        mask=mask,
    )

    expected = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True).sum(dim=-1)
    assert torch.allclose(loss, expected.sum() / mask.sum())


def test_compute_loss_uses_unweighted_reverse_kl_advantage_with_teacher_mixed_rollout():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, 2.0]))
    trainer.teacher_mixin_importance_clip = 10.0
    trainer.kd_temperature = 1.0
    trainer.rkl_advantage = True
    trainer.single_step_decomposition = False
    trainer.gamma = 0.0
    trainer.length_normalization = True
    captured = {}

    def fake_compute_loss(model, inputs):
        captured["advantages"] = inputs["advantages"].detach().clone()
        return torch.tensor(0.0)

    trainer._compute_loss = fake_compute_loss

    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 2]]),
        "completion_mask": torch.tensor([[1, 1]]),
        "advantages": torch.tensor([0.0]),
    }
    student_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, -2.0]))

    trainer.compute_loss(student_model, inputs)

    student_log_probs = F.log_softmax(torch.tensor([0.0, 0.0, -2.0]), dim=-1)
    teacher_log_probs = F.log_softmax(torch.tensor([0.0, 0.0, 2.0]), dim=-1)
    expected_reward = teacher_log_probs[2] - student_log_probs[2]
    assert torch.allclose(captured["advantages"], torch.tensor([[expected_reward.item(), 0.0]]))


def test_compute_loss_overwrites_input_advantages_in_pure_opd_mode():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, 2.0]))
    trainer.teacher_mixin_importance_clip = 10.0
    trainer.kd_temperature = 1.0
    trainer.rkl_advantage = True
    trainer.single_step_decomposition = False
    trainer.gamma = 0.0
    trainer.length_normalization = True
    captured = {}

    def fake_compute_loss(model, inputs):
        captured["advantages"] = inputs["advantages"].detach().clone()
        return torch.tensor(0.0)

    trainer._compute_loss = fake_compute_loss

    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 2]]),
        "completion_mask": torch.tensor([[1, 1]]),
        "advantages": torch.tensor([3.0]),
    }
    student_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, -2.0]))

    trainer.compute_loss(student_model, inputs)

    student_log_probs = F.log_softmax(torch.tensor([0.0, 0.0, -2.0]), dim=-1)
    teacher_log_probs = F.log_softmax(torch.tensor([0.0, 0.0, 2.0]), dim=-1)
    expected_reward = teacher_log_probs[2] - student_log_probs[2]
    assert torch.allclose(captured["advantages"], torch.tensor([[expected_reward.item(), 0.0]]))


def test_compute_loss_sets_zero_advantages_when_reverse_kl_is_disabled():
    trainer = MiniLLMTrainer.__new__(MiniLLMTrainer)
    trainer.teacher_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, 2.0]))
    trainer.teacher_mixin_importance_clip = 10.0
    trainer.kd_temperature = 1.0
    trainer.rkl_advantage = False
    trainer.single_step_decomposition = True
    trainer.gamma = 0.0
    trainer.length_normalization = True
    captured = {}

    def fake_compute_loss(model, inputs):
        captured["advantages"] = inputs["advantages"].detach().clone()
        return torch.tensor(0.0)

    trainer._compute_loss = fake_compute_loss
    trainer._single_step_decomposition_loss = lambda *args, **kwargs: torch.tensor(0.0)

    inputs = {
        "prompt_ids": torch.tensor([[1]]),
        "prompt_mask": torch.tensor([[1]]),
        "completion_ids": torch.tensor([[2, 2]]),
        "completion_mask": torch.tensor([[1, 1]]),
        "advantages": torch.tensor([3.0]),
    }
    student_model = ConstantLogitsModel(torch.tensor([0.0, 0.0, -2.0]))

    trainer.compute_loss(student_model, inputs)

    assert torch.equal(captured["advantages"], torch.zeros((1, 2)))


@pytest.mark.low_priority
class TestMiniLLMTrainer(TrlTestCase):
    def test_train(self):
        # Get the dataset
        dataset = Dataset.from_dict(
            {
                "prompt": [
                    "Write one short sentence about the moon.",
                    "Name a primary color.",
                    "Count from one to three.",
                    "Say hello in one word.",
                    "Name an animal that can fly.",
                    "Write one adjective for calm weather.",
                    "Name a fruit.",
                    "Write one short positive word.",
                    "Name a planet.",
                ]
            }
        )

        # Initialize the trainer
        training_args = MiniLLMConfig(
            output_dir=self.tmp_dir,
            per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
            num_generations=3,  # reduce the number of generations to reduce memory usage
            max_completion_length=32,  # reduce the completion length to reduce memory usage
            bf16=False,
            report_to="none",
        )
        processing_class = AutoTokenizer.from_pretrained(
            "trl-internal-testing/small-Qwen3ForCausalLM", local_files_only=True
        )
        processing_class.padding_side = "left"
        if processing_class.pad_token is None:
            processing_class.pad_token = processing_class.eos_token
        trainer = MiniLLMTrainer(
            model="trl-internal-testing/small-Qwen3ForCausalLM",
            teacher_model="trl-internal-testing/tiny-Qwen3ForCausalLM",
            args=training_args,
            train_dataset=dataset,
            processing_class=processing_class,
        )

        # Save the initial parameters to compare them later
        previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

        # Train the model
        trainer.train()

        # Check that the training loss is not None
        assert trainer.state.log_history[-1]["train_loss"] is not None

        # Check the params have changed
        has_changed = False
        for n, param in previous_trainable_params.items():
            new_param = trainer.model.get_parameter(n)
            if not torch.allclose(param, new_param):
                has_changed = True
                break
        assert has_changed, "No trainable parameter changed during training"
