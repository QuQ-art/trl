"""Run the local GKD + MOPD baseline with three offloaded teachers."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import threading
import time

import torch
from datasets import Dataset
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl.experimental.gkd import GKDConfig, GKDTrainer

from bf16_stochastic_adamw import BF16StochasticAdamW


EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRL_ROOT = EXPERIMENT_ROOT.parents[2]
WORKSPACE_ROOT = TRL_ROOT.parent
DEFAULT_MODEL_ROOT = Path("/home/xxf/Distill/models/OPD")
DEFAULT_COVERAGE = WORKSPACE_ROOT / "gkd-vLLM-MOPD/data/coverage.json"
DEFAULT_FIXED_ROLLOUTS = WORKSPACE_ROOT / "gkd-vLLM-MOPD/output/20261027_144730_mopd_3778386/steps.json"
TEACHER_NAMES = ("Math", "Code", "IF")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("online", "fixed"), default="online")
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
parser.add_argument("--fixed-rollouts", type=Path, default=DEFAULT_FIXED_ROLLOUTS)
parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT / "results")
cli = parser.parse_args()
if cli.steps <= 0 or not 0 <= cli.warmup < cli.steps:
    parser.error("require steps > 0 and 0 <= warmup < steps")


torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
set_seed(42)

run = cli.output_root / f"gkd-{cli.mode}-bf16-{cli.steps}-{time.strftime('%Y%m%d_%H%M%S')}"
run.mkdir(parents=True)
shutil.copy2(__file__, run / "train.py")
shutil.copy2(EXPERIMENT_ROOT / "bf16_stochastic_adamw.py", run / "bf16_stochastic_adamw.py")
print("RUN_DIR", run, flush=True)


def write_json(name, value):
    (run / name).write_text(json.dumps(value, indent=2, default=str) + "\n")


def clock():
    torch.cuda.synchronize()
    return time.perf_counter()


coverage_file = cli.coverage.resolve()
coverage = json.loads(coverage_file.read_text())["samples"]
for index, row in enumerate(coverage):
    if row["teacher_id"] != index % len(TEACHER_NAMES):
        raise ValueError("coverage rows must route Math, Code, IF in order")
    if not 0 < len(row["input_ids"]) <= 512:
        raise ValueError("coverage prompt length must be in [1, 512]")
    row["sample_id"] = index
dataset = Dataset.from_list(coverage)

fixed_rollouts = None
if cli.mode == "fixed":
    fixed_rollouts = json.loads(cli.fixed_rollouts.resolve().read_text())
    for index, row in enumerate(coverage):
        if fixed_rollouts[index]["input_ids"][: len(row["input_ids"])] != row["input_ids"]:
            raise ValueError(f"fixed rollout {index} does not start with its coverage prompt")


def load_model(name, trainable):
    path = cli.model_root / name
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True
    ).to(dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model.config.use_cache = False
    if not trainable:
        model.requires_grad_(False).eval().to("cpu")
    return model, tokenizer


student, tokenizer = load_model("MixSFT", trainable=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
teacher_models = {}
for teacher_id, name in enumerate(TEACHER_NAMES):
    teacher_models[teacher_id], teacher_tokenizer = load_model(name, trainable=False)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError(f"{name} tokenizer vocabulary differs from MixSFT")


def collate(rows):
    batch = tokenizer.pad({"input_ids": [row["input_ids"] for row in rows]}, return_tensors="pt")
    return {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
        "labels": torch.full_like(batch.input_ids, -100),
        "prompts": batch.input_ids,
        "prompt_attention_mask": batch.attention_mask,
        "teacher_id": torch.tensor([row["teacher_id"] for row in rows], dtype=torch.long),
        "sample_id": torch.tensor([row["sample_id"] for row in rows], dtype=torch.long),
    }


class OrderedTrainer(GKDTrainer):
    def _get_train_sampler(self, *args, **kwargs):
        return torch.utils.data.SequentialSampler(self.train_dataset)


class Monitor(TrainerCallback):
    def __init__(self):
        self.rows = []
        self.current = None
        self.process_peak = 0
        self.process_samples = []
        self.done = threading.Event()

    def add(self, key, value):
        if self.current is not None:
            self.current[key] = self.current.get(key, 0) + value

    def start_process_monitor(self):
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        def poll():
            while not self.done.wait(0.1):
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                    if process.pid == os.getpid():
                        self.process_peak = max(self.process_peak, process.usedGpuMemory)
                        self.process_samples.append([time.perf_counter(), process.usedGpuMemory])

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        self.process_peak = 0
        self.current = {
            "step": state.global_step + 1,
            "rollout": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
            "h2d": 0.0,
            "d2h": 0.0,
            "teacher_loads": 0,
            "teacher_switches": 0,
        }
        self.started = clock()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.parameter_probes = {
            name: parameter.detach().flatten()[::4096].clone() for name, parameter in student.named_parameters()
        }
        self.optimizer_started = clock()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.add("optimizer", clock() - self.optimizer_started)
        self.current["changed_parameter_probes"] = sum(
            not torch.equal(parameter.detach().flatten()[::4096], self.parameter_probes[name])
            for name, parameter in student.named_parameters()
        )
        del self.parameter_probes

    def on_step_end(self, args, state, control, **kwargs):
        self.current.update(
            wall=clock() - self.started,
            allocated_peak=torch.cuda.max_memory_allocated(),
            reserved_peak=torch.cuda.max_memory_reserved(),
            process_peak=self.process_peak,
        )
        self.rows.append(self.current)
        self.current = None

    def on_log(self, args, state, control, logs, **kwargs):
        if self.rows and "grad_norm" in logs:
            self.rows[-1]["grad_norm"] = float(logs["grad_norm"])
        write_json("steps.json", self.rows)


monitor = Monitor()
training_args = GKDConfig(
    output_dir=str(run / "checkpoint"),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    max_steps=cli.steps,
    max_length=1536,
    max_new_tokens=1024,
    learning_rate=1e-6,
    weight_decay=0.01,
    max_grad_norm=1.0,
    lr_scheduler_type="constant",
    warmup_steps=0,
    lmbda=1.0,
    beta=1.0,
    seq_kd=False,
    temperature=1.0,
    seed=42,
    bf16=True,
    fp16=False,
    gradient_checkpointing=False,
    logging_steps=1,
    logging_nan_inf_filter=False,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)
trainer = OrderedTrainer(
    model=student,
    teacher_models=teacher_models,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
    data_collator=collate,
    callbacks=[monitor],
)
student.to(dtype=torch.bfloat16)
student.gradient_checkpointing_disable()
if student.is_gradient_checkpointing:
    raise RuntimeError("gradient checkpointing must remain disabled")
trainer.model_accepts_loss_kwargs = False
trainer.optimizer = BF16StochasticAdamW(
    student.parameters(), lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False
)
trainer.generation_config.use_cache = True
trainer.generation_config.do_sample = True
trainer.generation_config.temperature = 1.0
trainer.generation_config.top_k = 0
trainer.generation_config.top_p = 1.0
trainer.generation_config.pad_token_id = tokenizer.pad_token_id
trainer.generation_config.eos_token_id = tokenizer.eos_token_id
trainer.generation_kwargs.update(use_cache=True, do_sample=True, temperature=1.0, top_k=0, top_p=1.0)

optimizer_file = EXPERIMENT_ROOT / "bf16_stochastic_adamw.py"
config = {
    "backend": "gkd",
    "mode": cli.mode,
    "precision": "bf16",
    "steps": cli.steps,
    "warmup": cli.warmup,
    "model_root": str(cli.model_root.resolve()),
    "coverage": str(coverage_file),
    "coverage_sha256": hashlib.sha256(coverage_file.read_bytes()).hexdigest(),
    "fixed_rollouts": str(cli.fixed_rollouts.resolve()) if cli.mode == "fixed" else None,
    "student": "MixSFT",
    "teachers": list(TEACHER_NAMES),
    "teacher_order": [row["teacher_id"] for row in coverage],
    "versions": {
        package: importlib.metadata.version(package) for package in ("torch", "transformers", "trl", "accelerate")
    },
    "trainer_file": inspect.getsourcefile(GKDTrainer),
    "generation_backend": "Transformers generate",
    "fast_inference": False,
    "vllm_engine_present": "vllm_engine" in student.__dict__,
    "gradient_checkpointing": student.is_gradient_checkpointing,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    "parameter_dtypes": sorted({str(parameter.dtype) for parameter in student.parameters()}),
    "accelerator_mixed_precision": trainer.accelerator.mixed_precision,
    "effective_optimizer": {
        "class": "BF16StochasticAdamW",
        "source": str(optimizer_file),
        "source_sha256": hashlib.sha256(optimizer_file.read_bytes()).hexdigest(),
        "lr": 1e-6,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "moment_dtype": "torch.float32",
        "parameter_group": "all student parameters, including norms",
    },
    "generation_config": trainer.generation_config.to_dict(),
    "training_config": json.loads(training_args.to_json_string()),
}
write_json("config.json", config)
if config["vllm_engine_present"]:
    raise RuntimeError("this baseline expects Transformers generate, not a vLLM engine")


teachers = trainer.teacher_models
teacher_calls = {teacher_id: 0 for teacher_id in teachers}
transfers = []
for teacher_id, teacher in teachers.items():
    original_to = teacher.to

    def measured_to(*args, _original=original_to, _teacher_id=teacher_id, **kwargs):
        previous_device = next(teachers[_teacher_id].parameters()).device.type
        started = clock()
        output = _original(*args, **kwargs)
        elapsed = clock() - started
        current_device = next(teachers[_teacher_id].parameters()).device.type
        if previous_device != current_device:
            direction = "h2d" if current_device == "cuda" else "d2h"
            monitor.add(direction, elapsed)
            monitor.add("teacher_loads" if direction == "h2d" else "teacher_switches", 1)
            transfers.append({"teacher_id": _teacher_id, "direction": direction, "seconds": elapsed})
        return output

    teacher.to = measured_to

    def observe_teacher(module, args, kwargs, _teacher_id=teacher_id):
        if trainer._active_teacher_id != _teacher_id or trainer.teacher_model is not module:
            raise RuntimeError("the active teacher does not match teacher_id")
        if module.training or torch.is_grad_enabled():
            raise RuntimeError("teachers must run in eval/no-grad mode")
        for candidate_id, candidate in teachers.items():
            expected_device = "cuda" if candidate_id == _teacher_id else "cpu"
            if any(parameter.device.type != expected_device for parameter in candidate.parameters()):
                raise RuntimeError("exactly one teacher must reside on CUDA")
        teacher_calls[_teacher_id] += 1

    teacher.register_forward_pre_hook(observe_teacher, with_kwargs=True)


original_compute_loss = trainer.compute_loss


def measured_compute_loss(model, inputs, *args, **kwargs):
    started = clock()
    output = original_compute_loss(model, inputs, *args, **kwargs)
    monitor.add("forward", clock() - started)
    loss = output[0] if isinstance(output, tuple) else output
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite loss")
    if monitor.current is not None:
        monitor.current.update(
            loss=loss.item(),
            teacher_id=int(inputs["teacher_id"].item()),
            sample_id=int(inputs["sample_id"].item()),
            tokens=int((inputs["labels"][:, 1:] != -100).sum()),
        )
    return output


trainer.compute_loss = measured_compute_loss
original_backward = trainer.accelerator.backward


def measured_backward(*args, **kwargs):
    started = clock()
    output = original_backward(*args, **kwargs)
    monitor.add("backward", clock() - started)
    return output


trainer.accelerator.backward = measured_backward
original_generate = trainer.generate_on_policy_outputs


def generate_for_mode(model, inputs, generation_config):
    if cli.mode == "online":
        started = clock()
        output = original_generate(model, inputs, generation_config)
        monitor.add("rollout", clock() - started)
        return output

    sample_id = int(inputs["sample_id"].item())
    sequences = torch.tensor([fixed_rollouts[sample_id]["input_ids"]], device=inputs["prompts"].device)
    prompt_length = inputs["prompts"].shape[1]
    completion_ids = sequences[:, prompt_length:]
    eos_ids = torch.tensor(generation_config.eos_token_id, device=sequences.device).reshape(-1)
    is_eos = torch.isin(completion_ids, eos_ids)
    completion_mask = (is_eos.cumsum(-1) - is_eos.long() == 0).long()
    attention_mask = torch.cat([inputs["prompt_attention_mask"], completion_mask], dim=1)
    labels = sequences.clone()
    labels[:, :prompt_length] = -100
    labels[attention_mask == 0] = -100
    return sequences, attention_mask, labels


trainer.generate_on_policy_outputs = generate_for_mode
student_before = {name: parameter.detach().cpu().clone() for name, parameter in student.named_parameters()}
teacher_versions = {
    teacher_id: [parameter._version for parameter in teacher.parameters()] for teacher_id, teacher in teachers.items()
}

monitor.start_process_monitor()
started = clock()
train_result = trainer.train()
train_wall = clock() - started
monitor.done.set()
monitor.thread.join()
write_json("process_memory.json", monitor.process_samples)

for teacher_id, teacher in teachers.items():
    if teacher_versions[teacher_id] != [parameter._version for parameter in teacher.parameters()]:
        raise RuntimeError(f"teacher {teacher_id} weights changed")
optimizer = trainer.optimizer.optimizer
if not all(
    state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32 for state in optimizer.state.values()
):
    raise RuntimeError("optimizer moments must remain FP32")
changed_parameters = [
    name
    for name, parameter in student.named_parameters()
    if not torch.equal(parameter.detach().cpu(), student_before[name])
]
if not changed_parameters:
    raise RuntimeError("no student parameter changed")

write_json("steps.json", monitor.rows)
trainer.save_model()
trainer.save_state()
saved_parameters = {}
for path in (run / "checkpoint").glob("*.safetensors"):
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        saved_parameters.update({name: checkpoint.get_tensor(name) for name in checkpoint.keys()})
checkpoint_exact_match = all(
    torch.equal(parameter.detach().cpu(), saved_parameters[name]) for name, parameter in student.named_parameters()
)
if not checkpoint_exact_match:
    raise RuntimeError("serialized checkpoint differs from the in-memory student")

measured_rows = monitor.rows[cli.warmup :]
totals = {
    key: sum(row[key] for row in measured_rows)
    for key in ("rollout", "forward", "backward", "optimizer", "h2d", "d2h", "wall", "tokens")
}
summary = {
    "train_loss": train_result.training_loss,
    "train_wall": train_wall,
    "steps": cli.steps,
    "warmup": cli.warmup,
    "measured": totals,
    "rollout_tokens_per_second": totals["tokens"] / totals["rollout"] if totals["rollout"] else None,
    "e2e_tokens_per_second": totals["tokens"] / totals["wall"],
    "peak_allocated": max(row["allocated_peak"] for row in monitor.rows),
    "peak_reserved": max(row["reserved_peak"] for row in monitor.rows),
    "peak_process": max(row["process_peak"] for row in monitor.rows),
    "teacher_calls": teacher_calls,
    "transfers": transfers,
    "changed_parameters": changed_parameters,
    "checkpoint_exact_match": checkpoint_exact_match,
    "optimizer_parameter_states": len(optimizer.state),
}
write_json("summary.json", summary)
print("COMPLETED", run, flush=True)
