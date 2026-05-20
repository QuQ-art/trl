from datasets import load_dataset
from trl.experimental.minillm import MiniLLMTrainer

dataset = load_dataset("~/Distill/data-tldr", split="train")

trainer = MiniLLMTrainer(
    model="~/Distill/models/Qwen3.5-0.8B-Base",
    teacher_model="~/Distill/models/Qwen3.5-2B",
    train_dataset=dataset,
)
trainer.train()