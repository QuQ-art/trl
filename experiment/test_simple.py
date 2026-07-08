import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from trl.experimental.minillm import MiniLLMTrainer,MiniLLMConfig

dataset = load_dataset("/home/xxf/Distill/data-tldr", split="train")
train_dataset = dataset.select(range(1000))
# train_dataset = dataset.select(range(8000))
eval_dataset = dataset.select(range(100))


training_args = MiniLLMConfig(
    num_train_epochs=1,
    output_dir="/home/xxf/Distill/Qwen3.5-0.8B-Base-MiniLLM",
    bf16=True,
    model_init_kwargs= {"dtype": "bfloat16"},
    teacher_model_init_kwargs= {"dtype": "bfloat16"},

    eval_strategy="steps",
    eval_steps=50,
    eval_on_start=True,

    save_strategy="steps",
    save_steps=200,
    save_total_limit=10,
    logging_steps=1,

    generation_batch_size=32, # 48会oom
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    max_completion_length=128, # 大了oom

    teacher_mixin_alpha=0.2,

    report_to="tensorboard",
)


trainer = MiniLLMTrainer(
    model="/home/xxf/Distill/models/Qwen3.5-0.8B-Base",
    teacher_model="/home/xxf/Distill/models/Qwen3.5-2B",
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


print("len(dataset):", len(train_dataset))
print("per_device_train_batch_size:", trainer.args.per_device_train_batch_size)
print("gradient_accumulation_steps:", trainer.args.gradient_accumulation_steps)
print("num_train_epochs:", trainer.args.num_train_epochs)
print("max_steps:", trainer.args.max_steps)
train_dl = trainer.get_train_dataloader()
print("len(train_dataloader):", len(train_dl))
print("generation_batch_size:", getattr(trainer.args, "generation_batch_size", None))
print("steps_per_generation:", getattr(trainer.args, "steps_per_generation", None))
print("num_generations:", getattr(trainer.args, "num_generations", None))
print("teacher_mixin_alpha", getattr(trainer.args, "teacher_mixin_alpha", None))


trainer.train()
