# GKD + MOPD Baseline

基于本仓库修改后的 `trl.experimental.gkd.GKDTrainer` 的单卡多教师 On-Policy Distillation 实验。按样本的 `teacher_id` 选择 Math、Code 或 IF Teacher，Student 使用 Transformers `generate()` 生成回复；当前未使用 vLLM。

## Setup

- **Models**：Student 为 MixSFT，Teachers 为 Math / Code / IF。每步仅将当前 Teacher 加载到 GPU，其余保留在 CPU；Teacher 冻结。
- **Data**：`coverage.json` 中的 6 条样本（每个 domain 2 条），按 Math → Code → IF 循环。该数据仅用于训练闭环与性能测试。
- **Training**：单卡 BF16 全参数，batch size 1，gradient accumulation 1，关闭 gradient checkpointing。
- **Loss**：GKD，`lmbda=1`、`beta=1`（Reverse KL）、`seq_kd=False`、`temperature=1`。
- **Optimizer**：`BF16StochasticAdamW`，LR `1e-6`，weight decay `0.01`，FP32 moments、BF16 随机舍入；max grad norm `1.0`。
- **Generation**：prompt 最长 512 tokens，completion 最长 1024 tokens，`top_p=1`、`top_k=0`，seed 42。

优化器实现保存在 `bf16_stochastic_adamw.py`，与本地 Open-MOPD 共享 rollout 权重路径所用实现的源码一致。运行时记录完整配置、版本和优化器源码哈希。

## Run

从 TRL 仓库根目录运行：

```bash
# 6-step online training
bash trl/experiment/gkd_mopd_baseline/run.sh online --steps 6

# 30-step fixed / online benchmark (first 6 steps as warmup)
bash trl/experiment/gkd_mopd_baseline/run.sh fixed --steps 30 --warmup 6
bash trl/experiment/gkd_mopd_baseline/run.sh online --steps 30 --warmup 6

# Checkpoint reload verification
bash trl/experiment/gkd_mopd_baseline/run.sh reload
```

脚本默认使用实验服务器上的模型、coverage 和固定 rollout 路径。迁移到其他目录或机器时，可通过 `--model-root`、`--coverage`、`--fixed-rollouts` 和 `--output-root` 指定路径；GPU 可用 `CUDA_VISIBLE_DEVICES` 设置。默认结果目录为 `trl/experiment/gkd_mopd_baseline/results/`。

每次运行生成 `config.json`、`steps.json`、`summary.json` 和 Student checkpoint；`reload` 将冷加载检查写入 `reload.json`。

## Results

2026-10-31，在 NVIDIA RTX PRO 5000 72GB 上测得。30-step benchmark 排除前 6 步 warmup，统计后 24 步。

| Mode | Wall time | Generation | End-to-end | Peak process memory |
| --- | ---: | ---: | ---: | ---: |
| Online, 6 steps | 84.78 s | 72.98 tokens/s | 50.93 tokens/s | 51.15 GiB |
| Fixed rollout, 30/6 | 102.92 s | — | 167.59 tokens/s | 51.74 GiB |
| Online, 30/6 | 385.29 s | 62.48 tokens/s | 43.83 tokens/s | 51.84 GiB |

6-step 与 30-step 在线训练均完成 Student 更新和三 Teacher 轮转；Teacher 参数保持不变，checkpoint 经独立 CPU 进程冷加载验证。固定 rollout 模式不包含生成耗时。

**Notes**：以上测试重复使用 6 条覆盖样本，仅用于基本正确性和性能测量；不能代表完整训练集的收敛结果。与 Unsloth 的 FP32 数值对照通过，BF16 严格数值一致性门限仍未通过。