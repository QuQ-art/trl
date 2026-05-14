# MiniLLM Teacher KV Cache Experiment Scripts

This folder contains three standalone training scripts for comparing teacher-side KV cache modes in TRL MiniLLM.

## Scripts

| Script | `MINILLM_TEACHER_KV_MODE` | Meaning |
|---|---|---|
| `train_minillm_bigmodel_no_kv_no_offload.py` | `none` | Original teacher forward: `use_cache=False`; no teacher KV cache. |
| `train_minillm_bigmodel_teacher_kv_gpu.py` | `gpu` | Teacher forward uses KV cache with `DynamicCache(offloading=False)`; KV stays on GPU. |
| `train_minillm_bigmodel_teacher_kv_offload.py` | `offload` | Teacher forward uses KV cache with `DynamicCache(offloading=True)`; KV is offloaded to CPU. |

## Default runtime

By default, each script runs a quick 20-step training-only resource check:

```bash
python scripts/train_minillm_bigmodel_no_kv_no_offload.py
python scripts/train_minillm_bigmodel_teacher_kv_gpu.py
python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

Evaluation is disabled by default to make GPU/CPU measurements faster and cleaner.

## Optional overrides

```bash
RUN_MODE=full python scripts/train_minillm_bigmodel_teacher_kv_offload.py
MAX_STEPS=100 python scripts/train_minillm_bigmodel_teacher_kv_offload.py
DO_EVAL=1 python scripts/train_minillm_bigmodel_teacher_kv_offload.py
OUTPUT_DIR=outputs/custom_dir python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

## Monitoring

Use `monitor_system_usage.sh` in a separate tmux pane:

```bash
bash scripts/monitor_system_usage.sh records/final_C_kv_offload_20/system_monitor_log.txt
```
