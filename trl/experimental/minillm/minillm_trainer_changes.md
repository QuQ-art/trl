# MiniLLMTrainer Teacher KV Cache 修改说明

## 1. 文件说明

基于原始 `minillm_trainer.py` 修改，用于实验 teacher 计算阶段的 KV cache 行为。

原始 TRL MiniLLMTrainer 中，teacher forward 使用：

```python
teacher_outputs = self.teacher_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=False,
)
```

因此原始实现中 teacher 计算阶段不会产生 KV cache。

本次修改新增了 teacher KV cache 的三种模式，用于做对照实验。

---

## 2. 新增运行模式

通过环境变量控制：

```bash
export MINILLM_TEACHER_KV_MODE=none
export MINILLM_TEACHER_KV_MODE=gpu
export MINILLM_TEACHER_KV_MODE=offload
```

| 模式 | 含义 |
|---|---|
| `none` | 保持原始 MiniLLM 行为，teacher forward 使用 `use_cache=False` |
| `gpu` | teacher forward 使用 `use_cache=True`，KV cache 保留在 GPU |
| `offload` | teacher forward 使用 `use_cache=True`，KV cache offload 到 CPU |

默认值是：

```bash
MINILLM_TEACHER_KV_MODE=none
```

也就是不设置环境变量时，行为尽量接近原始 TRL MiniLLMTrainer。

---

## 3. 主要代码改动

### 3.1 新增 import

新增：

```python
import os
from transformers.cache_utils import DynamicCache
```

用途：

- `os`：读取环境变量 `MINILLM_TEACHER_KV_MODE`
- `DynamicCache`：构造 teacher-side KV cache，并控制是否 offload

---

### 3.2 新增 teacher KV mode 开关

在 `MiniLLMTrainer.__init__()` 中新增：

```python
self.teacher_kv_mode = os.environ.get("MINILLM_TEACHER_KV_MODE", "none").lower()

if self.teacher_kv_mode not in {"none", "gpu", "offload"}:
    raise ValueError(...)

self.teacher_kv_enabled = self.teacher_kv_mode in {"gpu", "offload"}
self.teacher_kv_offload = self.teacher_kv_mode == "offload"
self._teacher_kv_debug_printed = False
```

这部分用于控制 teacher forward 是否启用 KV cache，以及 KV cache 是否 offload。

---

### 3.3 新增 `_get_teacher_config()`

新增 helper：

```python
def _get_teacher_config(self):
    ...
```

用途：从 `self.teacher_model` 或 `self.teacher_model.module` 中获取 teacher config，用于初始化 `DynamicCache`。

---

### 3.4 新增 `_teacher_logits_with_kv_cache()`

新增 helper：

```python
def _teacher_logits_with_kv_cache(...):
    ...
```

该函数用于在 `gpu` / `offload` 模式下计算 teacher logits。

核心逻辑：

1. 对 prompt 做一次 prefill forward；
2. 用 `past_key_values` 保存 KV cache；
3. 对 completion token 做 token-by-token cached forward；
4. 收集每个 completion token 对应的 teacher logits；
5. 返回形状为 `[batch_size, completion_length, vocab_size]` 的 logits。

其中：

```python
past_key_values = DynamicCache(
    config=teacher_config,
    offloading=self.teacher_kv_offload,
)
```

- `self.teacher_kv_offload=False`：KV cache 留在 GPU；
- `self.teacher_kv_offload=True`：KV cache offload 到 CPU。

---

### 3.5 修改 `compute_loss()`

原始 teacher forward：

```python
teacher_outputs = self.teacher_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    use_cache=False,
)
teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
```

修改后：

```python
if self.teacher_kv_enabled:
    teacher_logits = self._teacher_logits_with_kv_cache(...)
else:
    teacher_outputs = self.teacher_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    teacher_logits = teacher_outputs.logits[:, prompt_lengths - 1 : -1, :]
```

也就是：

- `none`：仍走原始 full-sequence teacher forward；
- `gpu` / `offload`：走新增的 teacher KV cache 计算路径。

---

## 4. 已清理内容

相比实验过程中的版本，本次整理版做了这些清理：

1. 删除旧环境变量 `MINILLM_TEACHER_KV_OFFLOAD` 的兼容逻辑，避免和新模式混淆；
2. 删除冗余 debug flag `_teacher_kv_offload_debug_printed`；
3. 将函数名从 `_teacher_logits_with_offloaded_kv_cache` 改成 `_teacher_logits_with_kv_cache`；
4. 更新 docstring，明确该函数同时支持 `gpu` 和 `offload` 两种模式；
5. 删除 `compute_loss()` 中未使用的 `labels` 变量；
6. 保留首次调用时的 debug print，便于日志确认实际模式。

---

## 5. 运行示例

### A. 原始 no KV cache baseline

```bash
unset MINILLM_TEACHER_KV_MODE

python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

或显式指定：

```bash
export MINILLM_TEACHER_KV_MODE=none
python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

预期日志：没有 `[TEACHER_KV] enabled`。

---

### B. Teacher KV cache on GPU

```bash
export MINILLM_TEACHER_KV_MODE=gpu
python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

预期日志：

```text
[TEACHER_KV] enabled
[TEACHER_KV] mode: gpu
[TEACHER_KV] offloading: False
```

---

### C. Teacher KV cache offload to CPU

```bash
export MINILLM_TEACHER_KV_MODE=offload
python scripts/train_minillm_bigmodel_teacher_kv_offload.py
```

预期日志：

```text
[TEACHER_KV] enabled
[TEACHER_KV] mode: offload
[TEACHER_KV] offloading: True
```

---

## 6. 实验结论相关说明

该修改主要用于验证 teacher 计算阶段 KV cache 的三种路径：

1. 不产生 teacher KV cache；
2. 产生 teacher KV cache 并保留在 GPU；
3. 产生 teacher KV cache 并 offload 到 CPU。

需要注意：原始 TRL MiniLLMTrainer 的 teacher forward 使用 `use_cache=False`，因此原始 baseline 本身不会产生 teacher KV cache。  
所以 `offload` 模式不是简单地从原始 baseline 中“减掉一块 KV cache 显存”，而是先引入 cached teacher forward，再将 KV cache offload 到 CPU。
