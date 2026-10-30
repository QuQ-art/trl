#!/usr/bin/env bash
set -Eeuo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRL_ROOT="$(cd "$EXPERIMENT_ROOT/../../.." && pwd)"
COMMAND="${1:?expected online or fixed}"
shift

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$TRL_ROOT"
export TMPDIR="$EXPERIMENT_ROOT/cache/tmp"
export HF_DATASETS_CACHE="$EXPERIMENT_ROOT/cache/datasets"
export TORCHINDUCTOR_CACHE_DIR="$EXPERIMENT_ROOT/cache/torchinductor"
export TRITON_CACHE_DIR="$EXPERIMENT_ROOT/cache/triton"

mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
    "$EXPERIMENT_ROOT/results" "$EXPERIMENT_ROOT/logs"

PYTHON=/home/zst/miniconda3/envs/minillm_vllm_qwen35/bin/python
case "$COMMAND" in
    online|fixed)
        "$PYTHON" "$EXPERIMENT_ROOT/train.py" --mode "$COMMAND" "$@" 2>&1 \
            | tee "$EXPERIMENT_ROOT/logs/${COMMAND}-$(date +%Y%m%d_%H%M%S).log"
        ;;
    *)
        echo "unknown command: $COMMAND" >&2
        exit 2
        ;;
esac
