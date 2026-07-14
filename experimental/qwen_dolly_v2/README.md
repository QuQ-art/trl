# Qwen Dolly MiniLLM pilot

This directory records a small controlled MiniLLM experiment on a fixed Dolly
pilot split. It is intended to validate the training and evaluation workflow,
not to reproduce the scale or headline scores of the MiniLLM paper.

## Setup

- Student: `Qwen/Qwen3-0.6B`, SFT on the pilot train split.
- Teacher: `Qwen/Qwen3-1.7B`, SFT on the same train split.
- Data: Dolly, shuffled with seed 42, then split into 1,000 train, 200 valid,
  and 200 test examples. MiniLLM retained 989 train prompts after length
  filtering.
- MiniLLM comparison: teacher-mixed alpha 0.0 versus 0.2. All other training
  arguments were held constant.
- Hardware/software: one NVIDIA RTX PRO 5000 72GB Blackwell GPU, PyTorch
  2.7.1+cu128, BF16.

The MiniLLM runs used 1,000 optimizer steps, learning rate 5e-6,
`constant_with_warmup` with 20 warmup steps, generation batch size 16,
effective train batch size 4, four optimization iterations per rollout,
GRPO loss, single-step decomposition, and length normalization. The random
seed was 42.

## Locked evaluation protocol

The `dolly_eval_v1` protocol uses greedy decoding, at most 256 new tokens, and
cuts a generated answer at the first `\n\n###` boundary. Mean Rouge-L F1 is
the primary metric. Checkpoints are selected only by valid Rouge-L; test is
evaluated once after selection.

## Results

| Model | Selected step | Valid Rouge-L | Test Rouge-L | Test classification accuracy | Test unfinished at 256 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFT student | best SFT eval-loss checkpoint | 0.2728 | 0.2916 | 0/33 | 42/200 |
| MiniLLM alpha=0.0 | 200 | **0.3267** | **0.3352** | 3/33 | 11/200 |
| MiniLLM alpha=0.2 | 1000 | 0.3224 | 0.3297 | 2/33 | 13/200 |

The teacher SFT baseline reached 0.3510 Rouge-L on valid, above the student
baseline as required for the distillation comparison.

Both MiniLLM variants improved test Rouge-L over the SFT student. In this
single-seed pilot, however, alpha 0.2 did not improve over alpha 0.0: its test
Rouge-L was lower by 0.0055 (0.55 Rouge-L points). This result does not show
that teacher mixing is generally harmful; it only shows that this run provides
no evidence for a positive teacher-mixing effect under this Qwen/pilot setup.
Multiple seeds and a larger, paper-aligned experiment are needed for a general
claim.

Machine-readable metrics and the complete valid checkpoint curves are in
[`results.json`](results.json).

## Reproduction entry points

- `../scripts/prepare_dolly_pilot.py`: deterministic Dolly pilot split.
- `../scripts/sft_dolly.py`: student and teacher SFT.
- `../scripts/train_minillm_dolly.py`: controlled MiniLLM training.
- `../scripts/select_minillm_checkpoint.py`: valid-set checkpoint selection.
- `../scripts/eval_dolly_generation.py`: locked generation and scoring.
- `../configs/dolly_eval_v1.json`: evaluation protocol declaration.

Training outputs, checkpoints, and per-example predictions are deliberately
excluded from Git because they are large generated artifacts.
