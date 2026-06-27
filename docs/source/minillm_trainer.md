# MiniLLM Trainer

[![All_models-MiniLLM-blue](https://img.shields.io/badge/All_models-MiniLLM-blue)](https://huggingface.co/models?other=minillm,trl)

## Overview

TRL supports the MiniLLM Trainer for distilling large language models into smaller ones using reverse KLD for better precision, quality, and performance, as described in the paper [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543) by [Yuxian Gu](https://huggingface.co/t1101675), [Li Dong](https://huggingface.co/unilm), [Furu Wei](https://huggingface.co/thegenerality), and Minlie Huang.
The abstract from the paper is the following:

> Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge from white-box generative LLMs is still under-explored, which becomes more and more important with the prosperity of LLMs. In this work, we propose MiniLLM that distills smaller language models from generative larger language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. Extensive experiments in the instruction-following setting show that the MiniLLM models generate more precise responses with the higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance. Our method is also scalable for different model families with 120M to 13B parameters. We will release our code and model checkpoints at https://aka.ms/MiniLLM.

This post-training method was contributed by [Yuxian Gu](https://huggingface.co/t1101675).

It is a generalized version of [Think Machine Lab's On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/), with the option to add distribution-level single-step distillation signals (like GKD when `beta=1`) and long-context reverse KLD signals.

$$
\begin{align}
L_{\text{MiniLLM}}&=\alpha_1\mathbb{E}_{x\sim \pi_{\theta}}\sum_{t'=t}^{|x|}\frac{\gamma^{t'-t}}{\sum_{t'}\gamma^{t'-t}}\left[\log \frac{\pi_{\theta}(x_{t'+1}|x_{1..t'})}{\pi_{\text{teacher}}(x_{t'+1}|x_{1..t'})}\right] \\
&+ \alpha_2\mathbb{E}_{x\sim \pi_{\theta}} \text{KL}\left[\pi_\theta(\cdot|x_{1..t})||\pi_{\text{teacher}}(\cdot | x_{1..t})\right].
\end{align}
$$

`MiniLLMTrainer` runs in pure OPD mode: external `reward_funcs` are not supported, and the GRPO reward path is only
kept as an internal placeholder while the training signal comes from MiniLLM reverse-KL terms.

By default, the trainer keeps the original TRL pure-student rollout behavior with `teacher_mixin_alpha=0.0`. To use
teacher-mixed rollouts, set `teacher_mixin_alpha=0.2` so tokens are sampled from
`teacher_mixin_alpha * p_teacher + (1 - teacher_mixin_alpha) * q_student`. For teacher-mixed rollouts, the behavior
log-probabilities are used as `old_per_token_logps`, so the long-term policy-gradient surrogate applies the
`q_student / p_mixed` correction. The single-step KL term remains the exact vocabulary-level
`KL(q_student || p_teacher)` term and is not scaled by sampled-token importance weights, matching the public MiniLLM
implementation. The deprecated `teacher_mixin_importance_clip` argument is kept for compatibility and does not affect
the single-step KL term.

Teacher-mixed rollouts require the teacher and student distributions to share the same token-id space. Use models with
matching tokenizers when enabling `teacher_mixin_alpha > 0.0`.

When  \\( \alpha_1=1 \\), \\( \alpha_2=0 \\), \\( \gamma=0 \\), which corresponds to

```python
from trl.experimental.minillm import MiniLLMConfig

training_args = MiniLLMConfig(
    teacher_mixin_alpha=0.2,
    rkl_advantage=True,
    single_step_decomposition=False,
    gamma=False
)
```

\\( L_{\text{MiniLLM}} \\) becomes the on-policy KD implemented in [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook/blob/5d08be6d130596b7bedd02197861c41fa81ea436/tinker_cookbook/distillation/train_on_policy.py#L88):

$$
L_{\text{tinker}}=\mathbb{E}_{x\sim \pi_{\theta}}\left[\log \frac{\pi_{\theta}(x_{t'+1}|x_{1..t'})}{\pi_{\text{teacher}}(x_{t'+1}|x_{1..t'})}\right].
$$

When \\( \alpha_1=0 \\), \\( \alpha_2=1 \\), which corresponds to

```python
from trl.experimental.minillm import MiniLLMConfig

training_args = MiniLLMConfig(
    teacher_mixin_alpha=0.2,
    rkl_advantage=False,
    single_step_decomposition=True
)
```

\\( L_{\text{MiniLLM}} \\) becomes the reverse KLD version of the GKD loss as in [GKD Trainer](./gkd.md):

$$
L_{\text{GKD-RKL}}=\mathbb{E}_{x\sim \pi_{\theta}} \text{KL}\left[\pi_\theta(\cdot|x_{1..t})||\pi_{\text{teacher}}(\cdot | x_{1..t})\right].
$$

## MiniLLMTrainer

[[autodoc]] experimental.minillm.MiniLLMTrainer
    - train
    - save_model
    - push_to_hub

## MiniLLMConfig

[[autodoc]] experimental.minillm.MiniLLMConfig
