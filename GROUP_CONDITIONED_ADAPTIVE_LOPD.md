# Group-conditioned adaptive LoPD

Research method used in VON-3B reinforcement learning. Written by **Joseph Ayanda** (18), who designed the routing rule, implemented the trainer path, and ran the SFT -> RL -> LoRA stack.

This is our method, not a copy of one paper. It uses Latent On-Policy Distillation as the dense-supervision substrate, then changes the teacher, the mix, and the group router.

## Why we needed our own LoPD

Standard GRPO needs a mixed group: at least one verified-correct rollout so the policy can contrast winners against losers. When a whole group fails, there is no trustworthy positive contrast. Blind distillation on every token is also wrong: a privileged teacher can overwrite a group that already has a real win.

We therefore **keep GRPO on whenever the group still has a verified correct rollout**, and we **apply LoPD only to failed rollouts that need a privileged teacher**.

## Objective

For group `g`:

```text
L_student = c_GRPO(g) * L_GRPO + alpha * lambda_g * L_LOPD(eligible(g))
```

- `c_GRPO(g)` is `0` only if every rollout failed, else `1`.
- `c_GRPO` is **not** `(1 - lambda_g)`. LoPD must not shrink a valid GRPO contrast.
- LoPD eligibility:
  - 0 correct: LoPD on all failed rollouts
  - 1 correct: GRPO on the full group, LoPD only on the failed rollouts
  - 2+ correct: GRPO only

`lambda_g` in `[0, 1]` is a group-level weight. It rises when reward evidence is weak and teacher-verifier agreement is strong. Poor teacher agreement or strong reward contrast drives `lambda_g` toward zero.

## Distillation (this is the distillation step)

The teacher is **not** an EMA of the student. It is a frozen policy that rescores the student's actual visited prefixes with teacher-only latent context (retrieved / composed experience that is **not** available at laptop inference).

Deadline mix is equal bidirectional KL:

```text
L_BiKL = 0.5 * KL(teacher || student) + 0.5 * KL(student || teacher)
```

Teacher support is top-64 logits plus a tail bucket. Privileged-margin checks matter: without them a composer can collapse to context that makes the teacher no more informative than the student.

## What we kept vs what we changed

| Source | Kept | Changed |
| --- | --- | --- |
| LoPD (arXiv:2608.13040) | Retrieve experience, teacher-only latent context, rescore student prefixes, top-M + tail | Group-conditioned eligibility; no claim of their full learned QFormer unless we actually train it |
| I-SDPO (arXiv:2608.12957) | Shared group routing; low teacher entropy as a confidence weight `exp(-beta * H_teacher)` | Keep GRPO at coefficient 1; replace EMA teacher with frozen latent-context teacher; fix 50/50 BiKL; success-count gate is ours |
| AutoResearch | One change at a time, fixed budget, keep/discard from matched evidence | Panel is coding + tool/agent correctness, not a single val scalar |

## Weight path this project actually ran

1. Supervised fine-tuning for code and the one-line `<tool_call>` envelope
2. Reinforcement learning (GRPO / RLVR) with this group-conditioned adaptive LoPD
3. Direct weight updates on the 3B student during that RL phase
4. A small trusted LoRA (r=16, alpha=32) as the last adapter
5. Laptop export: merge + GGUF Q8_0 (inference-only; no trainer on the 8 GB machine)

## Concise reasoning

Correctness and valid tool structure come first. Length / latency penalties apply only after a response is verified correct or structurally valid. Short think is a reward **inside the correct set**, not a reward for failing fast. That is why VON-3B can stay brief and still emit a real tool call.

## What we do not claim from this note

This note is the method. Scores that we publish are only the matched HumanEval / HumanEval+ and the 32-row tool probe in `REPORT.md`. Training loss, unlike-harness numbers, and unpublished banks are not evidence of a win.

## References

- LoPD: https://arxiv.org/pdf/2608.13040
- I-SDPO: https://arxiv.org/pdf/2608.12957
- AutoResearch: https://github.com/karpathy/autoresearch
