# VON-3B — Technical Report

**Track:** coding assistants  
**Model:** VON-3B (GGUF Q8_0)  
**Runtime:** llama.cpp  
**Target:** 8 GB laptop, fully offline inference

## Problem

Reliable coding help is often locked behind cloud APIs, metered tokens, and a stable connection. Students and developers working from constrained networks still need a local assistant that can write code, fix bugs, and take tool actions on their machine.

VON-3B is a 3B coding assistant and autonomous agent for that setting. After a single public download it runs on a commodity 8 GB laptop through llama.cpp. Inference makes no network calls.

## Design

**Base model.** `WeiboAI/VibeThinker-3B` revision `77bd2cced09193c8b9a59a32bd8577bbd1f3e01c`.

**Parent checkpoint.** G100 (`ad99e0471f6e73f4fbaf96211cacdb341f4a0704`), a 3B coding/agent checkpoint from the same lineage. This submission is not a from-scratch pretrain.

**Training stack.**

1. Supervised fine-tuning for coding and the tool envelope.
2. Reinforcement learning with **group-conditioned adaptive LoPD**, a routing rule I designed: GRPO stays on whenever a group still contains a verified correct rollout; LoPD is applied only to failed rollouts that need a privileged teacher; all-fail groups do not invent a positive contrast. LoPD does not replace GRPO.
3. A final LoRA adaptation (trusted-0.725 update 2) for a small trusted-Python pass.

**Final adapter.** SHA-256 `2620e2ddb659455e6bfb13a0594d34279a4aa91fd98cb88ae74f02d9606b0305`. LoRA rank 16, alpha 32, dropout 0.05, targeting `q/k/v/o` and `gate/up/down`. Envelope token rows were frozen on this last pass. AdamW 1e-7, sequence length 4096, gradient accumulation 8. Winner checkpoint is update 2. That pass used about 0.17 GPU-hours on one NVIDIA RTX PRO 6000 (96 GB). Earlier mixed SFT on the same class of GPU was about 0.49 GPU-hours. G100 RL was full-parameter on a second 96 GB Blackwell GPU. A complete program-wide GPU-hour ledger will be added with the final results.

**Agent interface.** Offered tools are placed in the user turn. The model is trained to keep thinking short and emit a single canonical one-line `<tool_call>` when a tool is required.

**Laptop runtime (this repository).**

| Setting | Value |
|---|---|
| Weights | GGUF Q8_0 (`von3b-Q8_0.gguf`) |
| SHA-256 | `8854c9db34b0e4331e2258deefc5b7a9c16e40eb1f272e912540db93e5691ac5` |
| Size | ~3.28 GB |
| Context | 65,536 |
| K/V cache | Q4_0 (`-ctk q4_0 -ctv q4_0`) |
| Offload | CPU (`-ngl 0`) |
| Host | public Hugging Face `josephmayo/von3b` |

`download_model.sh` fetches that GGUF with no credentials. BF16 and additional GGUF quantizations live on the Hugging Face repo for research use; the ADTC laptop artifact is Q8_0.

## Constraints

- Official profile: 4 vCPU, 8 GB RAM, integrated graphics, Ubuntu 22.04
- Profiler memory budget: 7 GB
- llama.cpp / GGUF only
- Zero outbound network after download
- Exactly two public test prompts in `metadata.json`; organizers add two hidden prompts

The African context is connectivity and cost: one download, then offline help for coursework and local software work.

## Benchmarks

Official coding and profiler numbers will be written here when the matched native evaluations finish. We do not report training loss as a quality claim.

Compact selector evidence used only to pick the adapter (not the joint laptop score):

| Probe | VON-3B | G100 parent |
|---|---:|---:|
| HumanEval (12) | 11 | 9 |
| MBPP (12) | 12 | 11 |
| Product-tool valid / exact / short-think (32) | 31 / 6 / 32 | — |

Matched BigCodeBench, LiveCodeBench, and Aider scores, plus ADTC profiler `Sacc` / `Sperf` / `Seff`, will replace this section when both arms are complete under the same harness.

```bash
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```
