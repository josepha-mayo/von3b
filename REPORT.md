# VON-3B - Technical Report

**Team ID:** von-coder-3b-offline-coding-for-8-gb-laptops
**Domain:** coding_assistants
**Model:** VON-3B (GGUF Q8_0)
**Created by:** I am Joseph Ayanda. This is my model. I ran every stage: SFT, RL, group-conditioned adaptive LoPD (our distillation), LoRA, and the weight edits that became the laptop GGUF.

This project is **two tracks in one model**: a coding assistant (write / repair code) and an **agent** (short think, then a real one-line `<tool_call>`). It is not a chatbot that only dumps functions.

---

## Problem

Students and developers on mid-range laptops often cannot depend on a cloud coding API. Connectivity is metered or unstable, and token fees add up. The target user is a student or working programmer who needs a local assistant that can write code, repair bugs, and call tools on the machine in front of them.

VON-3B is a 3B coding assistant and autonomous agent for that setting. After one public download it runs fully offline on an 8 GB laptop through llama.cpp. On a matched HumanEval check the model beats the VibeThinker-3B snapshot we started from. During GRPO we trained it to prefer shorter and correct tokens, so it keeps thinking short and then emits a one-line tool call instead of only chatting.

---

## Design Decisions

- **Base model:** `WeiboAI/VibeThinker-3B`. This is a continued 3B system, not a from-scratch pretrain. We started from that snapshot. We report the matched HumanEval lift because we were compute-constrained and this is the same-harness coding check we could finish, not because it is the official profiler score or a claim that this is the best possible result.
- **Training stack (end-to-end):** (1) supervised fine-tuning for code and the tool envelope; (2) reinforcement learning with **our** group-conditioned adaptive LoPD, which is the distillation step; (3) direct student weight updates during RL; (4) a small final LoRA pass (trusted-0.725, update 2) then merge + GGUF Q8_0.
- **Group-conditioned adaptive LoPD (ours):** Standard GRPO needs at least one verified-correct rollout in the group. An all-failed group has no trustworthy positive contrast. Blind distillation on every token is also wrong, because a privileged teacher can overwrite a group that already has a real win. So we keep GRPO on whenever the group still has a verified correct rollout, and we apply LoPD only to failed rollouts that need a privileged teacher. For group `g`: `L_student = c_GRPO(g) * L_GRPO + alpha * lambda_g * L_LOPD(eligible(g))`. `c_GRPO` is 0 only if every rollout failed, else 1; it is not `(1 - lambda_g)`. Eligibility: 0 correct -> LoPD on all failures; 1 correct -> GRPO on the full group and LoPD only on the failures; 2+ correct -> GRPO only. `lambda_g` rises when reward evidence is weak and teacher-verifier agreement is strong. The teacher is not an EMA of the student. It is a frozen policy that rescores the student's actual prefixes with teacher-only latent context that is not available at laptop inference. Deadline mix is `0.5 * KL(teacher || student) + 0.5 * KL(student || teacher)` (top-64 logits plus a tail bucket). LoPD (arXiv:2608.13040) supplies the latent-context substrate; I-SDPO (arXiv:2608.12957) supplies shared group routing. We keep GRPO at coefficient 1, replace the EMA teacher, fix 50/50 bidirectional KL, and add the success-count gate. Correctness comes first; during GRPO we apply length penalties only after a response is verified correct or structurally valid, so the model is trained to prefer shorter and correct tokens.
- **Quantization:** GGUF **Q8_0** for the laptop artifact, with Q4_0 K/V cache (`-ctk q4_0 -ctv q4_0`) and 65,536 context. Q8_0 is the only submitted weight. We chose it under the 8 GB / compute constraint, not as a claim that it is the best possible quant. CPU-only llama.cpp (`-ngl 0`).
- **What changed vs the same base:** HumanEval and HumanEval+ both went up on the same EvalPlus 0.3.1 bank. The 32-row tool probe went from 0 valid one-line calls on the base to **32/32** on VON-3B, all with short think. The base writes long multi-line think and never emits a usable one-line `<tool_call>`. That is the agentic-coding claim: less rambling, then a real tool call, then code.
- **Agent interface:** offered tools go in the user turn. The model is trained to keep thinking short and emit one canonical one-line `<tool_call>` when a tool is required.

### Tools and why

| Tool | Why |
| --- | --- |
| llama.cpp | Required runtime. Only GGUF + llama.cpp is accepted by the ADTC profiler. |
| GGUF Q8_0 | Fits the 8 GB RAM profile with headroom when K/V is Q4_0. |
| Hugging Face (public) | Credential-free weight host. `bash download_model.sh` is the one-shot fetch. |
| PyTorch + PEFT | Train-time only. Not used at laptop inference. |

### Download (judges)

`bash download_model.sh` is the official fetch. It is public, needs no Hugging Face token, uses `curl` (present on Ubuntu 22.04) or `wget`, writes `model/von3b-Q8_0.gguf`, and checks the exact published size **3,285,475,488** bytes so the file on disk is the file we shipped. Idempotent: if that file is already complete, it skips.

```bash
bash download_model.sh
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

### Training compute (program, not the last LoRA step)

The last LoRA update was a short pass on one NVIDIA RTX PRO 6000 (96 GB). That is not the program cost. Approximate **lower bounds** across the VON-3B / G100 line (training + eval):

| GPU | Role | Lower bound |
| --- | --- | --- |
| NVIDIA RTX PRO 6000 96 GB (Blackwell) | Full-parameter RL (G65-G100 class), later LoRA, some paid eval | **~80+ GPU-hours** |
| NVIDIA RTX A6000 48 GB | Desktop RL, leftover coding evals | **~40+ GPU-hours** |
| NVIDIA Tesla T4 (Kaggle) | Parallel leftover coding-eval shards | **~150 GPU-hours** of weekly quota |

These are floors, not a closed invoice. We do not claim a result from training loss.

---

## Constraints

- Official profile: 4 vCPU, 8 GB RAM, integrated graphics, Ubuntu 22.04
- Profiler memory budget: 7 GB
- llama.cpp / GGUF only
- Zero outbound network after `download_model.sh`
- Exactly two public test prompts in `metadata.json`; organizers add two hidden prompts

The African constraint that matters here is connectivity and cost: one download, then offline help for coursework and local software work.

---

## Benchmarks

Matched greedy HumanEval on the same 164-task EvalPlus 0.3.1 bank, same `WeiboAI/VibeThinker-3B` snapshot we started from, same generation settings (`max_new=8192`, greedy). We report this because we were compute-constrained. It is extra same-harness coding evidence, not the ADTC profiler accuracy track. The model beats the base on both columns.

| Arm | HumanEval pass@1 | HumanEval+ pass@1 |
| --- | --- | --- |
| **VON-3B** | **0.921** (151 / 164) | **0.884** (145 / 164) |
| VibeThinker-3B base | 0.866 (142 / 164) | 0.817 (134 / 164) |

Matched greedy tool probe on the 32-task envelope bank (`max_new=256`, same snapshot). A valid row is one short think block, then exactly one canonical one-line `<tool_call>`. The model goes **32/32**. The base goes **0/32**.

| Arm | Valid one-line tool | Short think |
| --- | --- | --- |
| **VON-3B** | **32 / 32** | **32 / 32** |
| VibeThinker-3B base | 0 / 32 | 0 / 32 |

Participant `adtc-profiler` run of this same Q8_0 GGUF (`measured_on`: `participant_laptop`). These are the numbers from that run. Organizers will run the profiler themselves. We will replace this table with the volunteer 8 GB laptop run when that file exists.

| Metric | Value from `submission.json` |
| --- | --- |
| Generation speed | 3.16 tok/s (512 prompt / 128 gen) |
| Time to first token | 54435.52 ms |
| Peak RSS | 3987.84 MB |
| Steady-state RSS | 3651.76 MB |
| Thermal throttling | false |
| `cpu_percent_p99` | 100.0 |
| `arc_easy` n=50 `acc_norm` | 0.30 (profiler accuracy stage on this machine, not a coding claim) |

Devpost self-reported **Sperf = 21.07** and **Seff = 43.03** are computed from that same profiler JSON using the official formulas (`100 * 3.16 / 15` and `100 * (7.0 - peak_rss_gb) / 7.0` with 7 GB as 7000 MB). No other benches are claimed as profiler output.

```bash
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

We do not claim a result from training loss. Official `Sacc` is the organizers' accuracy track (prompts + judges).
