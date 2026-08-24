# VON-3B — Technical Report

**Team ID:** 1133460  
**Domain:** coding_assistants  
**Model:** VON-3B (GGUF Q8_0)  
**Created by:** Joseph Ayanda, 18-year-old ML engineer. Solo on this stack: SFT, RL, group-conditioned adaptive LoPD (our distillation), LoRA, and the weight edits that became the laptop GGUF.

This project is **two tracks in one model**: a coding assistant (write / repair code) and an **agent** (short think, then a real one-line `<tool_call>`). It is not a chatbot that only dumps functions.

LoPD research note (public): [`GROUP_CONDITIONED_ADAPTIVE_LOPD.md`](GROUP_CONDITIONED_ADAPTIVE_LOPD.md)

---

## Problem

Students and developers on mid-range laptops often cannot depend on a cloud coding API. Connectivity is metered or unstable, and token fees add up. The target user is a student or working programmer who needs a local assistant that can write code, repair bugs, and call tools on the machine in front of them.

VON-3B is a 3B coding assistant and autonomous agent for that setting. After one public download it runs fully offline on an 8 GB laptop through llama.cpp. Versus the same base snapshot it is stronger on HumanEval, emits short thinking, and actually produces one-line tool calls, so it can do agentic coding on the laptop instead of only chatting.

---



## Design Decisions

- **Base model:** `WeiboAI/VibeThinker-3B`. a 3B coding/agent model from the same lineage. This is a continued 3B system, not a from-scratch pretrain.
- **Training stack (Joseph Ayanda, end-to-end):** (1) supervised fine-tuning for code and the tool envelope; (2) reinforcement learning with **our** group-conditioned adaptive LoPD (distillation: GRPO stays on for groups that still have a verified correct rollout; LoPD is applied only to failed rollouts that need a privileged teacher; 50/50 bidirectional KL, frozen latent-context teacher, no EMA); (3) direct student weight updates during RL; (4) a small final LoRA pass (trusted-0.725, update 2) then merge + GGUF Q8_0. Method writeup: `GROUP_CONDITIONED_ADAPTIVE_LOPD.md`.
- **Quantization:** GGUF **Q8_0** for the laptop artifact, with Q4_0 K/V cache (`-ctk q4_0 -ctv q4_0`) and 65,536 context. Q8_0 is the only submitted weight. CPU-only llama.cpp (`-ngl 0`) — no discrete GPU at eval time.
- **What changed vs base:** the same coding bank (HumanEval / HumanEval+) went up. The 32-row tool probe went from 0 valid one-line calls on base to 32/32 on VON-3B, all with short think. Base writes long multi-line think and never emits a usable one-line `<tool_call>`. That is the agentic-coding claim: less rambling, then a real tool call, then code.
- **Agent interface:** offered tools go in the user turn. The model is trained to keep thinking short and emit one canonical one-line `<tool_call>` when a tool is required.



### Tools and why


| Tool                  | Why                                                                                          |
| --------------------- | -------------------------------------------------------------------------------------------- |
| llama.cpp             | Required runtime. Only GGUF + llama.cpp is accepted by the ADTC profiler.                    |
| GGUF Q8_0             | Fits 8 GB RAM with headroom when K/V is Q4_0; better quality than aggressive 2/3-bit quants. |
| Hugging Face (public) | Credential-free weight host for `download_model.sh`.                                         |
| PyTorch + PEFT        | Train-time only. Not used at laptop inference.                                               |




### Training compute (program, not the last LoRA step)

The last LoRA update was a short pass on one NVIDIA RTX PRO 6000 (96 GB). That is **not** the program cost. Approximate **lower bounds** across the VON-3B / G100 line (training + eval), still incomplete:


| GPU                                   | Role                                                           | Lower bound                                                                                                         |
| ------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| NVIDIA RTX PRO 6000 96 GB (Blackwell) | Full-parameter RL (G65–G100 class), later LoRA, some paid eval | **~80+ GPU-hours** (multi-day exclusive Vast boxes; last LoRA was ~0.2 h of this)                                   |
| NVIDIA RTX A6000 48 GB                | Desktop RL, sponsor leftover coding evals                      | **~40+ GPU-hours** (desktop train + sponsor A6000 jobs; this leftover wave alone is already many hours on one card) |
| NVIDIA Tesla T4 (Kaggle)              | Parallel leftover coding-eval shards                           | **~150 GPU-hours** of weekly quota consumed across the Kaggle accounts used this wave                               |


These are floors, not a closed invoice. Official quality claims are **not** taken from GPU-hours.

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

Matched greedy HumanEval on the same 164-task EvalPlus 0.3.1 bank, same base snapshot, same generation settings (`max_new=8192`, greedy). This is a development-machine coding check, not the ADTC profiler score.


| Arm    | HumanEval pass@1      | HumanEval+ pass@1     |
| ------ | --------------------- | --------------------- |
| VON-3B | **0.921** (151 / 164) | **0.884** (145 / 164) |
| base   | 0.866 (142 / 164)     | 0.817 (134 / 164)     |


Matched greedy tool probe on the official 32-task envelope bank (`max_new=256`, same base snapshot). A valid row is one short think block, then exactly one canonical one-line `<tool_call>`.


| Arm    | Valid one-line tool | Exact target | Short think |
| ------ | ------------------- | ------------ | ----------- |
| VON-3B | **32 / 32**         | 4 / 32       | **32 / 32** |
| base   | 0 / 32              | 0 / 32       | 0 / 32      |


These are **desktop** `adtc-profiler` numbers (`submission.json` on this Windows machine: 64 GB RAM, CPU-only). They are **not** the official 8 GB volunteer profile. Official `Sacc` / `Sperf` / `Seff` are measured on the standard 8 GB evaluation machine.


| Metric              | Value                                                                 |
| ------------------- | --------------------------------------------------------------------- |
| Machine             | Windows desktop, 63.8 GB RAM, CPU-only llama.cpp (`participant_laptop` flag; not 8 GB) |
| RAM at peak         | 3987.84 MB RSS (steady 3651.76 MB)                                    |
| Time to first token | 54435.52 ms                                                           |
| Generation speed    | 3.16 tok/s (512 prompt / 128 gen)                                     |
| Thermal throttling  | false (`cpu_percent_p99` 100.0; core temp unavailable)                |
| Profiler accuracy   | `arc_easy` n=50 `acc_norm` 0.30 (desktop smoke, not a coding claim)   |


```bash
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

Official `Sacc` / `Sperf` / `Seff` are measured by the ADTC profiler. We do not claim a result from training loss.