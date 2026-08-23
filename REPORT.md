# VON-3B — Technical Report

**Team ID:** REPLACE_WITH_DEVPOST_TEAM_ID  
**Domain:** coding_assistants  
**Model:** VON-3B (GGUF Q8_0)

---

## Problem

Students and developers on mid-range laptops often cannot depend on a cloud coding API. Connectivity is metered or unstable, and token fees add up. The target user is a student or working programmer who needs a local assistant that can write code, repair bugs, and call tools on the machine in front of them.

VON-3B is a 3B coding assistant and autonomous agent for that setting. After one public download it runs fully offline on an 8 GB laptop through llama.cpp.

---

## Design Decisions

- **Base model:** `WeiboAI/VibeThinker-3B`. Parent checkpoint is G100, a 3B coding/agent model from the same lineage. This is a continued 3B system, not a from-scratch pretrain.
- **Training stack:** (1) supervised fine-tuning for code and the tool envelope; (2) reinforcement learning with group-conditioned adaptive LoPD (GRPO stays on for groups that still have a verified correct rollout; LoPD is applied only to failed rollouts that need a privileged teacher); (3) a small final LoRA pass (trusted-0.725, update 2).
- **Quantization:** GGUF **Q8_0** for the laptop artifact, with Q4_0 K/V cache (`-ctk q4_0 -ctv q4_0`) and 65,536 context. Q8_0 keeps more of the 3B quality inside the 8 GB / 7 GB profiler budget. Q4_K_M was kept as a tighter-memory option; it is not the ADTC laptop default. CPU-only llama.cpp (`-ngl 0`) — no discrete GPU at eval time.
- **Agent interface:** offered tools go in the user turn. The model is trained to keep thinking short and emit one canonical one-line `<tool_call>` when a tool is required.
- **Alternatives considered:** shipping Q4_K_M as the primary artifact (smaller file, more quality loss); 64-token context (would fit easily, useless for real coding); embedding a chat UI (out of scope — judges score the llama.cpp model, not an app).

### Tools and why

| Tool | Why |
|---|---|
| llama.cpp | Required runtime. Only GGUF + llama.cpp is accepted by the ADTC profiler. |
| GGUF Q8_0 | Fits 8 GB RAM with headroom when K/V is Q4_0; better quality than aggressive 2/3-bit quants. |
| Hugging Face (public) | Credential-free weight host for `download_model.sh`. |
| PyTorch + PEFT | Train-time only. Not used at laptop inference. |

### Training compute (program, not the last LoRA step)

The last LoRA update was a short pass on one NVIDIA RTX PRO 6000 (96 GB). That is **not** the program cost. Approximate **lower bounds** across the VON-3B / G100 line (training + eval), still incomplete:

| GPU | Role | Lower bound |
|---|---|---|
| NVIDIA RTX PRO 6000 96 GB (Blackwell) | Full-parameter RL (G65–G100 class), later LoRA, some paid eval | **~80+ GPU-hours** (multi-day exclusive Vast boxes; last LoRA was ~0.2 h of this) |
| NVIDIA RTX A6000 48 GB | Desktop RL, sponsor leftover coding evals | **~40+ GPU-hours** (desktop train + sponsor A6000 jobs; this leftover wave alone is already many hours on one card) |
| NVIDIA Tesla T4 (Kaggle) | Parallel leftover BigCodeBench / LiveCodeBench / Aider shards | **~150 GPU-hours** of weekly quota consumed across the Kaggle accounts used this wave |

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

These are the **development-machine inference** numbers the template asks for. Official `Sacc` / `Sperf` / `Seff` are measured by the ADTC profiler on the standard evaluation machine.

| Metric | Value |
|---|---|
| Machine | Participant laptop, 8 GB RAM, CPU-only llama.cpp (profiler run pending) |
| RAM at peak | To be filled from `adtc-profiler` `submission.json` |
| Time to first token | To be filled from profiler |
| Generation speed | To be filled from profiler |
| Thermal throttling | To be filled from profiler |

```bash
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

Matched coding evals (BigCodeBench, LiveCodeBench, Aider) vs G100 are in progress under one harness. They will be added here only when both arms are complete. We do not claim a result from training loss.
