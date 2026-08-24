# VON-3B

I am **Joseph Ayanda**. This is my repository and my 3B model. I designed and ran the full path: SFT, RL, group-conditioned adaptive LoPD (my distillation), LoRA, weight edits, and the laptop GGUF pack.

VON-3B is one 3B model that covers **two jobs**: an **offline coding assistant** and an **autonomous agent**. After one public download it runs locally in llama.cpp on a standard 8 GB machine. No API key and no network at inference.

I built it to:

- write and repair code
- keep reasoning short
- emit a real one-line tool call (`<tool_call>{...}</tool_call>`) so it can act as an agent, not only a chatbot

Weights live on Hugging Face. This repository is the evaluation package.

- Model card and weights: https://huggingface.co/josephmayo/von3b
- Laptop artifact: `von3b-Q8_0.gguf` (llama.cpp, GGUF Q8_0, **3,285,475,488** bytes)
- Runtime: llama.cpp only
- Target machine: 4 vCPU, 8 GB RAM, integrated GPU, Ubuntu 22.04
- Writeup: `REPORT.md`

I also measured a matched coding check against the same `WeiboAI/VibeThinker-3B` snapshot, same EvalPlus 0.3.1 harness, same 164 HumanEval tasks, greedy `max_new=8192`. I win both numbers:

| Arm | HumanEval pass@1 | HumanEval+ pass@1 |
| --- | --- | --- |
| **VON-3B (mine)** | **0.921** (151 / 164) | **0.884** (145 / 164) |
| VibeThinker-3B base | 0.866 (142 / 164) | 0.817 (134 / 164) |

Tool probe, 32 tasks, greedy `max_new=256`, same snapshot: I emit a valid one-line `<tool_call>` with short think on **32 / 32**. The base emits **0 / 32**.

Download is one public command. Ubuntu 22.04 already has `curl`. No Hugging Face token. The script writes `model/von3b-Q8_0.gguf` and checks the exact byte size so you get the same file I published.

```bash
bash download_model.sh
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

Context length is 65,536 with Q4_0 K/V cache so the 8 GB profile can hold long coding sessions.

Required package files:

- `metadata.json`
- `download_model.sh`
- `REPORT.md`
- `model/` (GGUF downloaded by the script, not committed)
