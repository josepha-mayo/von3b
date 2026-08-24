# VON-3B

i am joseph ayanda this is my hackathon submission repo and my 3B model. i designed and ran the full path: SFT, RL, group-conditioned adaptive LoPD (rl & distillation method from my research), LoRA, weight edits, and the laptop GGUF pack.

VON-3B is one 3B model that covers **two jobs**: an **offline coding assistant** and an **autonomous agent**. after one public download it runs locally in llama.cpp or any inference engine on a standard 8 GB machine. no API key / network at inference.

we built it to:

- write and repair code
- keep reasoning short (base reasoned too long and took lots of time)
- emit a real one-line tool call (`<tool_call>{...}</tool_call>`) so it can act as an agent, not only a chatbot

other details:

- model card and weights: [https://huggingface.co/josephmayo/von3b](https://huggingface.co/josephmayo/von3b)
- laptop artifact: `von3b-Q8_0.gguf` (llama.cpp, GGUF Q8_0, **3,285,475,488** bytes)
- runtime: llama.cpp only
- target machine: 4 vCPU, 8 GB RAM, integrated GPU, Ubuntu 22.04
- writeup: `REPORT.md`

we started from `WeiboAI/VibeThinker-3B`. On a matched EvalPlus 0.3.1 HumanEval check (same 164 tasks, greedy `max_new=8192`), the model beats that snapshot:


| Arm                 | HumanEval pass@1      | HumanEval+ pass@1     |
| ------------------- | --------------------- | --------------------- |
| **VON-3B**          | **0.921** (151 / 164) | **0.884** (145 / 164) |
| VibeThinker-3B base | 0.866 (142 / 164)     | 0.817 (134 / 164)     |


we report that comparison because we were compute-constrained - thats why we didnt run more evals. tool probe, 32 tasks, greedy `max_new=256`, same snapshot: the model emits a valid one-line `<tool_call>` with short think on **32 / 32**. The base emits **0 / 32**.

download:

```bash
bash download_model.sh
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

ctx is 65,536 with Q4_0 K/V cache so the 8 GB profile can hold long coding sessions.

required package files (official template):

- `metadata.json`
- `download_model.sh`
- `REPORT.md`
- `model/` (GGUF downloaded by the script, not committed)

