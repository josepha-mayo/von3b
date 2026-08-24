# VON-3B

**Created by Joseph Ayanda**, 18-year-old ML engineer. He designed and ran the full path: SFT, RL, group-conditioned adaptive LoPD (our distillation), LoRA, weight edits, and the laptop GGUF pack.

VON-3B is one 3B model that covers **two jobs**: an **offline coding assistant** and an **autonomous agent**. After one public download it runs locally in llama.cpp on a standard 8 GB machine. No API key and no network at inference.

It is built for:

- writing and repairing code
- short, bounded reasoning
- one-line tool use (`<tool_call>{...}</tool_call>`) so it can act as an agent, not only a chatbot

Weights are hosted on Hugging Face. This repository is the evaluation package, not the weight dump.

- Model card and weights: https://huggingface.co/josephmayo/von3b
- Laptop artifact for this track: `von3b-Q8_0.gguf` (llama.cpp, GGUF Q8_0)
- Runtime: llama.cpp only
- Target machine: 4 vCPU, 8 GB RAM, integrated GPU, Ubuntu 22.04
- Writeup: `REPORT.md` (problem, design decisions, constraints, laptop benchmarks)
- LoPD research (our method): `GROUP_CONDITIONED_ADAPTIVE_LOPD.md`
- Versus the same base: better HumanEval, shorter think, and real one-line tool calls (agentic coding). EvalPlus 0.3.1, 164 tasks: VON-3B **0.921** HumanEval / **0.884** HumanEval+ vs base **0.866** / **0.817**
- Tool probe (32 tasks): VON-3B 32/32 valid one-line tool calls vs base 0/32

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
