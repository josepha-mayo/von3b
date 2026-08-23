# VON-3B

VON-3B is a 3B-parameter **offline coding assistant and autonomous agent** for the ADTC 2026 laptop track. After one public download it runs locally in llama.cpp on a standard 8 GB machine. No API key and no network at inference.

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
