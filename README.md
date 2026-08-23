# von3b

adtc 2026 laptop track. coding assistants.

sft, then rl with group-conditioned adaptive lopd, then a small lora at the end. weights on hf, not in git.

- weights: https://huggingface.co/josephmayo/von3b
- file: `von3b-Q8_0.gguf`
- runtime: llama.cpp only
- laptop: 4 cpu, 8gb ram, igpu, ubuntu 22.04

```bash
bash download_model.sh
```

then run it like this (this is the intended laptop setup, q8 weights + q4 kv + 64k ctx):

```bash
llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0
```

if q8 is too fat on the 8gb box we drop to q4_k_m. not before.

scores and profiler numbers get filled when they finish. bf16 evals are the ones we trust for picking the adapter. q8 should sit about 0.5-1 pp under that. dont treat loss as a win.

required files are exactly what adtc asked for: `metadata.json`, `download_model.sh`, `REPORT.md`, ignored `model/*.gguf`.
