# von3b — adtc 2026 report

team still needs the real devpost team id pasted into metadata.json. everything else below is the actual model.

## problem

cloud coding chat is expensive and needs decent internet. a lot of students and junior devs here are on a normal 8gb laptop. von3b is a 3b coding/tool model that is supposed to run fully offline after one download. llama.cpp, no extra daemon, no api key at inference.

target user is someone writing python / small repo tasks, plus a one-line tool call when they ask the model to read files.

## design

base: `WeiboAI/VibeThinker-3B` at `77bd2cced09193c8b9a59a32bd8577bbd1f3e01c`.

parent: g100 (`ad99e0471f6e73f4fbaf96211cacdb341f4a0704`). not a from-scratch train.

stack was sft first, then rl, then a small lora at the end. the rl is the load-bearing bit.

i built the group-conditioned adaptive lopd we used in that rl. keep grpo when the group still has a real correct rollout. only route lopd onto failed rollouts that actually need a teacher. all-fail groups dont fake a positive contrast. lopd does not replace grpo.

winner on disk is the last lora on top of that: trusted-0.725 update-2. adapter sha256 `2620e2ddb659455e6bfb13a0594d34279a4aa91fd98cb88ae74f02d9606b0305`. r 16, alpha 32, dropout 0.05, q/k/v/o + gate/up/down. envelope token rows frozen on this last pass. adamw 1e-7, seq 4096, accum 8. planned 8 updates, kept update 2. ~0.17 gpu-hours on one rtx pro 6000 96gb. mixed sft before rl was ~0.49 gpu-hours on the same class of box. g100 rl was full-parameter on another 96gb blackwell. no clean total-hour ledger for the whole program yet.

evals for this submit are on a6000 + kaggle t4, greedy, max_new 8192, same banks.

quant: **q8_0** is the submit file. q4_k_m exists only as a fallback if q8 blows the 8gb ram budget. we are not starting on q4.

runtime they asked for, and what we are shipping:

- llama.cpp
- weights q8_0
- kv cache q4_0 (`-ctk q4_0 -ctv q4_0`)
- context 65536, not 64

gguf q8_0 sha256 `8854c9db34b0e4331e2258deefc5b7a9c16e40eb1f272e912540db93e5691ac5` (~3.28 gb). q4_k_m sha256 `ccf7139d811e18a777702b9211bcb86e9869e49abc4fda13de14eb944886bcc7` if we have to switch.

host: public hf `josephmayo/von3b`, file `von3b-Q8_0.gguf`. download_model.sh has the url. no creds.

## constraints

adtc box is 4 vcpu, 8gb ram, igpu, ubuntu 22.04. profiler ram budget is 7gb. no network once the gguf is on disk. exactly two public test prompts, they add two hidden ones. i kept the prompts short and coding-shaped so the model does the function, not a blog post.

africa / connectivity: one download, then offline. that is the whole point.

## benchmarks

not frozen yet. we score native bcb1140 / lcb175 / aider210 on the same banks, greedy, cleaned_native only, both arms. compact selectors already looked ok (humaneval 11/12 vs g100 9/12, mbpp 12/12 vs 11/12, product tool valid 31/32, exact 6/32, short think 32/32) but those are not the joint gate.

when the native scores land they go here. q8 laptop numbers should be about **0.5-1 pp under** the bf16/a6000 numbers. dont quote loss. dont mix harnesses.

profiler (s_acc / s_perf / s_eff, tps, peak rss, thermals): pending. we will paste the official `submission.json` numbers, not a vibe estimate.

intended llama.cpp line:

`llama-cli -m model/von3b-Q8_0.gguf -c 65536 -ctk q4_0 -ctv q4_0 -ngl 0`
