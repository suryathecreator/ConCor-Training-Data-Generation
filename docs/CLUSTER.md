# Cluster execution

The campaign runner is scheduler-agnostic Python. The included Slurm templates add atomic work claims, `afterany` continuation arrays, time-aware draining, node-local model staging, model reuse inside a unit, and architecture-specific vLLM/Triton cache archives.

## Mixed 24-way run

The production layout uses 16 A40 workers and 8 H200 workers. H200 workers claim twice as many source units; atomic claims prevent duplicate work if the pools race. Set site-specific partitions, accounts, GRES strings, model locations, and cache storage:

```bash
export HF_HOME=/shared/cache/huggingface
export QWEN_SOURCE=/shared/models/Qwen3.8-27B
export SAM3_CHECKPOINT_SOURCE=/shared/models/sam3.pt
export SAM3_REPO_ROOT=$PWD/third_party/sam3
export CPU_PARTITION=cpu
export CPU_ACCOUNT=my_cpu_account
export GPU_ACCOUNT=my_gpu_account
export A40_PARTITION=gpu-a40
export H200_PARTITION=gpu-h200

TARGET_TOTAL=20000 A40_WORKERS=16 H200_WORKERS=8 \
CAMPAIGN_ROOT=$PWD/outputs/campaigns/gpic_20k \
bash scripts/submit_hybrid_24way.sh
```

Add `EXCLUDE_CSV=/path/list.csv` to skip specific GPIC records. Set `PUBLISH_HF=1 HF_REPO_ID=owner/dataset` only when the final export job should upload externally.

## Any sufficiently large GPU partition

The two pool names are just resource profiles. You may disable either one and point the other at any partition/GRES that can hold the selected tensor-parallel setup. For example, an eight-worker one-H200 pool:

```bash
A40_WORKERS=0 H200_WORKERS=8 \
H200_PARTITION=my_gpu_partition H200_GRES_QWEN=gpu:1 H200_GRES_SAM3=gpu:1 H200_TP=1 \
bash scripts/submit_hybrid_24way.sh
```

For two-GPU 48 GB-class nodes, use the A40 profile with `A40_TP=2` and `A40_GRES_QWEN=gpu:2`. SAM 3 normally needs one GPU, so `A40_GRES_SAM3=gpu:1` is sufficient. If your Slurm installation has no accounts, leave `CPU_ACCOUNT`/`GPU_ACCOUNT` empty; the launcher omits `--account`.

## Resume and append

Rerunning the launcher against the same `CAMPAIGN_ROOT` skips durable stage checkpoints. To add more source images, increase `TARGET_TOTAL`; the manifest extension is append-only and existing image IDs/pair keys are deduplicated. Workers stop claiming new units before walltime, finish and fsync the active unit, then successor arrays resume remaining claims.

Claims are heartbeated every 60 seconds. A different allocation can recover a claim only after Slurm reports the owner inactive and the orphan grace has elapsed; a requeued instance of the exact same array task may resume immediately. Stale inspection and replacement happen under one stage lock, and fencing tokens stop an old process from committing or cleaning after ownership changes.

Before resuming from suspected storage or preemption damage, run `concor campaign-integrity`. Use `campaign-repair` to rewind only reported units, then set `START_STAGE` to the earliest repaired stage. Quarantined units require an explicit repair; adding more continuation arrays will not silently retry them.
