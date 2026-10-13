#!/bin/bash
#SBATCH --job-name=mwhanna_0p6b_l14to22
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>
set -euo pipefail
RD=$WORK/signature
EP=$WORK/envs/rs-qwen-env/bin/python
export HF_HOME=$WORK/.cache/huggingface; export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RD/logs/m2" "$RD/notes"
echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID part=${SLURM_JOB_PARTITION:-?}"
cd "$RD/code"; export PYTHONPATH="$RD/code:${PYTHONPATH:-}"
"$EP" scripts/m2_attribution_graph.py \
    --backbone Qwen/Qwen3-0.6B-Base --sae-release mwhanna-qwen3-0.6b-transcoders-lowl0 \
    --src-layer 14 --dst-layer 22 --src-sae-id layer_14 --dst-sae-id layer_22 \
    --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
    --keep-top-n 10000 --per-state-top-k 2000 \
    --out "$RD/notes/attribution_graph_mwhanna_0p6b_l14to22.json"
echo "[DONE] $(date -Is)"
