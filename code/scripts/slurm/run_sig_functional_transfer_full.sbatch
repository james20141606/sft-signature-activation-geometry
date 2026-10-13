#!/bin/bash
#SBATCH --job-name=sig-ft-full
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1 --time=01:30:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>
# Registered full functional-transfer sweep (full). Design:
# PREREGISTRATION_functional_transfer.md + AMENDMENT 1 + notes/PREREG_functional_transfer_full_20260924.md
# (committed before this job). smoke = pipeline check on 64 prompts, never read as a result;
# it also writes the tokenization-only held-out count. full = configs A/B/C x k in {2000,5000,10000}.
set -euo pipefail
RD=$WORK/signature
PY=$WORK/envs/rs-qwen-env/bin/python
AUD=$WORK/grpo_project/checkpoints/sft-qwen3-1p7b
S0=$RD/checkpoints/sft-seed0
export HF_HOME=$WORK/.cache/huggingface HF_HUB_CACHE=$WORK/.cache/huggingface/hub
export HF_DATASETS_CACHE=$WORK/.cache/huggingface/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p "$RD/logs/m2" "$RD/notes/smoke"
cd "$RD/code"; export PYTHONPATH="$RD/code:${PYTHONPATH:-}"
echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID part=${SLURM_JOB_PARTITION:-?} date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | sed "s/^/[gpu] $(date +%T) /"; sleep 15; done ) &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT
test -s "$RD/notes/functional_transfer_full_heldout_count.json" || { echo "[FATAL] committed held-out count file missing"; exit 3; }
"$PY" scripts/functional_transfer_full.py --audited-adapter "$AUD" --seed0-adapter "$S0" --batch 64 --tag full --out-dir "$RD/notes"
echo "[DONE] $(date -Is)"
