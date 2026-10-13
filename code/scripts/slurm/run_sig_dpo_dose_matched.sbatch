#!/bin/bash
#SBATCH --job-name=sig-dpo-dose
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
# <SLURM_PARTITION> first: the published 12-step runs ran on an A100 (JID 15102095), and
# the registered trajectory-continuity check compares per-step losses against them.
# bf16 reduction order is GPU-model dependent, so matching hardware keeps that check
# at its tight tolerance; the registration fixes a looser one if we land elsewhere.
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>
#
# L8(a): a DOSE-MATCHED DPO seed null. Registered in
# PREREGISTRATION_dpo_dose_matched.md BEFORE any dose-matched seed was trained;
# doses, branches M1-M5 and every check fixed there.
#
# WHY. The published DPO null (JID 15102095) returned branch D2 -- base->DPO does
# not separate from DPO's own seed band, z=-1.02 -- but those runs are 12 optimizer
# steps against the SFT arm's 29. So "DPO writes less footprint than SFT" was not
# separable from "this DPO run was shorter", and we disclosed that as L8(a) instead
# of fixing it. This job fixes it.
#
# WHAT. One 60-step trajectory per seed (0-3), adapters saved at the two registered
# rungs, giving a three-rung dose ladder:
#     12 steps -- the PUBLISHED arm; its adapters already exist and are NOT retrained
#     29 steps -- SFT's exact optimizer-step count; the PRIMARY comparison
#     60 steps -- ~2x SFT's dose; the extrapolation rung
# Recipe, data, seeds, graph protocol and scoring are byte-identical to the 12-step
# arm (PREREGISTRATION_dpo_seed_null.md); only the step count moves. lr stays 5e-6.
#
# Submit:  sbatch code/scripts/slurm/run_sig_dpo_dose_matched.sbatch
set -euo pipefail

RD=$WORK/signature
SFT_PY=$WORK/envs/shaped-2026/bin/python     # trl 0.11.4 + peft 0.13.2
ATTR_PY=$WORK/envs/rs-qwen-env/bin/python   # sae-lens + transformer-lens

export HF_HOME=$WORK/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || true)
# HF_HUB_OFFLINE is set per-command, NOT globally: transformers in shaped-2026 calls
# hub model_info() from its tokenizer loader even for a fully cached model, so an
# offline training step dies with OfflineModeIsEnabled (JID 14851118). The
# attribution step (rs-qwen-env) does want offline.
OFFLINE="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CKPT_ROOT=$RD/checkpoints
mkdir -p "$RD/logs/m2" "$RD/notes" "$CKPT_ROOT"

echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID part=${SLURM_JOB_PARTITION:-?} date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# The registration forbids rebuilding the preference set: all rungs must see the
# identical 398 pairs the published arm saw.
PAIRS=$RD/data/bfcl_dpo_pairs.parquet
[ -f "$PAIRS" ] || { echo "[FATAL] missing $PAIRS -- the registered pair set must NOT be rebuilt"; exit 2; }

cd "$RD/code"
export PYTHONPATH="$RD/code:${PYTHONPATH:-}"

# ---------------------------------------------------------------- train + graph
for SEED in 0 1 2 3; do
  BASE_OUT=$CKPT_ROOT/dpo-seed$SEED
  if [ ! -f "$BASE_OUT-step60/adapter_model.safetensors" ]; then
    echo "=== [DPO seed $SEED] 60-step trajectory, adapters at the registered rungs 29 and 60 ==="
    "$SFT_PY" scripts/sft_seed.py \
      --sft-script "$RD/code/training/dpo_warmup.py" \
      --seed "$SEED" \
      --base-model Qwen/Qwen3-1.7B \
      --train-parquet "$PAIRS" \
      --out-dir "$BASE_OUT" \
      --max-steps 60 --save-at 29,60 \
      --lr 5e-6 --beta 0.1 --batch-size 4 --grad-accum 8 \
      --lora-r 16 --lora-alpha 32
  else
    echo "=== [DPO seed $SEED] rung adapters already present, skipping training ==="
  fi

  for STEP in 29 60; do
    OUT=$BASE_OUT-step$STEP
    grep -E '"r"|lora_alpha' "$OUT/adapter_config.json" | head -4
    G=$RD/notes/basediff_dpo_seed${SEED}_step${STEP}_l14to22.json
    if [ ! -f "$G" ]; then
      echo "=== [DPO seed $SEED @ step $STEP] pile attribution graph, qwen-scope l14->22 n=500 ==="
      env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py \
        --backbone Qwen/Qwen3-1.7B --hf-model-path "$OUT" \
        --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
        --src-layer 14 --dst-layer 22 --src-sae-id layer14 --dst-sae-id layer22 \
        --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
        --keep-top-n 10000 --per-state-top-k 2000 \
        --out "$G"
    else
      echo "=== [DPO seed $SEED @ step $STEP] graph already present, skipping ==="
    fi
  done
done

# --------------------------------------------------- per-rung seed-noise nulls
# One null per rung: 6 pairs over the four runs AT THAT RUNG, focus = seed 0 at that
# rung. Identical construction and scored run as the published 12-step arm.
for STEP in 29 60; do
  echo
  echo "=== DPO seed-noise null @ dose $STEP (6 pairs over 4 runs) ==="
  "$ATTR_PY" scripts/circuit_rewiring.py \
    --graphs \
      base=$RD/notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json \
      SFT=$RD/notes/basediff_sft_l14to22.json \
      DPO_s0=$RD/notes/basediff_dpo_seed0_step${STEP}_l14to22.json \
      DPO_s1=$RD/notes/basediff_dpo_seed1_step${STEP}_l14to22.json \
      DPO_s2=$RD/notes/basediff_dpo_seed2_step${STEP}_l14to22.json \
      DPO_s3=$RD/notes/basediff_dpo_seed3_step${STEP}_l14to22.json \
    --seed-family DPO_s0 DPO_s1 DPO_s2 DPO_s3 \
    --focus DPO_s0 \
    --out "$RD/notes/circuit_rewiring_dpo_dose${STEP}.json"
done

echo
echo "[DONE] $(date -Is)"
