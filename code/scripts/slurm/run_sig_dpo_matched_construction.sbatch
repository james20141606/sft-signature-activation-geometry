#!/bin/bash
#SBATCH --job-name=sig-dpo-mc
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>
#
# MATCHED-CONSTRUCTION DPO null. Registered in
# PREREGISTRATION_dpo_matched_construction.md BEFORE any reshuffled seed was
# trained; branches MC1-MC3 and the reader (scripts/mc_null_read.py) fixed there.
#
# WHY. The disjointness lead (SFT 0.0588-0.0630 vs DPO 0.0323-0.0365 at 29 steps)
# rests on a DPO band with NO data-order variance (fixed index order across seeds)
# against an SFT band that has it -- disclosed in the paper as anti-conservative
# for detecting DPO. This run adds that component: --reshuffle-per-epoch, RNG
# seeded with the run seed. Everything else byte-identical to the dose-29 rung.
#
# Per-seed checkpointing (the Addendum-40 lesson): each seed's training and each
# graph is skipped if its artifact already exists, so a QOSGrpGRES eviction costs
# at most one seed's work, and the rewiring step only runs when all 4 graphs exist.
#
# Submit:  sbatch code/scripts/slurm/run_sig_dpo_matched_construction.sbatch
# Read (Mac, committed reader only):
#   /usr/bin/python3 code/scripts/mc_null_read.py
set -euo pipefail

RD=$WORK/signature
SFT_PY=$WORK/envs/shaped-2026/bin/python     # trl 0.11.4 + peft 0.13.2
ATTR_PY=$WORK/envs/rs-qwen-env/bin/python   # sae-lens + transformer-lens

export HF_HOME=$WORK/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || true)
OFFLINE="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CKPT_ROOT=$RD/checkpoints
mkdir -p "$RD/logs/m2" "$RD/notes" "$CKPT_ROOT"

echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID part=${SLURM_JOB_PARTITION:-?} date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

PAIRS=$RD/data/bfcl_dpo_pairs.parquet
[ -f "$PAIRS" ] || { echo "[FATAL] missing $PAIRS -- the registered pair set must NOT be rebuilt"; exit 2; }

cd "$RD/code"
export PYTHONPATH="$RD/code:${PYTHONPATH:-}"

# ---------------------------------------------------------------- train + graph
for SEED in 0 1 2 3; do
  BASE_OUT=$CKPT_ROOT/dpo-mc-seed$SEED
  if [ ! -f "$BASE_OUT-step29/adapter_model.safetensors" ]; then
    echo "=== [DPO-MC seed $SEED] 29-step reshuffled trajectory ==="
    "$SFT_PY" scripts/sft_seed.py \
      --sft-script "$RD/code/training/dpo_warmup.py" \
      --seed "$SEED" \
      --base-model Qwen/Qwen3-1.7B \
      --train-parquet "$PAIRS" \
      --out-dir "$BASE_OUT" \
      --max-steps 29 --save-at 29 --reshuffle-per-epoch \
      --lr 5e-6 --beta 0.1 --batch-size 4 --grad-accum 8 \
      --lora-r 16 --lora-alpha 32
  else
    echo "=== [DPO-MC seed $SEED] adapter already present, skipping training ==="
  fi

  OUT=$BASE_OUT-step29
  grep -E '"r"|lora_alpha' "$OUT/adapter_config.json" | head -4
  G=$RD/notes/basediff_dpo_mc_seed${SEED}_step29_l14to22.json
  if [ ! -f "$G" ]; then
    echo "=== [DPO-MC seed $SEED] pile attribution graph, qwen-scope l14->22 n=500 ==="
    env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py \
      --backbone Qwen/Qwen3-1.7B --hf-model-path "$OUT" \
      --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
      --src-layer 14 --dst-layer 22 --src-sae-id layer14 --dst-sae-id layer22 \
      --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
      --keep-top-n 10000 --per-state-top-k 2000 \
      --out "$G"
  else
    echo "=== [DPO-MC seed $SEED] graph already present, skipping ==="
  fi
done

# --------------------------------------------------- matched-construction null
echo
echo "=== DPO matched-construction null @ dose 29 (6 pairs over 4 runs) ==="
"$ATTR_PY" scripts/circuit_rewiring.py \
  --graphs \
    base=$RD/notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json \
    SFT=$RD/notes/basediff_sft_l14to22.json \
    DPOMC_s0=$RD/notes/basediff_dpo_mc_seed0_step29_l14to22.json \
    DPOMC_s1=$RD/notes/basediff_dpo_mc_seed1_step29_l14to22.json \
    DPOMC_s2=$RD/notes/basediff_dpo_mc_seed2_step29_l14to22.json \
    DPOMC_s3=$RD/notes/basediff_dpo_mc_seed3_step29_l14to22.json \
  --seed-family DPOMC_s0 DPOMC_s1 DPOMC_s2 DPOMC_s3 \
  --focus DPOMC_s0 \
  --out "$RD/notes/circuit_rewiring_dpo_mc29.json"

echo
echo "[DONE] $(date -Is)"
