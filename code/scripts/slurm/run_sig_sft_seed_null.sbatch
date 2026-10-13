#!/bin/bash
#SBATCH --job-name=sig-sft-seednull
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=<SLURM_ACCOUNT>
# >=80 GB only: the SFT step (bs 8 x 2048 tokens, bf16, no grad ckpt) OOMs on a
# 44 GB L40S (JID 14851165); the original SFT ran on an 80 GB H100. Same partition
# list as the original SFT warm-up job, plus A100.
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>,<SLURM_PARTITION>
#
# T1 (decision-driving): build a PER-ALGORITHM seed-noise null for the SFT stage.
#
# Until now base->SFT (z=-15.9) was scored against a GRPO-family null, i.e. the
# positive half of the paper's title was the uncalibrated single-run difference the
# introduction condemns: there was exactly ONE SFT checkpoint. This job trains three
# further LoRA-SFT seeds under the IDENTICAL recipe as the audited checkpoint
# (code/training/sft_warmup.py, vendored into this repo: trl SFTTrainer, LoRA r=16 alpha=32 on
# q/k/v/o_proj, 900 BFCL SFT rows, 1 epoch = 29 optimizer steps, lr 5e-5, bs 8 x
# accum 4, bf16 -- verified against checkpoints/sft-qwen3-1p7b/adapter_config.json),
# rebuilds the pile-arm attribution graph for each, and recomputes the rewiring
# comparison with the SFT family as the null. Training is ~40 s/seed (measured: JID
# 7794435 train_runtime 38.9 s), so the whole battery is one short GPU job.
#
# The audited SFT checkpoint used the script default --seed 0 but its LoRA init was
# NOT seeded (get_peft_model runs before Trainer's set_seed), so it is not exactly
# reproducible; the three new seeds go through code/scripts/sft_seed.py, which seeds
# before the model is built. Seed 0 is therefore re-run here as sft-seed0 alongside
# 1/2/3, giving a 4-run, 6-pair SFT null of fully specified runs, while the original
# checkpoint stays the audited one.
#
# Submit:  sbatch code/scripts/slurm/run_sig_sft_seed_null.sbatch
set -euo pipefail

RD=$WORK/signature
# Everything the seeded-SFT chain needs is vendored INSIDE this repo: the training script
# (code/training/sft_warmup.py), its 900-row input (data/bfcl_sft_train.parquet) and the
# seeding wrapper (code/scripts/sft_seed.py). No sibling-repo path remains, so a clean
# checkout plus the two conda envs and the HF cache reproduces the SFT seed null.
SFT_PY=$WORK/envs/shaped-2026/bin/python     # trl 0.11.4 + peft 0.13.2
ATTR_PY=$WORK/envs/rs-qwen-env/bin/python   # sae-lens + transformer-lens

export HF_HOME=$WORK/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || true)
# NB: HF_HUB_OFFLINE is set per-command, NOT globally. transformers in the
# shaped-2026 env calls hub model_info() from its tokenizer loader even for a fully
# cached model, so an offline SFT step dies with OfflineModeIsEnabled (JID 14851118).
# The attribution step (rs-qwen-env) does want offline, as in the sibling sbatches.
OFFLINE="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CKPT_ROOT=$RD/checkpoints
mkdir -p "$RD/logs/m2" "$RD/notes" "$CKPT_ROOT"

echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID part=${SLURM_JOB_PARTITION:-?} date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

SFT_PARQUET=$RD/data/bfcl_sft_train.parquet
[ -f "$SFT_PARQUET" ] || { echo "[FATAL] missing $SFT_PARQUET"; exit 2; }

cd "$RD/code"
export PYTHONPATH="$RD/code:${PYTHONPATH:-}"

# ---------------------------------------------------------------- train + graph
for SEED in 0 1 2 3; do
  OUT=$CKPT_ROOT/sft-seed$SEED
  if [ ! -f "$OUT/adapter_model.safetensors" ]; then
    echo "=== [SFT seed $SEED] train (identical recipe to the audited checkpoint) ==="
    "$SFT_PY" scripts/sft_seed.py \
      --sft-script "$RD/code/training/sft_warmup.py" \
      --seed "$SEED" \
      --base-model Qwen/Qwen3-1.7B \
      --train-parquet "$SFT_PARQUET" \
      --out-dir "$OUT" \
      --epochs 1.0 --lr 5e-5 --batch-size 8 --grad-accum 4 \
      --lora-r 16 --lora-alpha 32
  else
    echo "=== [SFT seed $SEED] adapter already present, skipping training ==="
  fi
  grep -E '"r"|lora_alpha|target_modules' "$OUT/adapter_config.json" | head -8

  G=$RD/notes/basediff_sft_seed${SEED}_l14to22.json
  if [ ! -f "$G" ]; then
    echo "=== [SFT seed $SEED] pile attribution graph, qwen-scope l14->22 n=500 ==="
    env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py \
      --backbone Qwen/Qwen3-1.7B --hf-model-path "$OUT" \
      --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
      --src-layer 14 --dst-layer 22 --src-sae-id layer14 --dst-sae-id layer22 \
      --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
      --keep-top-n 10000 --per-state-top-k 2000 \
      --out "$G"
  else
    echo "=== [SFT seed $SEED] graph already present, skipping ==="
  fi
done

# ------------------------------------------------------------- SFT seed-noise null
# Null = all 6 pairs among the four specified SFT runs. Contrasts = base->each SFT
# run, plus the audited SFT checkpoint and base->GRPO for reference on the same axis.
echo
echo "=== SFT seed-noise null (6 pairs over 4 SFT runs) ==="
"$ATTR_PY" scripts/circuit_rewiring.py \
  --graphs \
    base=$RD/notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json \
    SFT=$RD/notes/basediff_sft_l14to22.json \
    SFT_s0=$RD/notes/basediff_sft_seed0_l14to22.json \
    SFT_s1=$RD/notes/basediff_sft_seed1_l14to22.json \
    SFT_s2=$RD/notes/basediff_sft_seed2_l14to22.json \
    SFT_s3=$RD/notes/basediff_sft_seed3_l14to22.json \
    GRPO=$RD/notes/basediff_grpo_l14to22.json \
  --seed-family SFT_s0 SFT_s1 SFT_s2 SFT_s3 \
  --focus SFT \
  --out "$RD/notes/circuit_rewiring_sft_seednull.json"

echo
echo "[DONE] $(date -Is)"
