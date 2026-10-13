#!/bin/bash
#SBATCH --job-name=sig-dpo-seednull
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
# L8 (AC-named): build a PER-ALGORITHM seed-noise null for DPO -- the one stage that
# had neither its own null nor a title claim. Registered in
# PREREGISTRATION_dpo_seed_null.md BEFORE any seed was trained; branches D1-D4 fixed there.
#
# NB (2026-08-01): the header that stood here until now was a stale copy of
# run_sig_sft_seed_null.sbatch's. It described training "three further LoRA-SFT seeds"
# and told the reader to submit the SFT script. The executable body below was always
# the DPO one and is what produced JID 15102095 and the committed receipt; only the
# prose was wrong. No gate in this repo reads sbatch comments, which is why it
# survived -- recorded here rather than silently corrected.
#
# WHAT THIS JOB ACTUALLY DOES. Builds the preference pairs ONCE (build_dpo_pairs.py,
# greedy/deterministic, so all four seeds see an identical set), then trains four
# LoRA-DPO seeds (0-3) through code/scripts/sft_seed.py so set_seed runs before the
# model is built and LoRA init is seeded, rebuilds the pile-arm attribution graph for
# each, and scores the 6-pair DPO null with the DPO family as the null.
#
# Recipe, matched to the audited SFT recipe wherever DPO permits: LoRA r=16 alpha=32
# on q/k/v/o_proj (6,422,528 trainable params, 0.37% -- identical to the SFT arm),
# 398 BFCL-derived preference pairs after AMENDMENT 1, 1 epoch = 12 optimizer steps,
# lr 5e-6 (10x below SFT's 5e-5, the standard DPO/SFT ratio, fixed by the
# registration and not tuned), beta 0.1, bs 4 x accum 8, bf16.
#
# The DPO loss is implemented directly in code/training/dpo_warmup.py rather than via
# trl: trl 0.11.4's DPOTrainer is incompatible with transformers 4.57.3 in this env
# (get_batch_samples() takes 3 positional arguments but 4 were given, JID 15099744).
# The reference model is the same weights under disable_adapter(), so no second model
# is loaded; loss sign and completion-only masking are unit-tested in
# code/tests/test_dpo_loss.py.
#
# The 12 optimizer steps here are the dose confound disclosed as L8(a); the
# dose-matched ladder that removes it is run by run_sig_dpo_dose_matched.sbatch.
#
# Submit:  sbatch code/scripts/slurm/run_sig_dpo_seed_null.sbatch
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
PAIRS=$RD/data/bfcl_dpo_pairs.parquet
if [ ! -f "$PAIRS" ]; then
  echo "=== [pairs] building BFCL preference pairs ONCE (greedy, deterministic) ==="
  "$SFT_PY" "$RD/code/training/build_dpo_pairs.py" \
    --train-parquet "$SFT_PARQUET" --base-model Qwen/Qwen3-1.7B --out "$PAIRS"
fi
[ -f "$PAIRS" ] || { echo "[FATAL] pair construction failed"; exit 2; }

for SEED in 0 1 2 3; do
  OUT=$CKPT_ROOT/dpo-seed$SEED
  if [ ! -f "$OUT/adapter_model.safetensors" ]; then
    echo "=== [DPO seed $SEED] train (LoRA target/rank matched to the SFT recipe) ==="
    "$SFT_PY" scripts/sft_seed.py \
      --sft-script "$RD/code/training/dpo_warmup.py" \
      --seed "$SEED" \
      --base-model Qwen/Qwen3-1.7B \
      --train-parquet "$PAIRS" \
      --out-dir "$OUT" \
      --epochs 1.0 --lr 5e-6 --beta 0.1 --batch-size 4 --grad-accum 8 \
      --lora-r 16 --lora-alpha 32
  else
    echo "=== [DPO seed $SEED] adapter already present, skipping training ==="
  fi
  grep -E '"r"|lora_alpha|target_modules' "$OUT/adapter_config.json" | head -8

  G=$RD/notes/basediff_dpo_seed${SEED}_l14to22.json
  if [ ! -f "$G" ]; then
    echo "=== [DPO seed $SEED] pile attribution graph, qwen-scope l14->22 n=500 ==="
    env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py \
      --backbone Qwen/Qwen3-1.7B --hf-model-path "$OUT" \
      --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
      --src-layer 14 --dst-layer 22 --src-sae-id layer14 --dst-sae-id layer22 \
      --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
      --keep-top-n 10000 --per-state-top-k 2000 \
      --out "$G"
  else
    echo "=== [DPO seed $SEED] graph already present, skipping ==="
  fi
done

# ------------------------------------------------------------- DPO seed-noise null
# Null = all 6 pairs among the four specified DPO runs. Contrasts = base->each DPO
# run, plus the audited SFT checkpoint and base->GRPO for reference on the same axis.
echo
echo "=== DPO seed-noise null (6 pairs over 4 DPO runs) -- closes L8 ==="
"$ATTR_PY" scripts/circuit_rewiring.py \
  --graphs \
    base=$RD/notes/attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json \
    SFT=$RD/notes/basediff_sft_l14to22.json \
    DPO_s0=$RD/notes/basediff_dpo_seed0_l14to22.json \
    DPO_s1=$RD/notes/basediff_dpo_seed1_l14to22.json \
    DPO_s2=$RD/notes/basediff_dpo_seed2_l14to22.json \
    DPO_s3=$RD/notes/basediff_dpo_seed3_l14to22.json \
    GRPO=$RD/notes/basediff_grpo_l14to22.json \
  --seed-family DPO_s0 DPO_s1 DPO_s2 DPO_s3 \
  --focus DPO_s0 \
  --out "$RD/notes/circuit_rewiring_dpo_seednull.json"

echo
echo "[DONE] $(date -Is)"
