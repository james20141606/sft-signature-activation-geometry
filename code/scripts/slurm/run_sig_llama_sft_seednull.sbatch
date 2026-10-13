#!/bin/bash
#SBATCH --job-name=sig-llama-sft
#SBATCH --output=$WORK/signature/logs/m2/%x-%j.out
#SBATCH --error=$WORK/signature/logs/m2/%x-%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=160G --gres=gpu:1 --time=03:30:00
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --partition=<SLURM_PARTITION>,<SLURM_PARTITION>
#
# Second backbone for the SFT calibrated estimator. Registered in
# PREREGISTRATION_llama_second_backbone.md BEFORE any lsft-seed* existed; layer
# rule fixed by formula (l16->l25 = fractional-depth transplant, coinciding with
# the committed Tulu instrument and its measured folds); branches R1-R4; reader
# code/scripts/llama_repl_read.py. Per-artifact skip-if-exists (Addendum 40).
#
# Submit:  sbatch code/scripts/slurm/run_sig_llama_sft_seednull.sbatch
set -euo pipefail
RD=$WORK/signature
SFT_PY=$WORK/envs/shaped-2026/bin/python
ATTR_PY=$WORK/envs/rs-qwen-env/bin/python
export HF_HOME=$WORK/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || true)
OFFLINE="HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
CKPT_ROOT=$RD/checkpoints
mkdir -p "$RD/logs/m2" "$RD/notes" "$CKPT_ROOT"
echo "[INFO] host=$(hostname) job=$SLURM_JOB_ID date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# v2 REGISTRATION: training uses the ungated NousResearch mirror ID; the GRAPH
# step reverts to the committed Tulu instrument's configuration -- TL name
# meta-llama/Llama-3.1-8B, weights via a cache symlink onto the mirror (created
# only after verifying the target exists), HF_HUB_OFFLINE=1. SAEs warm-cached
# online (CPU, no measurement) before the offline graph loop.
MIRROR=$HF_HUB_CACHE/models--NousResearch--Meta-Llama-3.1-8B
OFFICIAL=$HF_HUB_CACHE/models--meta-llama--Llama-3.1-8B
[ -L "$OFFICIAL" ] && [ ! -e "$OFFICIAL" ] && rm "$OFFICIAL"
if [ ! -e "$OFFICIAL" ]; then
  [ -e "$MIRROR" ] || { echo "[FATAL] mirror cache absent; refusing to create a dangling symlink"; exit 2; }
  ln -s "$MIRROR" "$OFFICIAL"
fi

SFT_PARQUET=$RD/data/bfcl_sft_train.parquet
[ -f "$SFT_PARQUET" ] || { echo "[FATAL] registered 900-row SFT set missing"; exit 2; }

cd "$RD/code"; export PYTHONPATH="$RD/code:${PYTHONPATH:-}"

FS=5.2784   # L16R fold, measured for the committed Tulu instrument
FD=2.0237   # L25R fold
GCOMMON=( --backbone meta-llama/Llama-3.1-8B --sae-release llama_scope_lxr_32x \
  --src-layer 16 --dst-layer 25 --src-sae-id l16r_32x --dst-sae-id l25r_32x \
  --fold-norm-scale-src $FS --fold-norm-scale-dst $FD \
  --n-prompts 500 --seq-len 64 --device cuda --dtype bfloat16 \
  --keep-top-n 10000 --per-state-top-k 2000 )

echo "=== warm-cache the two SAEs (online, CPU, no measurement) ==="
"$ATTR_PY" - <<'WARM'
from sae_lens import SAE
for sid in ("l16r_32x", "l25r_32x"):
    SAE.from_pretrained("llama_scope_lxr_32x", sid, device="cpu")
    print("cached", sid, flush=True)
WARM

# base graph (skip if the Tulu v2 base graph already exists -- same instrument)
GBASE=$RD/notes/tulu_v2_base_l16to25.json
if [ ! -f "$GBASE" ]; then
  GBASE=$RD/notes/llama_base_l16to25.json
  [ -f "$GBASE" ] || env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py \
      "${GCOMMON[@]}" --out "$GBASE"
fi
echo "[INFO] base graph: $GBASE"

for SEED in 0 1 2 3; do
  OUT=$CKPT_ROOT/lsft-seed$SEED
  if [ ! -f "$OUT/adapter_model.safetensors" ]; then
    echo "=== [LSFT seed $SEED] train (recipe byte-identical to the Qwen SFT null) ==="
    "$SFT_PY" scripts/sft_seed.py \
      --sft-script "$RD/code/training/sft_warmup.py" \
      --seed "$SEED" \
      --base-model NousResearch/Meta-Llama-3.1-8B \
      --train-parquet "$SFT_PARQUET" \
      --out-dir "$OUT" \
      --epochs 1.0 --lr 5e-5 --batch-size 4 --grad-accum 8 \
      --lora-r 16 --lora-alpha 32 \
      --chat-template-from NousResearch/Meta-Llama-3.1-8B-Instruct
  else
    echo "=== [LSFT seed $SEED] adapter already present, skipping ==="
  fi
  G=$RD/notes/basediff_llama_sft_seed${SEED}_l16to25.json
  if [ ! -f "$G" ]; then
    echo "=== [LSFT seed $SEED] graph l16->25 n=500 ==="
    env $OFFLINE "$ATTR_PY" scripts/m2_attribution_graph.py "${GCOMMON[@]}" \
      --hf-model-path "$OUT" --out "$G"
  else
    echo "=== [LSFT seed $SEED] graph already present, skipping ==="
  fi
done

echo "=== Llama SFT seed null (6 pairs over 4 runs) ==="
"$ATTR_PY" scripts/circuit_rewiring.py \
  --graphs \
    base=$GBASE \
    LSFT_s0=$RD/notes/basediff_llama_sft_seed0_l16to25.json \
    LSFT_s1=$RD/notes/basediff_llama_sft_seed1_l16to25.json \
    LSFT_s2=$RD/notes/basediff_llama_sft_seed2_l16to25.json \
    LSFT_s3=$RD/notes/basediff_llama_sft_seed3_l16to25.json \
  --seed-family LSFT_s0 LSFT_s1 LSFT_s2 LSFT_s3 \
  --focus LSFT_s0 \
  --out "$RD/notes/circuit_rewiring_llama_sft_seednull.json"
echo "[DONE] $(date -Is)"
