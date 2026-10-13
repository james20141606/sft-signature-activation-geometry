"""Activation-side single-checkpoint SAE 'narrowing/sparsification' metric battery.

Generalizes the base-vs-post control beyond attribution-graph *concentration*
(m2_attribution_graph.py / basediff_analyze.py) to the *activation-side* metrics
the RL-internals literature actually uses to claim RL "narrows" or "sparsifies"
features (Yue et al. 2025 'does RLVR incentivize reasoning beyond base'; the
entropy-mechanism line). For ONE checkpoint and ONE SAE it streams a probe set,
encodes every token position, and accumulates per-feature usage statistics:

  * mean_L0            : mean # active SAE features per token (sparsification axis;
                         RL-narrowing would predict L0 drops post-RL)
  * frac_dead          : fraction of dictionary features that never fire
  * n_ever_active      : # features active at least once
  * firing_freq Gini   : Gini of per-feature firing frequency over tokens
                         (narrowing -> a few features fire on ~all tokens -> higher)
  * firing_freq top1%  : share of total firing mass held by top-1% of active feats
  * meanact Gini       : Gini of per-feature mean activation magnitude
  * usage_entropy      : Shannon entropy (nats) of the feature-usage distribution
                         p_i = firing_count_i / total_firings (narrowing -> lower)
  * usage_entropy_norm : usage_entropy / log(n_ever_active)  (in [0,1])

Run this on base / SFT / GRPO with the SAME base-trained SAE; if the post-base
deltas are within noise, the activation-side 'narrowing' metrics are ALSO a
property of the pretrained model + SAE basis, not an RL signature -- the
generalized form of the headline cautionary result.

Usage:
    python scripts/sae_feature_stats.py \
        --backbone Qwen/Qwen3-1.7B \
        --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
        --dst-layer 22 --dst-sae-id blocks.22.hook_resid_post \
        [--hf-model-path /path/to/sft_or_grpo_ckpt] \
        --n-prompts 500 --seq-len 64 \
        --out notes/featstats_<ckpt>_l22.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os as _os
import time
from pathlib import Path

import numpy as np
import torch


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x[-1] == 0:
        return 0.0
    c = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--sae-release", required=True)
    p.add_argument("--dst-layer", type=int, required=True)
    p.add_argument("--dst-sae-id", type=str, default=None)
    p.add_argument("--hf-model-path", type=str, default=None,
                   help="fine-tuned (SFT/GRPO) checkpoint loaded into --backbone arch")
    p.add_argument("--n-prompts", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--dataset", choices=["pile", "gsm8k", "bfcl"], default="pile",
                   help="probe distribution: generic web text (pile-10k), math reasoning (gsm8k), "
                        "or the RL task distribution (BFCL-v4 function-calling verl parquet)")
    p.add_argument("--bfcl-parquet", type=str,
                   default="$WORK/grpo_project/data/bfcl_train.parquet",
                   help="verl parquet with a 'prompt' column of chat messages (--dataset bfcl)")
    p.add_argument("--keep-side", choices=["first", "last"], default="first",
                   help="which seq-len tokens to keep when truncating; 'last' keeps the "
                        "generation decision point of chat-templated task prompts")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--eps", type=float, default=1e-6, help="activation threshold for 'active'")
    p.add_argument("--fold-norm-scale", type=float, default=None,
                   help="fold_activation_norm_scaling_factor for the SAE (llama_scope: "
                        "sqrt(d_in)/dataset_average_activation_norm); required for SAEs with "
                        "normalize_activations='expected_average_only_in' to run at the intended operating point")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("featstats")

    from sae_lens import SAE, HookedSAETransformer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    dst_id = args.dst_sae_id or f"blocks.{args.dst_layer}.hook_resid_post"
    log.info(f"loading SAE_dst {args.sae_release} {dst_id}")
    sae_dst = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.sae_release, sae_id=dst_id, device=args.device,
    )[0].to(dtype)
    if args.fold_norm_scale:
        sae_dst.fold_activation_norm_scaling_factor(args.fold_norm_scale)
        log.info(f"folded norm scale {args.fold_norm_scale} into SAE")
    dst_hook = sae_dst.cfg.metadata["hook_name"]
    d_sae = int(sae_dst.cfg.d_sae)
    log.info(f"d_sae={d_sae}  dst_hook={dst_hook}")

    # ---- backbone (optionally fine-tuned weights into base arch) ----
    if args.hf_model_path:
        log.info(f"loading fine-tuned weights from {args.hf_model_path} into {args.backbone}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        is_lora = _os.path.exists(_os.path.join(args.hf_model_path, "adapter_config.json"))
        if is_lora:
            from peft import PeftModel
            log.info("detected LoRA adapter; loading base + merging")
            base = AutoModelForCausalLM.from_pretrained(
                args.backbone, torch_dtype=dtype, trust_remote_code=True,
            )
            hf_model = PeftModel.from_pretrained(base, args.hf_model_path).merge_and_unload()
            hf_tok = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.hf_model_path, torch_dtype=dtype, trust_remote_code=True,
            )
            hf_tok = AutoTokenizer.from_pretrained(args.hf_model_path, trust_remote_code=True)
        model = HookedSAETransformer.from_pretrained_no_processing(
            args.backbone, hf_model=hf_model, tokenizer=hf_tok, dtype=dtype, device=args.device,
        )
        del hf_model
    else:
        log.info(f"loading backbone {args.backbone}")
        model = HookedSAETransformer.from_pretrained_no_processing(
            args.backbone, dtype=dtype, device=args.device,
        )
    model.eval()

    # ---- probe set ----
    from datasets import load_dataset
    prompts = []
    if args.dataset == "pile":
        log.info(f"loading pile-10k for {args.n_prompts} prompts (generic web text)")
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        for r in ds:
            text = r.get("text") or ""
            if len(text) < 200:
                continue
            prompts.append(text[:2000])
            if len(prompts) >= args.n_prompts:
                break
    elif args.dataset == "gsm8k":  # in-distribution math reasoning (question + answer chain)
        log.info(f"loading gsm8k(main) for {args.n_prompts} prompts (in-distribution reasoning)")
        ds = load_dataset("gsm8k", "main", split="train")
        for r in ds:
            text = (r.get("question") or "") + "\n" + (r.get("answer") or "")
            if len(text) < 50:
                continue
            prompts.append(text[:2000])
            if len(prompts) >= args.n_prompts:
                break
    else:  # bfcl: the RL task distribution, chat-templated exactly as verl rollouts
        log.info(f"loading BFCL parquet {args.bfcl_parquet} for {args.n_prompts} prompts")
        import pandas as pd
        df = pd.read_parquet(args.bfcl_parquet)
        for _, row in df.iterrows():
            msgs = [dict(m) for m in row["prompt"]]
            prompts.append(model.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
            if len(prompts) >= args.n_prompts:
                break
    log.info(f"got {len(prompts)} prompts from {args.dataset}")

    # ---- accumulate per-feature stats over ALL token positions ----
    fire_count = np.zeros(d_sae, dtype=np.float64)   # # tokens where feature active
    act_sum = np.zeros(d_sae, dtype=np.float64)       # sum of activation over tokens
    n_tokens = 0
    l0_sum = 0.0
    # streaming FVU accumulators: reconstruction quality of the base-trained SAE
    # on THIS checkpoint's activations -> bounds the base-SAE blind spot
    # (FVE formula matches shaped sae_pilot.compute_fve)
    d_model = None
    resid_ss = 0.0       # sum over (tokens,dims) of (h - decode(encode(h)))^2
    h_ss = 0.0           # sum over (tokens,dims) of h^2
    h_dim_sum = None     # per-dim sum of h (for the per-dim mean)
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(prompts):
            tokens = model.to_tokens(text)
            tokens = tokens[:, -args.seq_len:] if args.keep_side == "last" else tokens[:, : args.seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=[dst_hook])
            h = cache[dst_hook][0]                       # [seq, d_model]
            acts = sae_dst.encode(h).float()             # [seq, d_sae]
            h_hat = sae_dst.decode(acts.to(h.dtype)).float()
            hf = h.float()
            if h_dim_sum is None:
                d_model = hf.shape[-1]
                h_dim_sum = np.zeros(d_model, dtype=np.float64)
            resid_ss += ((hf - h_hat) ** 2).sum().item()
            h_ss += (hf ** 2).sum().item()
            h_dim_sum += hf.sum(dim=0).cpu().numpy()
            active = acts > args.eps                      # bool [seq, d_sae]
            fire_count += active.sum(dim=0).cpu().numpy()
            act_sum += acts.sum(dim=0).cpu().numpy()
            l0_sum += active.sum().item()
            n_tokens += acts.shape[0]
            if (i + 1) % 100 == 0:
                log.info(f"{i+1}/{len(prompts)} prompts  n_tokens={n_tokens}")
    elapsed = time.time() - t0

    # ---- derive metrics ----
    ever = fire_count > 0
    n_ever = int(ever.sum())
    frac_dead = float(1.0 - n_ever / d_sae)
    mean_L0 = float(l0_sum / max(1, n_tokens))
    firing_freq = fire_count / max(1, n_tokens)
    mean_act = act_sum / max(1, n_tokens)

    ff_active = firing_freq[ever]
    ma_active = mean_act[ever]
    # firing-frequency top-1%-of-active share
    s = np.sort(ff_active)[::-1]
    k = max(1, n_ever // 100)
    ff_top1 = float(s[:k].sum() / ff_active.sum()) if ff_active.sum() > 0 else 0.0
    # usage entropy over total firings
    total_fire = fire_count.sum()
    pvec = fire_count[ever] / total_fire if total_fire > 0 else np.zeros(n_ever)
    usage_entropy = float(-(pvec * np.log(np.clip(pvec, 1e-12, None))).sum())
    usage_entropy_norm = float(usage_entropy / math.log(n_ever)) if n_ever > 1 else 0.0
    # activation-intensity level (the axis rl_circuit_intensity2026 claims RL raises):
    # total activation mass per token, and mean activation magnitude among firing events
    act_mass_per_token = float(act_sum.sum() / max(1, n_tokens))
    total_fires = fire_count.sum()
    mean_act_when_active = float(act_sum.sum() / total_fires) if total_fires > 0 else 0.0
    # FVU/FVE of the base-trained SAE on this checkpoint's activations
    # var_total = mean over (tokens,dims) of (h - per-dim mean)^2
    nd = max(1, n_tokens) * (d_model or 1)
    var_resid = resid_ss / nd
    var_total = (h_ss - float((h_dim_sum ** 2).sum()) / max(1, n_tokens)) / nd
    fvu = float(var_resid / max(var_total, 1e-8))
    fve = float(1.0 - fvu)

    payload = {
        "backbone": args.backbone,
        "hf_model_path": args.hf_model_path,
        "sae_release": args.sae_release,
        "dst_layer": args.dst_layer,
        "dst_hook": dst_hook,
        "d_sae": d_sae,
        "n_prompts": len(prompts),
        "n_tokens": n_tokens,
        "seq_len": args.seq_len,
        "dataset": args.dataset,
        "prompt_source": args.bfcl_parquet if args.dataset == "bfcl" else args.dataset,
        "keep_side": args.keep_side,
        "elapsed_seconds": elapsed,
        "mean_L0": round(mean_L0, 4),
        "n_ever_active": n_ever,
        "frac_dead": round(frac_dead, 4),
        "firing_freq_gini": round(gini(ff_active), 4),
        "firing_freq_top1pct_active": round(ff_top1, 4),
        "meanact_gini": round(gini(ma_active), 4),
        "usage_entropy_nats": round(usage_entropy, 4),
        "usage_entropy_norm": round(usage_entropy_norm, 4),
        "act_mass_per_token": round(act_mass_per_token, 4),
        "mean_act_when_active": round(mean_act_when_active, 4),
        "fvu": round(fvu, 4),
        "fve": round(fve, 4),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"metrics: {json.dumps({k: v for k, v in payload.items() if k not in ('backbone','sae_release','dst_hook')})}")
    log.info(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
