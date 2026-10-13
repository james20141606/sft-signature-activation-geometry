"""Signature M2 — first production attribution graph build.

Pipeline:
  1. Load Pythia-70m + its pythia-70m-deduped-res-sm SAE at layers L_src, L_dst
  2. Load the 32-feature panel from notes/panel_v0_pythia70m.json (selection
     receipt: function-word-outcome screen; committed with this paper)
     (these are the upstream features of interest — restrict downstream
     bookkeeping to those; downstream layer scans the full SAE)
  3. Stream pile-10k for N prompts; capture (h_src, h_dst) per prompt
  4. Run linear_probe_attribution_sparse + build_graph_from_sparse_fn
  5. Dump AttributionGraph as JSON (edges_src, edges_dst, weights, d_sae_*)

Usage (under sbatch):
    python scripts/m2_attribution_graph.py \
        --backbone EleutherAI/pythia-70m-deduped \
        --sae-release pythia-70m-deduped-res-sm \
        --src-layer 3 --dst-layer 4 \
        --panel-features notes/panel_v0_pythia70m.json \
        --n-prompts 500 --seq-len 64 \
        --out $WORK/signature/notes/attribution_graph_v0.json

Defaults match shaped's feature_id_pythia_real.py for backbone consistency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from attribution_fn import linear_probe_attribution_sparse
from attribution_graph import build_graph_from_sparse_fn


def stream_states(model, prompts, seq_len, src_hook, dst_hook, src_layer, dst_layer,
                  keep_side="first", states_per_prompt=1):
    """Yield states with h_src/h_dst from each prompt.

    Following shaped's `feature_id_pythia_real.py` convention: take the residual
    at the last position because the SAE was trained on per-token activations
    and we want the per-prompt summary state most informative about the
    next-token outcome (function-word prediction in shaped's case).

    keep_side="last" keeps the FINAL seq_len tokens instead of the first —
    for chat-templated task prompts (BFCL) the last position is the
    generation decision point, which must not be truncated away.

    `states_per_prompt` (default 1) controls how many token positions are
    sampled per prompt. **1 reproduces the original last-token-only behaviour
    bit-for-bit.** Values >1 take evenly spaced positions across the kept
    window (always including the last).

    Why this exists: with 1 state/prompt the effective sample size for the
    graph is n_prompts. On chat-templated task prompts (BFCL) every prompt ends
    at the same generation-decision point after a near-identical system prompt,
    so those states are highly redundant, the averaged edge weights are noisy,
    and the hard top-N truncation flips membership between runs — the measured
    seed-null band is ~25x wider than on pile and the SFT positive control does
    not separate. Sampling more positions per prompt raises the effective n and
    narrows the band (`sae_feature_stats.py` already accumulates over all
    positions; the graph path did not)."""
    for i, prompt in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt)
            tokens = tokens[:, -seq_len:] if keep_side == "last" else tokens[:, :seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=[src_hook, dst_hook])
            T = tokens.shape[1]
            if states_per_prompt <= 1:
                positions = [T - 1]
            else:
                k = min(int(states_per_prompt), T)
                positions = sorted(set(
                    np.linspace(0, T - 1, k).round().astype(int).tolist()
                ))
            for pos in positions:
                yield {
                    "prompt_idx": i,
                    "pos": int(pos),
                    "h_src": cache[src_hook][0, pos].clone(),
                    "h_dst": cache[dst_hook][0, pos].clone(),
                }
        except Exception as e:
            logging.warning(f"prompt {i} failed: {e}")
            continue


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--sae-release", default="pythia-70m-deduped-res-sm")
    p.add_argument("--src-layer", type=int, default=3)
    p.add_argument("--dst-layer", type=int, default=4)
    p.add_argument("--src-sae-id", type=str, default=None,
                   help="Override SAE id for src (default: blocks.{src-layer}.hook_resid_post). "
                        "Some releases use e.g. 'layer14'.")
    p.add_argument("--dst-sae-id", type=str, default=None,
                   help="Override SAE id for dst (default: blocks.{dst-layer}.hook_resid_post).")
    p.add_argument("--panel-features", "--shaped-features", dest="shaped_features",
                   type=str, default=None,
                   help="JSON with 'selected' list of upstream feature IDs (shaped's M1 output).")
    p.add_argument("--n-prompts", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--dataset", choices=["pile", "bfcl"], default="pile",
                   help="probe distribution: generic web text (pile-10k) or the RL task "
                        "distribution (BFCL-v4 function-calling verl parquet, chat-templated)")
    p.add_argument("--bfcl-parquet", type=str,
                   default="$WORK/grpo_project/data/bfcl_train.parquet",
                   help="verl parquet with a 'prompt' column of chat messages (--dataset bfcl)")
    p.add_argument("--keep-side", choices=["first", "last"], default="first",
                   help="which seq-len tokens to keep when truncating; 'last' keeps the "
                        "generation decision point of chat-templated task prompts")
    p.add_argument("--random-sae", choices=["none", "random-dir", "rotate"],
                   default="none",
                   help="matched random-SAE control (arXiv:2602.14111 sanity check). "
                        "'random-dir' = random unit encoder directions; 'rotate' = real "
                        "encoder under a random orthogonal rotation of the input space "
                        "(more conservative). Both match d_sae and calibrated L0. If the "
                        "headline effect survives, it is not an SAE finding.")
    p.add_argument("--random-sae-seed", type=int, default=0,
                   help="seed for the random-SAE control")
    p.add_argument("--states-per-prompt", type=int, default=1,
                   help="token positions sampled per prompt (evenly spaced, always "
                        "including the last). 1 = original last-token-only behaviour. "
                        ">1 raises the effective sample size for the graph, which is "
                        "what narrows the seed-null band on redundant chat-templated "
                        "task prompts (BFCL).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--keep-top-n", type=int, default=10000)
    p.add_argument("--per-state-top-k", type=int, default=2000)
    p.add_argument("--hf-model-path", type=str, default=None,
                   help="Local fine-tuned checkpoint (SFT/GRPO) to load into --backbone architecture. "
                        "Enables base-vs-SFT-vs-GRPO baseline-difference comparison.")
    p.add_argument("--fold-norm-scale-src", type=float, default=None,
                   help="fold_activation_norm_scaling_factor for the src SAE. Needed for SAEs with "
                        "normalize_activations='expected_average_only_in' (llama_scope: "
                        "sqrt(d_in)/dataset_average_activation_norm), which otherwise run unnormalized.")
    p.add_argument("--fold-norm-scale-dst", type=float, default=None,
                   help="same for the dst SAE")
    p.add_argument("--dump-states", type=str, default=None,
                   help="sftband_20260926: also write an .npz with the exact (h_src, h_dst) "
                        "states the graph is built from, plus the dst-SAE code mean and "
                        "firing count per feature over those states. Off by default; "
                        "does not change the graph.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("m2_attr")

    # ---- 1. SAEs ----
    from sae_lens import SAE, HookedSAETransformer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    src_id = args.src_sae_id or f"blocks.{args.src_layer}.hook_resid_post"
    dst_id = args.dst_sae_id or f"blocks.{args.dst_layer}.hook_resid_post"

    log.info(f"loading SAE_src {args.sae_release} {src_id}")
    sae_src = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.sae_release, sae_id=src_id, device=args.device,
    )[0].to(dtype)
    log.info(f"loading SAE_dst {args.sae_release} {dst_id}")
    sae_dst = SAE.from_pretrained_with_cfg_and_sparsity(
        release=args.sae_release, sae_id=dst_id, device=args.device,
    )[0].to(dtype)
    if args.fold_norm_scale_src:
        sae_src.fold_activation_norm_scaling_factor(args.fold_norm_scale_src)
        log.info(f"folded norm scale {args.fold_norm_scale_src} into SAE_src")
    if args.fold_norm_scale_dst:
        sae_dst.fold_activation_norm_scaling_factor(args.fold_norm_scale_dst)
        log.info(f"folded norm scale {args.fold_norm_scale_dst} into SAE_dst")
    src_hook = sae_src.cfg.metadata["hook_name"]
    dst_hook = sae_dst.cfg.metadata["hook_name"]
    d_sae_src = sae_src.cfg.d_sae
    d_sae_dst = sae_dst.cfg.d_sae
    log.info(f"d_sae_src={d_sae_src}  d_sae_dst={d_sae_dst}")

    # ---- 2. Backbone ----
    # --hf-model-path lets us load a fine-tuned checkpoint (SFT/GRPO) into the
    # SAME base architecture (args.backbone) so the base-trained SAE still applies.
    # This is what makes the base-vs-SFT-vs-GRPO baseline-difference test possible.
    if args.hf_model_path:
        log.info(f"loading fine-tuned weights from {args.hf_model_path} into {args.backbone}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os as _os
        is_lora = _os.path.exists(_os.path.join(args.hf_model_path, "adapter_config.json"))
        if is_lora:
            # PEFT LoRA adapter (e.g. SFT checkpoint): load base + merge adapter.
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
            args.backbone, hf_model=hf_model, tokenizer=hf_tok,
            dtype=dtype, device=args.device,
        )
        del hf_model
    else:
        log.info(f"loading backbone {args.backbone}")
        model = HookedSAETransformer.from_pretrained_no_processing(
            args.backbone, dtype=dtype, device=args.device,
        )
    model.eval()

    # ---- 3. Shaped candidate features ----
    shaped_features = None
    if args.shaped_features:
        shaped_path = Path(args.shaped_features)
        if shaped_path.exists():
            blob = json.loads(shaped_path.read_text())
            shaped_features = blob.get("selected", [])
            log.info(f"loaded {len(shaped_features)} shaped candidate features")
        else:
            log.warning(f"shaped-features path not found: {shaped_path}")

    # ---- 4. probe prompts ----
    prompts = []
    if args.dataset == "bfcl":
        # RL task distribution: verl BFCL-v4 parquet, rendered through the
        # checkpoint tokenizer's chat template exactly as verl does for rollouts.
        log.info(f"loading BFCL parquet {args.bfcl_parquet} for {args.n_prompts} prompts")
        import pandas as pd
        df = pd.read_parquet(args.bfcl_parquet)
        for _, row in df.iterrows():
            msgs = [dict(m) for m in row["prompt"]]
            text = model.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
            if len(prompts) >= args.n_prompts:
                break
    else:
        log.info(f"loading pile-10k for {args.n_prompts} prompts")
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        for r in ds:
            text = r.get("text") or ""
            if len(text) < 200:
                continue
            prompts.append(text[:2000])
            if len(prompts) >= args.n_prompts:
                break
    log.info(f"got {len(prompts)} prompts from {args.dataset}")

    # ---- 4b. Optional matched RANDOM-SAE control (2602.14111 sanity check) ----
    # Re-runs the identical pipeline with random directions of matched d_sae and
    # matched L0. If the headline effect survives this, it is not an SAE finding.
    enc_src, enc_dst = sae_src.encode, sae_dst.encode
    if args.random_sae != "none":
        from random_sae_baseline import RandomSAEShim, calibrate_l0
        log.info(f"[random-sae] building {args.random_sae} control "
                 f"(seed={args.random_sae_seed})")
        cal_src, cal_dst = [], []
        for st in stream_states(model, prompts[:32], args.seq_len, src_hook, dst_hook,
                                args.src_layer, args.dst_layer,
                                keep_side=args.keep_side, states_per_prompt=1):
            cal_src.append(st["h_src"]); cal_dst.append(st["h_dst"])
        k_src = calibrate_l0(sae_src, cal_src)
        k_dst = calibrate_l0(sae_dst, cal_dst)
        shim_src = RandomSAEShim(sae_src, args.random_sae, args.random_sae_seed, k_src)
        shim_dst = RandomSAEShim(sae_dst, args.random_sae,
                                 args.random_sae_seed + 10_000, k_dst)
        enc_src, enc_dst = shim_src.encode, shim_dst.encode
        log.info(f"[random-sae] ACTIVE — matched L0 src={k_src} dst={k_dst}")

    # ---- 5. Build sparse attribution_fn closure ----
    def attr_fn_sparse(state, src_layer, dst_layer):
        return linear_probe_attribution_sparse(
            state, src_layer, dst_layer,
            sae_src=enc_src, sae_dst=enc_dst,
            h_src=state["h_src"], h_dst=state["h_dst"],
        )

    # ---- 6. Run build_graph_from_sparse_fn over the streamed states ----
    log.info("computing attribution graph (sparse path)...")
    t0 = time.time()
    _states = stream_states(model, prompts, args.seq_len, src_hook, dst_hook,
                            args.src_layer, args.dst_layer,
                            keep_side=args.keep_side,
                            states_per_prompt=args.states_per_prompt)
    dump = None
    if args.dump_states:
        dump = {"h_src": [], "h_dst": [], "prompt_idx": [], "pos": [],
                "zsum": None, "zfire": None}

        def _tap(it):
            # Pass-through: yields the identical state dicts, in order, so the graph
            # is unchanged. Records the states and the dst-SAE code statistics.
            for st in it:
                with torch.no_grad():
                    z = enc_dst(st["h_dst"].unsqueeze(0)).squeeze(0).detach().double().cpu().numpy()
                if dump["zsum"] is None:
                    dump["zsum"] = np.zeros_like(z); dump["zfire"] = np.zeros(z.shape, np.int64)
                dump["zsum"] += z; dump["zfire"] += (z > 0)
                dump["h_src"].append(st["h_src"].detach().float().cpu().numpy())
                dump["h_dst"].append(st["h_dst"].detach().float().cpu().numpy())
                dump["prompt_idx"].append(st["prompt_idx"]); dump["pos"].append(st["pos"])
                yield st
        _states = _tap(_states)
    graph = build_graph_from_sparse_fn(
        states_iter=_states,
        attribution_fn_sparse=attr_fn_sparse,
        src_layer=args.src_layer, dst_layer=args.dst_layer,
        keep_top_n=args.keep_top_n,
        per_state_top_k=args.per_state_top_k,
    )
    elapsed = time.time() - t0
    log.info(f"graph: edges={len(graph.weights)} d_sae_src={graph.d_sae_src} "
             f"d_sae_dst={graph.d_sae_dst} elapsed={elapsed:.1f}s")

    # ---- 7. Restrict / annotate the shaped-feature subgraph ----
    # Shaped's M1 features are at the dst SAE layer (they were selected via
    # function-word-outcome attribution at layer 4 in the original run). For
    # an attribution graph L_src→L_dst we therefore check `edges_dst` (not
    # edges_src) against the shaped set. Also report the src overlap as a
    # sanity check — if shaped features happen to span both layers it'll be
    # nonzero, otherwise expected 0.
    shaped_set = set(shaped_features) if shaped_features else None
    in_shaped_dst = []
    in_shaped_src = []
    if shaped_set:
        for s, d, _ in zip(graph.edges_src, graph.edges_dst, graph.weights):
            in_shaped_dst.append(int(d) in shaped_set)
            in_shaped_src.append(int(s) in shaped_set)
        in_shaped_dst = np.array(in_shaped_dst, dtype=bool)
        in_shaped_src = np.array(in_shaped_src, dtype=bool)
        log.info(f"edges to shaped-32-features (dst): {in_shaped_dst.sum()}/{len(graph.weights)}")
        log.info(f"edges from shaped-32-features (src): {in_shaped_src.sum()}/{len(graph.weights)}")

    # ---- 8. Dump ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone": args.backbone,
        "sae_release": args.sae_release,
        "src_layer": args.src_layer,
        "dst_layer": args.dst_layer,
        "src_hook": src_hook,
        "dst_hook": dst_hook,
        "d_sae_src": int(graph.d_sae_src),
        "d_sae_dst": int(graph.d_sae_dst),
        "n_prompts": len(prompts),
        "seq_len": args.seq_len,
        "dataset": args.dataset,
        "prompt_source": args.bfcl_parquet if args.dataset == "bfcl" else "NeelNanda/pile-10k",
        "keep_side": args.keep_side,
        # WHICH CHECKPOINT produced this graph. Without it a graph JSON cannot be told
        # apart from another run of the same protocol over different weights, which is
        # exactly the provenance gap that made the seeded-SFT experiment unverifiable
        # from the receipts alone. None == the unmodified base backbone.
        "hf_model_path": args.hf_model_path,
        "checkpoint": (Path(args.hf_model_path).name
                       if args.hf_model_path else f"{args.backbone} (base, no adapter)"),
        "states_per_prompt": args.states_per_prompt,
        "random_sae": args.random_sae,
        "random_sae_seed": args.random_sae_seed if args.random_sae != "none" else None,
        "elapsed_seconds": elapsed,
        "edges_src": graph.edges_src.tolist(),
        "edges_dst": graph.edges_dst.tolist(),
        "weights": graph.weights.tolist(),
        "shaped_features": shaped_features,
        "edges_to_shaped_dst": in_shaped_dst.tolist() if shaped_set else None,
        "edges_from_shaped_src": in_shaped_src.tolist() if shaped_set else None,
        "n_edges_to_shaped_dst": int(in_shaped_dst.sum()) if shaped_set else None,
        "n_edges_from_shaped_src": int(in_shaped_src.sum()) if shaped_set else None,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(f"saved → {out_path}")
    if dump is not None:
        n = len(dump["h_dst"])
        dp = Path(args.dump_states); dp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dp, h_src=np.stack(dump["h_src"]), h_dst=np.stack(dump["h_dst"]),
                 prompt_idx=np.asarray(dump["prompt_idx"]), pos=np.asarray(dump["pos"]),
                 zdst_mean=dump["zsum"] / max(n, 1), zdst_fire=dump["zfire"],
                 n_states=np.asarray(n), graph_out=np.asarray(str(out_path)),
                 hf_model_path=np.asarray(str(args.hf_model_path)))
        log.info(f"dumped {n} states -> {dp}")


if __name__ == "__main__":
    main()
