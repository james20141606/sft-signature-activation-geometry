"""Name the SFT-rewired features: max-activating contexts for chosen SAE features.

For each requested (layer, feature_id) pair, stream the pinned pile probe set
through the base model, encode with the base-trained SAE, and record the top-k
activating (prompt, token) contexts. Used to interpret the features involved in
the SFT rewiring signature (dst hub #30329 @ l22; top dropped-src #20227,
#16308 @ l14; plus the top-5 dropped dst features).

Usage:
    python scripts/feature_interp.py \
        --backbone Qwen/Qwen3-1.7B --sae-release qwen-scope-3-1.7b-base-w32k-l50 \
        --spec 14:20227 14:16308 22:30329 22:3477 22:10558 22:9970 \
        --n-prompts 500 --seq-len 64 --topk 12 \
        --out notes/sft_rewired_feature_interp.json
"""
from __future__ import annotations

import argparse
import heapq
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--sae-release", required=True)
    p.add_argument("--spec", nargs="+", required=True, help="LAYER:FEATID entries")
    p.add_argument("--n-prompts", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("interp")

    from sae_lens import SAE, HookedSAETransformer

    # group features by layer
    by_layer: dict[int, list[int]] = defaultdict(list)
    for s in args.spec:
        l, f = s.split(":")
        by_layer[int(l)].append(int(f))

    saes = {}
    for layer in by_layer:
        log.info(f"loading SAE layer{layer}")
        saes[layer] = SAE.from_pretrained_with_cfg_and_sparsity(
            release=args.sae_release, sae_id=f"layer{layer}", device=args.device,
        )[0].to(torch.bfloat16)

    model = HookedSAETransformer.from_pretrained_no_processing(
        args.backbone, dtype=torch.bfloat16, device=args.device,
    )
    model.eval()
    hooks = {l: saes[l].cfg.metadata["hook_name"] for l in by_layer}

    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    prompts = []
    for r in ds:
        t = r.get("text") or ""
        if len(t) < 200:
            continue
        prompts.append(t[:2000])
        if len(prompts) >= args.n_prompts:
            break
    log.info(f"{len(prompts)} prompts")

    # per (layer, feat): heap of (act, prompt_idx, pos, token_str, context_str)
    tops: dict[tuple, list] = {(l, f): [] for l in by_layer for f in by_layer[l]}
    act_stats = {(l, f): {"fire": 0, "sum": 0.0} for l in by_layer for f in by_layer[l]}
    n_tok = 0
    with torch.no_grad():
        for pi, text in enumerate(prompts):
            tokens = model.to_tokens(text)[:, : args.seq_len]
            _, cache = model.run_with_cache(tokens, names_filter=list(hooks.values()))
            str_toks = model.to_str_tokens(tokens[0])
            n_tok += tokens.shape[1]
            for l, feats in by_layer.items():
                acts = saes[l].encode(cache[hooks[l]][0]).float()  # [seq, d_sae]
                for f in feats:
                    col = acts[:, f]
                    nz = (col > 1e-6)
                    act_stats[(l, f)]["fire"] += int(nz.sum().item())
                    act_stats[(l, f)]["sum"] += float(col.sum().item())
                    v, idx = col.max(dim=0)
                    v = float(v.item()); pos = int(idx.item())
                    if v > 1e-6:
                        lo, hi = max(0, pos - 8), min(len(str_toks), pos + 4)
                        ctx = "".join(str_toks[lo:pos]) + "[[" + str_toks[pos] + "]]" + "".join(str_toks[pos + 1:hi])
                        h = tops[(l, f)]
                        item = (v, pi, pos, str_toks[pos], ctx)
                        if len(h) < args.topk:
                            heapq.heappush(h, item)
                        elif v > h[0][0]:
                            heapq.heapreplace(h, item)
            if (pi + 1) % 100 == 0:
                log.info(f"{pi+1}/{len(prompts)}")

    out = {}
    for (l, f), h in tops.items():
        items = sorted(h, key=lambda x: -x[0])
        st = act_stats[(l, f)]
        out[f"l{l}_f{f}"] = {
            "layer": l, "feature": f,
            "firing_rate": round(st["fire"] / max(1, n_tok), 5),
            "mean_act_overall": round(st["sum"] / max(1, n_tok), 5),
            "top_contexts": [
                {"act": round(v, 3), "prompt": pi, "pos": pos, "token": tok, "context": ctx}
                for v, pi, pos, tok, ctx in items
            ],
        }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
