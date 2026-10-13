#!/usr/bin/env python3
"""Registered full sweep of the differential-edge necessity test.

Design: PREREGISTRATION_functional_transfer.md + AMENDMENT 1, as corrected by
notes/PREREG_functional_transfer_full_20260924.md (committed before this script
produced any receipt). The pilot implementation is code/scripts/functional_transfer.py
(JID 15406147); this file keeps its intervention and statistic and changes only what
the 2026-09-24 correction lists:

  * MATCHED PAIRS. The pilot built D_SFT from the audited SFT run's graph
    (notes/basediff_sft_l14to22.json, adapter .../checkpoints/sft-qwen3-1p7b) but ran
    the "SFT" model from a different seeded run (checkpoints/sft-seed0). Here every
    configuration pairs a graph with the checkpoint that produced it:
      A (primary)      graph basediff_sft_l14to22.json       + the audited SFT adapter
      B (descriptive)  graph basediff_sft_seed0_l14to22.json + checkpoints/sft-seed0
      C (descriptive)  the pilot's own pairing (A's graph + sft-seed0), k = 5,000 only
  * HELD-OUT SET. All rows 5,000..9,999 of NeelNanda/pile-10k whose first 2,000
    characters tokenize (model.to_tokens, as the graph builder) to >= 64 tokens; the
    first 64 tokens are used. Resample partner of eligible prompt i = eligible prompt
    (i+1) mod n (the pilot's rule, now on a fixed-length set so no truncation varies).
  * FREQUENCY MATCHING. The pilot ranked features by mean |z| on ONE 64-token prompt,
    so every feature silent on that prompt was tied at zero. Here the rank is the base
    model's firing rate (tie-break: mean activation) over the graph-building prompts
    (the first 500 pile-10k rows of >= 200 characters, first 64 tokens), and R is drawn
    without replacement within +-200 ranks of each target feature.
  * MODEL LOADING. from_pretrained_no_processing, as the graph builder
    (code/scripts/m2_attribution_graph.py); the pilot used from_pretrained.
  * BATCHING (GPU utilisation policy): prompts run in batches; arithmetic per prompt
    is identical.

Per (configuration, k) one receipt notes/functional_transfer_full_<cfg>_k<k>.json in
the pilot's schema (results.<arm>.per_prompt.<set>), plus provenance: SHA-256 of every
input, script and reader, seed, SLURM job id, eligible-row hash.

  python3 code/scripts/functional_transfer_full.py --out-dir notes \
      --audited-adapter <path> --seed0-adapter <path> [--n-max 64 --tag smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
GRAPH_BASE = NOTES / "attribution_graph_v1_qwen3-1p7b_qwenscope_l14to22.json"
GRAPH_SFT_AUDITED = NOTES / "basediff_sft_l14to22.json"
GRAPH_SFT_SEED0 = NOTES / "basediff_sft_seed0_l14to22.json"
READER = ROOT / "code/scripts/functional_transfer_read.py"
PREREGS = [ROOT / "PREREGISTRATION_functional_transfer.md",
           NOTES / "PREREG_functional_transfer_full_20260924.md"]
SEED = 0
KS = (2000, 5000, 10000)
WINDOW = 200


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def topk_edges(path, k):
    d = json.load(open(path))
    w = d["weights"]
    idx = sorted(range(len(w)), key=lambda i: -w[i])[:k]
    return {(d["edges_src"][i], d["edges_dst"][i]) for i in idx}


def differential_sources(graph_sft, k):
    """Identical to the pilot's rule (functional_transfer.py)."""
    Eb, Es = topk_edges(GRAPH_BASE, k), topk_edges(graph_sft, k)
    d_sft = {s for (s, _) in (Es - Eb)}
    d_base = {s for (s, _) in (Eb - Es)}
    shared = d_sft & d_base
    return sorted(d_sft - shared), sorted(d_base - shared), len(Eb & Es) / len(Eb | Es)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--audited-adapter", required=True)
    ap.add_argument("--seed0-adapter", required=True)
    ap.add_argument("--sae-release", default="qwen-scope-3-1.7b-base-w32k-l50")
    ap.add_argument("--src-layer", type=int, default=14)
    ap.add_argument("--src-sae-id", default="layer14")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--holdout-start", type=int, default=5000)
    ap.add_argument("--holdout-end", type=int, default=10000)
    ap.add_argument("--calib-prompts", type=int, default=500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-max", type=int, default=0, help="smoke only: cap n (0 = all)")
    ap.add_argument("--tag", default="full", help="'full' or 'smoke' (file prefix)")
    ap.add_argument("--out-dir", default=str(NOTES))
    ap.add_argument("--count-only", action="store_true",
                    help="tokenize the held-out split, print n, run no forward pass")
    return ap.parse_args()


def main():
    a = parse_args()
    import numpy as np
    import torch
    from datasets import load_dataset
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16
    hook = f"blocks.{a.src_layer}.hook_resid_post"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- registered sets ----------------
    configs = {"A": (GRAPH_SFT_AUDITED, "audited", KS),
               "B": (GRAPH_SFT_SEED0, "seed0", KS),
               "C": (GRAPH_SFT_AUDITED, "seed0", (5000,))}
    dsets = {}
    for cfg, (g, _, ks) in configs.items():
        for k in ks:
            d_sft, d_base, jac = differential_sources(g, k)
            dsets[(cfg, k)] = {"D_SFT": d_sft, "D_base": d_base, "edge_jaccard": jac}
            print(f"[sets] {cfg} k={k}: edge Jaccard={jac:.4f} |D_SFT|={len(d_sft)} "
                  f"|D_base|={len(d_base)}", flush=True)

    # ---------------- base model + tokens ----------------
    def load_model(adapter=None):
        if adapter is None:
            return HookedTransformer.from_pretrained_no_processing(
                a.backbone, dtype=dt, device=dev)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(a.backbone, torch_dtype=dt)
        hf = PeftModel.from_pretrained(hf, adapter).merge_and_unload()
        m = HookedTransformer.from_pretrained_no_processing(
            a.backbone, hf_model=hf, dtype=dt, device=dev)
        del hf
        return m

    base = load_model()
    base.eval()
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    rows, toks = [], []
    for i in range(a.holdout_start, min(a.holdout_end, len(ds))):
        t = base.to_tokens((ds[i]["text"] or "")[:2000])
        if t.shape[1] >= a.seq_len:
            rows.append(i)
            toks.append(t[0, :a.seq_len].cpu())
    n_elig = len(rows)
    rows_sha = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    print(f"[heldout] rows {a.holdout_start}..{a.holdout_end - 1}: eligible n={n_elig} "
          f"(excluded {a.holdout_end - a.holdout_start - n_elig} with < {a.seq_len} tokens) "
          f"rows_sha256={rows_sha[:16]}", flush=True)
    if a.count_only:
        json.dump({"n_eligible": n_elig, "rows_sha256": rows_sha, "rows": rows,
                   "holdout": [a.holdout_start, a.holdout_end], "seq_len": a.seq_len},
                  open(out_dir / "functional_transfer_full_heldout_count.json", "w"))
        return 0
    if a.n_max:
        rows, toks = rows[:a.n_max], toks[:a.n_max]
    n = len(rows)
    T = torch.stack(toks)                                        # (n, seq_len)
    partner = torch.roll(torch.arange(n), -1)                    # i -> (i+1) mod n

    sae = SAE.from_pretrained_with_cfg_and_sparsity(
        release=a.sae_release, sae_id=a.src_sae_id, device=dev)[0].float()
    W = sae.W_dec.detach().float()                               # (d_sae, d_model)

    # ---------------- frequency calibration on the graph-building prompts -------------
    cal = []
    for r in ds:
        txt = r.get("text") or ""
        if len(txt) < 200:
            continue
        cal.append(txt[:2000])
        if len(cal) >= a.calib_prompts:
            break
    fire = torch.zeros(W.shape[0], device=dev)
    act = torch.zeros(W.shape[0], device=dev)
    ntok = 0
    with torch.no_grad():
        for i in range(0, len(cal), a.batch):
            batch = [base.to_tokens(c)[0, :a.seq_len] for c in cal[i:i + a.batch]]
            for tt in batch:                                     # variable length: one by one
                _, cache = base.run_with_cache(tt[None].to(dev), names_filter=[hook],
                                               stop_at_layer=a.src_layer + 1)
                z = sae.encode(cache[hook][0].float())
                fire += (z > 0).float().sum(0)
                act += z.sum(0)
                ntok += z.shape[0]
    fire, act = (fire / ntok).cpu().numpy(), (act / ntok).cpu().numpy()
    order = sorted(range(len(fire)), key=lambda f: (-fire[f], -act[f], f))
    rank = {f: i for i, f in enumerate(order)}
    print(f"[calib] {len(cal)} graph-building prompts, {ntok} tokens; "
          f"features ever firing: {int((fire > 0).sum())}", flush=True)

    def matched_random(target, excl, seed_key):
        rng = np.random.default_rng(seed_key)
        pool = [f for f in order if f not in excl]               # rank order
        picks, used = [], set()
        for f in target:
            # nearest pool position to f's rank
            r = rank[f]
            lo_i, hi_i = 0, len(pool)
            while lo_i < hi_i:
                mid = (lo_i + hi_i) // 2
                if rank[pool[mid]] < r:
                    lo_i = mid + 1
                else:
                    hi_i = mid
            lo, hi = max(0, lo_i - WINDOW), min(len(pool), lo_i + WINDOW + 1)
            cand = [pool[j] for j in range(lo, hi) if pool[j] not in used]
            pick = int(cand[rng.integers(len(cand))])
            used.add(pick)
            picks.append(pick)
        return sorted(picks)

    sets = {}                                                    # (cfg,k) -> name -> feats
    for (cfg, k), dd in dsets.items():
        excl = set(dd["D_SFT"]) | set(dd["D_base"])
        cfg_id = {"A": 1, "B": 2, "C": 1}[cfg]                   # C shares A's graph -> A's R
        sets[(cfg, k)] = {
            "D_SFT": dd["D_SFT"], "D_base": dd["D_base"],
            "R_SFT": matched_random(dd["D_SFT"], excl, [SEED, cfg_id, k, 1]),
            "R_base": matched_random(dd["D_base"], excl, [SEED, cfg_id, k, 2])}
    if sets[("C", 5000)] != sets[("A", 5000)]:
        raise SystemExit("[FATAL] config C must reuse A's k=5000 sets")

    # which unique feature sets each model must evaluate
    need = {"base": set(), "audited": set(), "seed0": set()}
    for (cfg, k), ss in sets.items():
        adapter_key = configs[cfg][1]
        for name, feats in ss.items():
            need["base"].add(tuple(feats))
            need[adapter_key].add(tuple(feats))

    adapters = {"base": None, "audited": a.audited_adapter, "seed0": a.seed0_adapter}
    kl, top1 = {}, {}                                            # (model, featset) -> list
    busy = 0.0
    for mkey in ("base", "audited", "seed0"):
        m = base if mkey == "base" else load_model(adapters[mkey])
        m.eval()
        featsets = sorted(need[mkey], key=lambda s: (len(s), s))
        idx_t = {fs: torch.tensor(fs, device=dev, dtype=torch.long) for fs in featsets}
        for fs in featsets:
            kl[(mkey, fs)], top1[(mkey, fs)] = [], []
        tb = time.time()
        with torch.no_grad():
            for s in range(0, n, a.batch):
                ids = torch.arange(s, min(n, s + a.batch))
                tc, tr = T[ids].to(dev), T[partner[ids]].to(dev)
                logit_c, cache_c = m.run_with_cache(tc, names_filter=[hook])
                _, cache_r = m.run_with_cache(tr, names_filter=[hook],
                                              stop_at_layer=a.src_layer + 1)
                z_c = sae.encode(cache_c[hook].float())          # (b, T, d_sae)
                z_r = sae.encode(cache_r[hook].float())
                lp_c = torch.log_softmax(logit_c[:, -1].float(), -1)
                arg_c = lp_c.argmax(-1)
                del logit_c, cache_c, cache_r
                for fs in featsets:
                    ix = idx_t[fs]
                    dh = ((z_r[..., ix] - z_c[..., ix]) @ W[ix]).to(dt)

                    def patch(x, hook=None, _dh=dh):
                        return x + _dh

                    lg = m.run_with_hooks(tc, fwd_hooks=[(hook, patch)])
                    lp_p = torch.log_softmax(lg[:, -1].float(), -1)
                    klv = (lp_c.exp() * (lp_c - lp_p)).sum(-1)
                    kl[(mkey, fs)].extend(klv.tolist())
                    top1[(mkey, fs)].extend((lp_p.argmax(-1) == arg_c).int().tolist())
                    del lg, lp_p, dh
                del z_c, z_r
        torch.cuda.synchronize() if dev == "cuda" else None
        busy += time.time() - tb
        print(f"[{mkey}] {len(featsets)} feature sets x {n} prompts in "
              f"{time.time() - tb:.1f}s", flush=True)
        if mkey != "base":
            del m
            torch.cuda.empty_cache() if dev == "cuda" else None

    # ---------------- receipts ----------------
    import sae_lens, transformers, transformer_lens  # noqa: E401
    shas = {"script": sha(__file__), "reader": sha(READER),
            "graph_base": sha(GRAPH_BASE),
            "graph_sft_audited": sha(GRAPH_SFT_AUDITED), "graph_sft_seed0": sha(GRAPH_SFT_SEED0),
            "adapter_audited_safetensors": sha(Path(a.audited_adapter) / "adapter_model.safetensors"),
            "adapter_seed0_safetensors": sha(Path(a.seed0_adapter) / "adapter_model.safetensors")}
    for p in PREREGS:
        if p.exists():
            shas[p.name] = sha(p)
    written = []
    for (cfg, k), ss in sets.items():
        g, akey, _ = configs[cfg]
        res = {}
        for arm, mkey in (("base", "base"), ("SFT", akey)):
            pp = {name: kl[(mkey, tuple(f))] for name, f in ss.items()}
            res[arm] = {"means": {nm: float(np.mean(v)) for nm, v in pp.items()},
                        "per_prompt": pp,
                        "top1_agree": {nm: float(np.mean(top1[(mkey, tuple(f))]))
                                       for nm, f in ss.items()}}
        rec = {"k": k, "n_prompts": n, "config": cfg,
               "config_role": {"A": "primary (registered decision)",
                               "B": "descriptive (second matched SFT run)",
                               "C": "descriptive (the pilot's mismatched pairing)"}[cfg],
               "tag": a.tag,
               "graph_base": str(GRAPH_BASE.relative_to(ROOT)),
               "graph_sft": str(Path(g).relative_to(ROOT)),
               "sft_adapter": adapters[akey],
               "holdout": {"dataset": "NeelNanda/pile-10k", "rows": [a.holdout_start, a.holdout_end],
                           "rule": ">= seq_len tokens in the first 2000 chars; first seq_len tokens",
                           "n_eligible": n_elig, "rows_sha256": rows_sha,
                           "resample_partner": "(i+1) mod n over eligible prompts"},
               "seq_len": a.seq_len, "seed": SEED, "src_layer": a.src_layer,
               "sae_release": a.sae_release, "freq_match": {
                   "calib": f"first {len(cal)} pile-10k rows with >= 200 chars (graph prompts)",
                   "rank": "base firing rate desc, mean activation desc", "window": WINDOW},
               "prereg": "PREREGISTRATION_functional_transfer.md (+AMENDMENT 1) + "
                         "notes/PREREG_functional_transfer_full_20260924.md",
               "set_sizes": {nm: len(f) for nm, f in ss.items()},
               "sets": ss, "edge_jaccard": dsets[(cfg, k)]["edge_jaccard"],
               "results": res, "sha256": shas,
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "sae_lens": sae_lens.__version__},
               "timing_s": {"total": time.time() - t0, "forward_loops": busy}}
        fn = out_dir / f"functional_transfer_{a.tag}_{cfg}_k{k}.json"
        json.dump(rec, open(fn, "w"), indent=1)
        written.append(str(fn))
        print(f"[SAVED] {fn.name}  (means written; read only through the reader)", flush=True)
    print(f"[DONE] {len(written)} receipts, {time.time() - t0:.0f}s total", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
