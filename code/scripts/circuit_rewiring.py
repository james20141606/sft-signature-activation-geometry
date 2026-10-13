"""Circuit-rewiring base-vs-GRPO test with a SEED-NOISE null.

The paper's negatives so far concern *levels* (concentration, L0, Gini are
base-inherited). This asks a *difference* question the field hasn't controlled:
does GRPO REWIRE the attribution graph (change which src feature attributes to
which dst feature) beyond what two independent GRPO seeds differ from each other?

A real RL signature requires: base->GRPO rewiring  >  seed->seed rewiring.
If base->GRPO looks like seed<->seed (pure noise), there is no structural RL
signature even at the edge level -> the negative generalizes to graph structure.

Metrics between two top-K attribution graphs G1,G2 (edge = (src,dst) pair):
  * edge-set Jaccard          : |E1 & E2| / |E1 | E2|  (1 = identical wiring)
  * weighted edge cosine       : cosine of the two |w| vectors over the union of
                                 edges (missing edge = 0 weight); captures both
                                 set overlap and weight reallocation
  * top-K dst-hub Jaccard      : overlap of the K highest-inbound-weight dst feats

Contrasts:
  base->SFT, base->GRPO, base->GRPO_s4/s5, base->SHAPED  (post-training deltas)
  GRPO<->s4, GRPO<->s5, s4<->s5                          (SEED NOISE null)
Verdict: signature iff base->GRPO is separated from the seed-noise band.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def load_edges(path):
    g = json.loads(Path(path).read_text())
    src = np.asarray(g["edges_src"]); dst = np.asarray(g["edges_dst"])
    w = np.abs(np.asarray(g["weights"], dtype=float))
    ew = {}
    for s, d, x in zip(src.tolist(), dst.tolist(), w.tolist()):
        ew[(s, d)] = ew.get((s, d), 0.0) + x   # collapse dup pairs by summing |w|
    return ew


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 1.0


def wcosine(e1: dict, e2: dict) -> float:
    keys = list(set(e1) | set(e2))
    v1 = np.array([e1.get(k, 0.0) for k in keys], dtype=np.float64)
    v2 = np.array([e2.get(k, 0.0) for k in keys], dtype=np.float64)
    n1 = float(np.sqrt(np.sum(v1 * v1)))
    n2 = float(np.sqrt(np.sum(v2 * v2)))
    dot = float(np.sum(v1 * v2))
    return dot / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0


def hub_set(e: dict, k: int = 32) -> set:
    inb = {}
    for (s, d), x in e.items():
        inb[d] = inb.get(d, 0.0) + x
    return set(sorted(inb, key=inb.get, reverse=True)[:k])


def compare(name, e1, e2, k):
    # Revision R1 (2026-10-07, audit F6): the 4-dp field is kept for the committed receipts'
    # format, and the unrounded value is stored beside it so z can be computed at full
    # precision (a 4-dp rounding moved the DPO@29 upper-end z from +0.03 to -0.01). Table 1
    # z values are now computed at full precision from graph JSONs (code/scripts/samearch.py).
    return {
        "pair": name,
        "edge_jaccard_full": jaccard(set(e1), set(e2)),
        "edge_jaccard": round(jaccard(set(e1), set(e2)), 4),
        "weighted_edge_cosine": round(wcosine(e1, e2), 4),
        "hub_jaccard": round(jaccard(hub_set(e1, k), hub_set(e2, k)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", required=True, help="NAME=path.json")
    ap.add_argument("--hub-k", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed-family", nargs="*", default=None,
                    help="explicit run NAMEs whose pairwise similarities form the "
                         "seed-noise null (default: names starting with GRPO, plus "
                         "SHAPED). Use this to null a non-GRPO algorithm, e.g. "
                         "--seed-family SFT SFT_s1 SFT_s2 SFT_s3")
    ap.add_argument("--focus", default="GRPO",
                    help="the post-training run scored against the null in the "
                         "verdict block (default GRPO)")
    args = ap.parse_args()

    G = {n.split("=", 1)[0]: load_edges(n.split("=", 1)[1]) for n in args.graphs}
    names = list(G)
    print("loaded:", {n: len(G[n]) for n in names})

    # post-training contrasts vs base
    post = [n for n in names if n != "base"]
    rows = []
    if "base" in G:
        for n in post:
            rows.append(compare(f"base->{n}", G["base"], G[n], args.hub_k))
    # seed-noise null: all pairs among GRPO-family runs (independent seeds/recipes)
    seedfam = (args.seed_family if args.seed_family
               else [n for n in names if n.startswith("GRPO") or n in ("SHAPED",)])
    seed_rows = [compare(f"{a}<->{b}", G[a], G[b], args.hub_k)
                 for a, b in combinations(seedfam, 2)]

    def band(rs, key):
        vals = [r[key] for r in rs]
        return (min(vals), max(vals)) if vals else (None, None)

    print("\n=== post-training contrasts (vs base) ===")
    for r in rows:
        print(f"{r['pair']:<16} edgeJacc={r['edge_jaccard']:.4f}  "
              f"wEdgeCos={r['weighted_edge_cosine']:.4f}  hubJacc={r['hub_jaccard']:.4f}")
    print("\n=== seed-noise null (GRPO-family pairwise) ===")
    for r in seed_rows:
        print(f"{r['pair']:<16} edgeJacc={r['edge_jaccard']:.4f}  "
              f"wEdgeCos={r['weighted_edge_cosine']:.4f}  hubJacc={r['hub_jaccard']:.4f}")

    verdict = {}
    for key in ("edge_jaccard", "weighted_edge_cosine"):
        bg = next((r[key] for r in rows if r["pair"] == f"base->{args.focus}"), None)
        lo, hi = band(seed_rows, key)
        # signature iff base->focus is MORE different (lower similarity) than the
        # worst seed-pair by a clear margin
        sig = (bg is not None and lo is not None and bg < lo - 0.02)
        vals = [r[key] for r in seed_rows]
        # The z here is only interpretable when the null SD exceeds the value's own
        # reporting precision (values are rounded to 4 dp in compare()); a null whose
        # SD is below 1e-4 is quantized and its z is a rounding artifact, so record
        # the SD alongside so the reader can check.
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
        mean = float(np.mean(vals)) if vals else None
        verdict[key] = {
            f"base_to_{args.focus}": bg,
            "seed_null_range": [lo, hi],
            "seed_null_n_pairs": len(vals),
            "seed_null_mean": None if mean is None else round(mean, 6),
            "seed_null_sd": None if sd is None else round(sd, 6),
            "z": (None if (sd is None or not sd or bg is None)
                  else round((bg - mean) / sd, 2)),
            "precision_adequate": (None if sd is None else bool(sd >= 1e-4)),
            "rl_signature_beyond_seed_noise": bool(sig),
        }
        print(f"\n[{key}] base->{args.focus}={bg}  seed-null=[{lo},{hi}] "
              f"mean={mean} sd={sd} z={verdict[key]['z']} "
              f"precision_adequate={verdict[key]['precision_adequate']}  "
              f"-> signature beyond seed noise: {sig}")

    if args.out:
        # Record which graph file each NAME resolved to. Without this the receipt
        # cannot be told apart from another run of the same names over different
        # checkpoints: circuit_rewiring_qwen_6seed.json and
        # circuit_rewiring_4seed_step300.json both report a pair called
        # "base->GRPO_s1" (0.9594 vs 0.9610) because s1 exists at step 200 and at
        # step 300, and neither file said which it used. The step lives only in the
        # path, so the path is the provenance.
        inputs = {n.split("=", 1)[0]: n.split("=", 1)[1] for n in args.graphs}
        Path(args.out).write_text(json.dumps(
            {"inputs": inputs, "hub_k": args.hub_k,
             "post_contrasts": rows, "seed_null": seed_rows, "verdict": verdict}, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
