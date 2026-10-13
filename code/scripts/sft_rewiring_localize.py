"""Localize the SFT rewiring signature (the one positive from the seed-null audit).

base->SFT edge Jaccard (0.937) clears the GRPO-family seed band [0.961,0.966],
so SFT rewires the attribution graph beyond RL seed noise. This script asks WHERE:

  1. Build the GRPO-family CONSENSUS edge set (edges present in >= --consensus of
     the 6 GRPO-family graphs). Edges that churn across seeds are noise; consensus
     edges are the stable post-RL circuit.
  2. SFT-DROPPED edges: present in base AND in the GRPO consensus, absent in SFT.
     These are robustly-present edges (base kept them, every RL seed kept them)
     that SFT specifically removes -> the localized SFT rewiring signature.
  3. SFT-ADDED edges: present in SFT, absent in base AND absent from all GRPO runs.
  4. Localization: how concentrated are dropped/added edges on dst (and src)
     features? Report top features by rewired-edge count + weight, and the share
     of rewired edge weight on the top-8 features (localized vs diffuse verdict).

Zero-GPU: runs on the committed graph JSONs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_edges(path):
    g = json.loads(Path(path).read_text())
    ew = {}
    for s, d, x in zip(g["edges_src"], g["edges_dst"], np.abs(np.asarray(g["weights"], float)).tolist()):
        ew[(int(s), int(d))] = ew.get((int(s), int(d)), 0.0) + x
    return ew


def top_feats(edges, weights, axis, k=10):
    cnt = defaultdict(int); wsum = defaultdict(float)
    for (s, d), w in zip(edges, weights):
        f = d if axis == "dst" else s
        cnt[f] += 1; wsum[f] += w
    order = sorted(wsum, key=wsum.get, reverse=True)
    tot = sum(wsum.values())
    top8_share = sum(wsum[f] for f in order[:8]) / tot if tot > 0 else 0.0
    return [{"feature": f, "n_edges": cnt[f], "weight": round(wsum[f], 1)} for f in order[:k]], round(top8_share, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--grpo-family", nargs="+", required=True)
    ap.add_argument("--consensus", type=int, default=None, help="min # of GRPO graphs containing an edge (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load_edges(args.base)
    sft = load_edges(args.sft)
    fam = [load_edges(p) for p in args.grpo_family]
    need = args.consensus or len(fam)

    # GRPO consensus edge set
    count = defaultdict(int)
    for e in fam:
        for k in e:
            count[k] += 1
    consensus = {k for k, c in count.items() if c >= need}
    print(f"graphs: base={len(base)} sft={len(sft)} grpo-family={[len(e) for e in fam]}")
    print(f"GRPO consensus (edge in >= {need}/{len(fam)} runs): {len(consensus)} edges")

    # SFT-dropped: in base AND consensus, NOT in SFT (weight from base graph)
    dropped = [k for k in base if k in consensus and k not in sft]
    dropped_w = [base[k] for k in dropped]
    # SFT-added: in SFT, not in base, not in ANY GRPO run
    any_grpo = set(count)
    added = [k for k in sft if k not in base and k not in any_grpo]
    added_w = [sft[k] for k in added]

    print(f"\nSFT-DROPPED (in base & GRPO-consensus, absent in SFT): {len(dropped)} edges, "
          f"total |w|={sum(dropped_w):.0f} ({100*sum(dropped_w)/sum(base.values()):.2f}% of base weight)")
    dtop, dshare = top_feats(dropped, dropped_w, "dst")
    print(f"  dst-localization: top-8 dst features hold {100*dshare:.1f}% of dropped weight")
    for r in dtop:
        print(f"    dst #{r['feature']:>6}  n={r['n_edges']:>3}  w={r['weight']}")
    stop, sshare = top_feats(dropped, dropped_w, "src")
    print(f"  src-localization: top-8 src features hold {100*sshare:.1f}% of dropped weight")
    for r in stop[:5]:
        print(f"    src #{r['feature']:>6}  n={r['n_edges']:>3}  w={r['weight']}")

    print(f"\nSFT-ADDED (in SFT, absent in base and ALL GRPO runs): {len(added)} edges, "
          f"total |w|={sum(added_w):.0f} ({100*sum(added_w)/sum(sft.values()):.2f}% of SFT weight)")
    atop, ashare = top_feats(added, added_w, "dst")
    print(f"  dst-localization: top-8 dst features hold {100*ashare:.1f}% of added weight")
    for r in atop:
        print(f"    dst #{r['feature']:>6}  n={r['n_edges']:>3}  w={r['weight']}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "consensus_edges": len(consensus), "need": need,
            "dropped": {"n": len(dropped), "total_w": round(sum(dropped_w), 1),
                        "pct_of_base_w": round(100*sum(dropped_w)/sum(base.values()), 2),
                        "top_dst": dtop, "top8_dst_share": dshare,
                        "top_src": stop, "top8_src_share": sshare},
            "added": {"n": len(added), "total_w": round(sum(added_w), 1),
                      "pct_of_sft_w": round(100*sum(added_w)/sum(sft.values()), 2),
                      "top_dst": atop, "top8_dst_share": ashare},
        }, indent=2))
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
