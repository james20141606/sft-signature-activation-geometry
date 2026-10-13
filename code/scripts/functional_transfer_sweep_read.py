#!/usr/bin/env python3
"""Mechanical verdict for the registered full functional-transfer sweep
(notes/PREREG_functional_transfer_full_20260924.md). Committed before any full receipt.

Reads ONLY the primary configuration A at k in {2000, 5000, 10000}; refuses partial
input (a missing k, a smoke receipt, a receipt whose held-out rows differ from the
committed count file, or a receipt from another configuration). Each k is classified
by functional_transfer_read.decide() -- the exhaustive per-k table. The sweep verdict:

  all three k in the same branch X   -> verdict X
  otherwise                          -> MIXED (per-k branches listed; no direction read)

Configurations B (second matched SFT run) and C (the pilot's mismatched pairing) are
printed afterwards, labelled DESCRIPTIVE; they never enter the verdict.

  python3 code/scripts/functional_transfer_sweep_read.py [--prefix full]
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_transfer_read import NEEDED, SENTENCE, decide, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
KS = (2000, 5000, 10000)
COUNT = NOTES / "functional_transfer_full_heldout_count.json"

VERDICT_SENTENCE = {
    "F1": "F1 at every k: the SFT-specific sources matter more in SFT than in base, relative "
          "to the base-specific ones.",
    "F2": "F2 at every k: load-bearing but not stage-specific.",
    "F3": "F3 at every k: only one own-condition cell passes the gate; no transfer claim.",
    "F4": "F4 at every k: the differential sources are not load-bearing beyond matched "
          "random sources; only the structural claim stands.",
    "F5": "F5 at every k: the interaction is reversed (SFT-specific sources matter less in "
          "SFT, relative to base).",
    "U": "U at every k: bootstrap and permutation disagree at every k; no direction read.",
    "MIXED": "MIXED: the branch differs across k; no stage-specificity claim is made.",
}


def ratios(d):
    r = d["results"]
    out = {}
    for arm, own, ctl in (("SFT", "D_SFT", "R_SFT"), ("base", "D_base", "R_base"),
                          ("base", "D_SFT", "R_SFT")):
        a, b = r[arm]["per_prompt"][own], r[arm]["per_prompt"][ctl]
        out[f"{arm}:{own}/{ctl}"] = {
            "mean_ratio": (sum(a) / len(a)) / (sum(b) / len(b)),
            "median_ratio": st.median(a) / st.median(b)}
    return out


def read_cfg(cfg, prefix, ks, count, strict, notes):
    rows = []
    for k in ks:
        p = notes / f"functional_transfer_{prefix}_{cfg}_k{k}.json"
        if not p.exists():
            if strict:
                print(f"REFUSED: primary receipt missing: {p.name}")
                sys.exit(2)
            print(f"  (descriptive {cfg} k={k}: receipt missing)")
            continue
        d = load(p)
        if strict:
            if d.get("config") != cfg or d.get("tag") != "full":
                print(f"REFUSED: {p.name} is config={d.get('config')} tag={d.get('tag')}")
                sys.exit(2)
            h = d["holdout"]
            if (h["rows_sha256"] != count["rows_sha256"]
                    or d["n_prompts"] != count["n_eligible"]):
                print(f"REFUSED: {p.name} held-out set (n={d['n_prompts']}, "
                      f"{h['rows_sha256'][:12]}) != committed count "
                      f"(n={count['n_eligible']}, {count['rows_sha256'][:12]})")
                sys.exit(2)
        res = decide({a: d["results"][a]["per_prompt"] for a in ("base", "SFT")})
        rows.append((k, d, res))
    return rows


def show(rows, label):
    for k, d, res in rows:
        m = res["means"]
        print(f"  [{label}] k={k:>5}  n={res['n']}  sizes={d['set_sizes']}  "
              f"Jaccard={d['edge_jaccard']:.4f}")
        for arm in ("base", "SFT"):
            print(f"      {arm:4s} " + "  ".join(f"{s}={m[arm][s]:.5f}" for s in NEEDED))
        for key, v in ratios(d).items():
            print(f"      ratio {key}: mean {v['mean_ratio']:.1f}x  median {v['median_ratio']:.1f}x")
        print(f"      gate SFT={res['gate']['SFT']} base={res['gate']['base']}", end="")
        if "J" in res:
            print(f"  J={res['J']:+.5f} CI [{res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}] "
                  f"p={res['p_perm']:.4f}", end="")
        print(f"  -> {res['branch']}")


def main(argv):
    prefix = "full"
    if "--prefix" in argv:
        prefix = argv[argv.index("--prefix") + 1]
    notes = NOTES
    if "--notes" in argv:                      # testing only: a directory of synthetic receipts
        notes = Path(argv[argv.index("--notes") + 1])
    cfile = notes / COUNT.name
    if not cfile.exists():
        print(f"REFUSED: committed held-out count file missing: {cfile}")
        return 2
    count = json.load(open(cfile))
    strict = prefix == "full"
    prim = read_cfg("A", prefix, KS, count, strict, notes)
    print("PRIMARY (configuration A: audited SFT graph + its own adapter)")
    show(prim, "A")
    branches = [res["branch"] for _, _, res in prim]
    verdict = branches[0] if len(set(branches)) == 1 else "MIXED"
    print(f"\n>>> SWEEP VERDICT: {verdict}  (per k: "
          + ", ".join(f"{k}:{b}" for (k, _, _), b in zip(prim, branches)) + ")")
    print(f">>> {VERDICT_SENTENCE[verdict]}")
    if verdict != "MIXED":
        print(f">>> per-k table text: {SENTENCE[verdict]}")
    print("\nDESCRIPTIVE ONLY (never enters the verdict)")
    show(read_cfg("B", prefix, KS, count, False, notes), "B")
    show(read_cfg("C", prefix, (5000,), count, False, notes), "C")
    if not strict:
        print("\n[smoke/non-full prefix: pipeline check only -- NOT a result]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
