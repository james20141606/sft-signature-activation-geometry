#!/usr/bin/env python3
"""Score the Llama second-backbone SFT replication against the rule in
PREREGISTRATION_llama_second_backbone.md. Committed before any receipt exists.

REFUSES PARTIAL INPUT: 4 contrasts and 6 seed pairs or exit 2. States the
candidate set it saw (Addendum-45).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "notes/circuit_rewiring_llama_sft_seednull.json")
WANT = ["base->LSFT_s0", "base->LSFT_s1", "base->LSFT_s2", "base->LSFT_s3"]


def main():
    if not os.path.exists(P):
        print("REFUSED: receipt missing: notes/circuit_rewiring_llama_sft_seednull.json"
              " -- a partial roster is R4, not a verdict")
        return 2
    d = json.load(open(P))
    got = {c["pair"]: 1.0 - c["edge_jaccard"] for c in d.get("post_contrasts", [])}
    missing = [p for p in WANT if p not in got]
    pairs = [1.0 - x["edge_jaccard"] for x in d.get("seed_null", [])]
    if missing or len(pairs) != 6:
        print(f"REFUSED (R4): missing contrasts {missing}, seed pairs {len(pairs)}/6")
        return 2
    if not d.get("verdict", {}).get("edge_jaccard", {}).get("precision_adequate", False):
        print("R4 UNINFORMATIVE: edge_jaccard precision_adequate is false")
        return 0
    print(f"candidate set seen: {WANT} (4 of 4), seed pairs 6 of 6")
    for p in WANT:
        print(f"  {p:<14} displacement={got[p]:.4f}")
    band_hi = max(pairs)
    print(f"  seed-pair band max {band_hi:.4f} (pairs: "
          + ", ".join(f"{x:.4f}" for x in sorted(pairs)) + ")")
    n_out = sum(1 for p in WANT if got[p] > band_hi)
    if n_out == 4:
        print("\n>>> R1: THE CALIBRATED ESTIMATOR REPLICATES. Every base->SFT contrast "
              "exceeds the 6-pair seed band on a second backbone family and a second "
              "SAE family.")
    elif n_out == 0:
        print("\n>>> R3: NO SEPARATION. The SFT signature is not backbone-general at "
              "this recipe and scale. Reported as the finding, not buried.")
    else:
        print(f"\n>>> R2: PARTIAL ({n_out}/4 contrasts exceed the band). Reported as "
              "partial replication; no cherry-picking of passing seeds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
