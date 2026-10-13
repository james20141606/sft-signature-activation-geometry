#!/usr/bin/env python3
"""Score the matched-construction DPO null against the rule pre-registered in
PREREGISTRATION_dpo_matched_construction.md.

Written and committed BEFORE notes/circuit_rewiring_dpo_mc29.json exists, so the
scoring cannot be tuned to a result.

REFUSES PARTIAL INPUT: all four DPOMC_s{0..3} contrasts must be present in the
receipt, and the frozen SFT comparator receipt must contain all four SFT_s{0..3}
contrasts, else exit 2 with no number printed. The reader states the candidate
set it saw (the Addendum-45 lesson: a rule resolved over a silently truncated
domain reports the wrong governing quantity while looking healthy).

Branches, verbatim from the registration:
  MC1  disjointness survives   max(MC displacement) <  0.0588
  MC2  overlap                 any MC displacement  >= 0.0588
  MC3  uninformative           (< 4 receipts -- handled by the refusal above; a
                                partial roster is MC3, never a scored verdict)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MC = os.path.join(ROOT, "notes/circuit_rewiring_dpo_mc29.json")
SFT = os.path.join(ROOT, "notes/circuit_rewiring_sft_seednull.json")

# Frozen comparator endpoints, restated in the registration. Re-derived here from
# the committed receipt rather than trusted as prose.
SFT_PAIRS = ["base->SFT_s0", "base->SFT_s1", "base->SFT_s2", "base->SFT_s3"]
MC_PAIRS = ["base->DPOMC_s0", "base->DPOMC_s1", "base->DPOMC_s2", "base->DPOMC_s3"]
REG_SFT_MIN = 0.0588


def contrasts(path, wanted, tag):
    if not os.path.exists(path):
        print(f"REFUSED: {tag} receipt missing: {os.path.relpath(path, ROOT)} "
              f"-- a partial roster is MC3, not a verdict")
        sys.exit(2)
    d = json.load(open(path))
    got = {c["pair"]: c["edge_jaccard"] for c in d.get("post_contrasts", [])}
    missing = [p for p in wanted if p not in got]
    if missing:
        print(f"REFUSED: {tag} receipt lacks contrast(s) {missing}; present: "
              f"{sorted(got)}")
        sys.exit(2)
    return {p: 1.0 - got[p] for p in wanted}


def main():
    sft = contrasts(SFT, SFT_PAIRS, "SFT comparator")
    mc = contrasts(MC, MC_PAIRS, "matched-construction")

    sft_lo, sft_hi = min(sft.values()), max(sft.values())
    if abs(sft_lo - REG_SFT_MIN) > 5e-5:
        print(f"REFUSED: committed SFT receipt gives min displacement {sft_lo:.4f}, "
              f"registration froze {REG_SFT_MIN:.4f} -- resolve the provenance "
              "before any MC number is read")
        sys.exit(2)

    print(f"candidate set seen: {MC_PAIRS} (4 of 4), comparator {SFT_PAIRS} (4 of 4)")
    for p in MC_PAIRS:
        print(f"  {p:<16} displacement={mc[p]:.4f}")
    lo, hi = min(mc.values()), max(mc.values())
    print(f"  MC range [{lo:.4f}, {hi:.4f}]  width {hi - lo:.4f}")
    print(f"  SFT range [{sft_lo:.4f}, {sft_hi:.4f}]  (frozen comparator)")

    if hi < sft_lo:
        print("\n>>> MC1: DISJOINTNESS SURVIVES CONSTRUCTION MATCHING. The "
              "matched-construction DPO band, now carrying per-run data-order "
              "variance, remains disjoint from the SFT band at the identical 29 "
              "steps. The disclosed asymmetry closes; the anti-conservativeness "
              "concern retires.")
        return 0
    print("\n>>> MC2: OVERLAP. At least one matched-construction seed reaches the "
          "SFT range. Per the registration the disjointness lead is downgraded "
          "immediately: revert to the calibrated seed-null statement scoped to "
          "fixed-order nulls, report the overlap where the paper reports "
          "disjointness, rewrite the abstract's disjointness sentence. No re-run, "
          "no seed extension, no search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
