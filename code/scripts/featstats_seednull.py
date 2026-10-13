"""Seed-noise null for the activation-side metrics.

Generalizes the circuit_rewiring.py seed-null framework to the activation-side
battery: for each metric, is the base->GRPO delta within the band spanned by
pairwise differences among independent GRPO-family runs? If YES for every metric,
the negative is fully definitive: base->GRPO is indistinguishable from RL seed
noise on levels (concentration), activations (L0/Gini/entropy), intensity, AND
circuit structure.

Also directly tests the rl_circuit_intensity2026 foil claim (RL raises activation
intensity): if act_mass_per_token / mean_act_when_active base->GRPO is within seed
noise, that positive claim does not survive a matched-base seed-null control.

Usage:
    python scripts/featstats_seednull.py \
        --base notes/featstats_base_l22.json \
        --grpo notes/featstats_grpo_l22.json \
        --seed-family GRPO=notes/featstats_grpo_l22.json s1=... s3=... s4=... s5=... \
        --out notes/featstats_seednull.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

METRICS = ["mean_L0", "frac_dead", "firing_freq_gini", "firing_freq_top1pct_active",
           "meanact_gini", "usage_entropy_norm", "act_mass_per_token", "mean_act_when_active", "fvu"]


def load(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--grpo", required=True)
    ap.add_argument("--seed-family", nargs="+", required=True, help="NAME=path.json (>=2 GRPO-family runs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load(args.base)
    grpo = load(args.grpo)
    fam = {e.split("=", 1)[0]: load(e.split("=", 1)[1]) for e in args.seed_family}
    fam_names = list(fam)
    print(f"base={args.base}\nseed-family={fam_names}\n")

    hdr = f"{'metric':<26}{'base':>12}{'GRPO':>12}{'|dGRPO|':>10}{'seed-null band':>22}{'within?':>9}"
    print(hdr)
    rows = {}
    all_within = True
    for m in METRICS:
        if m not in base or m not in grpo:
            continue
        # A seed-null requires the metric in >=2 seed-family runs to form a pairwise
        # band. Metrics present only in base/grpo (e.g. fvu, added to the checkpoint
        # battery after the seed runs) are NOT seed-nullable and are excluded here;
        # fvu is reported separately as the SAE blind-spot bound, not in this battery.
        fam_with_m = [n for n in fam_names if m in fam[n]]
        if len(fam_with_m) < 2:
            continue
        d_grpo = abs(grpo[m] - base[m])
        # seed-null: pairwise |diff| among GRPO-family runs
        pair_diffs = [abs(fam[a][m] - fam[b][m]) for a, b in combinations(fam_with_m, 2)]
        band_max = max(pair_diffs) if pair_diffs else 0.0
        within = d_grpo <= band_max + 1e-9
        all_within = all_within and within
        rows[m] = {"base": base[m], "grpo": grpo[m], "abs_delta_grpo": round(d_grpo, 5),
                   "seed_null_max_pairwise": round(band_max, 5), "within_seed_noise": bool(within)}
        print(f"{m:<26}{base[m]:>12}{grpo[m]:>12}{d_grpo:>10.4f}{f'[0, {band_max:.4f}]':>22}{str(within):>9}")

    print(f"\nVERDICT: base->GRPO within seed noise for ALL activation metrics: {all_within}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"base": args.base, "seed_family": fam_names, "metrics": rows,
             "all_metrics_within_seed_noise": all_within}, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
