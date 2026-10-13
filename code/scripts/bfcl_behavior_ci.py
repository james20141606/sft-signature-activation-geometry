#!/usr/bin/env python3
"""Stratified 95% intervals for the BFCL behavioral sanity check (corrects the n=80 reading).

The harness is the held-out BFCL validation split built by prepare_bfcl.py: each of the four
single-turn categories (simple_python, multiple, parallel, parallel_multiple) shuffled with
seed 0 and split 90/10, i.e. 40/20/20/20 = 100 items. verl reports one accuracy per category
(val-core/<category>/acc/mean@1); the reported accuracy is their UNWEIGHTED mean
(mine_behavioral_deltas.py: sum(accs)/len(accs)). An earlier write-up assumed 4 x 20 = 80 items.

For an unweighted mean of independent category proportions,
    Var(acc) = (1/16) * sum_c p_c (1 - p_c) / n_c,   n_c = 40, 20, 20, 20,
and a normal-approximation 95% interval acc +- 1.96 SE. An arm without per-category values uses
the conservative bound p_c (1 - p_c) <= 1/4. Differences vs base are unpaired (no per-item
records exist), hence conservative. Point estimates are the receipt's values, unchanged.

Reads only notes/behavioral_calibration.json; writes notes/bfcl_behavior_ci.json.
    python3 code/scripts/bfcl_behavior_ci.py
"""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATS = ["simple_python", "multiple", "parallel", "parallel_multiple"]
N_C = [40, 20, 20, 20]
Z = 1.96


def var_from_cats(ps):
    return sum(p * (1 - p) / n for p, n in zip(ps, N_C)) / len(N_C) ** 2


def var_bound():
    return sum(0.25 / n for n in N_C) / len(N_C) ** 2


def arm(acc, ps=None):
    if ps is not None:
        assert len(ps) == 4
        mean = sum(ps) / 4
        assert abs(mean - acc) < 1e-3, (mean, acc)   # acc is the unweighted category mean
        v, how = var_from_cats(ps), "per-category p_c"
    else:
        v, how = var_bound(), "conservative bound p_c(1-p_c)<=1/4 (no per-category values)"
    se = math.sqrt(v)
    return {"acc": acc, "cats": ps, "var": v, "se": se,
            "ci95": [acc - Z * se, min(1.0, acc + Z * se)], "variance_from": how}


def diff(a, b):
    se = math.sqrt(a["var"] + b["var"])
    d = a["acc"] - b["acc"]
    return {"delta": d, "se": se, "ci95": [d - Z * se, d + Z * se],
            "excludes_zero": (d - Z * se) > 0 or (d + Z * se) < 0}


def main():
    src = json.load(open(os.path.join(ROOT, "notes", "behavioral_calibration.json")))
    # item counts implied by the per-category granularity: base simple_python 0.575 = 23/40
    for key in ("base", "sft"):
        for p, n in zip(src[key]["cats"], N_C):
            assert abs(p * n - round(p * n)) < 1e-6, (key, p, n)
    base = arm(src["base"]["acc"], src["base"]["cats"])
    sft = arm(src["sft"]["acc"], src["sft"]["cats"])
    g_merge = arm(src["grpo_at100"]["acc_merge_eval"])
    g_train = arm(src["grpo_at100"]["acc_intraining"])
    out = {
        "source": "notes/behavioral_calibration.json (base/SFT JID 12408266; GRPO merge-eval JID 8439216, in-training log 8323921)",
        "harness": "held-out BFCL validation split, 100 items = 40/20/20/20 over simple_python/multiple/parallel/parallel_multiple; accuracy = unweighted mean of the four category accuracies",
        "correction": "supersedes n_items=80 / '4 categories x 20' in behavioral_calibration.json",
        "n_items": sum(N_C), "n_per_category": dict(zip(CATS, N_C)),
        "method": "stratified normal approximation, Var=(1/16) sum_c p_c(1-p_c)/n_c; unpaired differences",
        "base": base, "sft": sft, "grpo_merge_eval": g_merge, "grpo_intraining": g_train,
        "delta_sft_vs_base": diff(sft, base),
        "delta_grpo_merge_vs_base": diff(g_merge, base),
        "delta_grpo_intraining_vs_base": diff(g_train, base),
    }
    json.dump(out, open(os.path.join(ROOT, "notes", "bfcl_behavior_ci.json"), "w"), indent=2)
    for k in ("base", "sft", "grpo_merge_eval", "grpo_intraining"):
        a = out[k]
        print(f"{k:16s} acc {a['acc']:.4f}  SE {a['se']:.4f}  CI [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]  ({a['variance_from']})")
    for k in ("delta_sft_vs_base", "delta_grpo_merge_vs_base", "delta_grpo_intraining_vs_base"):
        a = out[k]
        print(f"{k:30s} {a['delta']:+.4f}  SE {a['se']:.4f}  CI [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}]  excludes 0: {a['excludes_zero']}")


if __name__ == "__main__":
    main()
