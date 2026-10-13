"""Mine verl training logs for BFCL-v4 validation-accuracy curves of the audited
GRPO runs, and build the behavioral-vs-SAE-space calibration table.

For each audited run we extract, from its verl .out log, the mean over the four
val-core/bfcl_v4_*/acc/mean@1 categories at (a) the initial validation (before
step 1) and (b) the audited checkpoint step. Behavioral delta = (b) - (a).
Zero GPU: pure log mining.

Usage (on the cluster):
    python mine_behavioral_deltas.py \
        --log-dir $WORK/grpo_project/logs/grpo \
        --out notes/behavioral_deltas.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CATS = ["bfcl_v4_simple_python", "bfcl_v4_multiple", "bfcl_v4_parallel", "bfcl_v4_parallel_multiple"]

# audited run -> (log filename prefix, audited checkpoint step)
RUNS = {
    "GRPO":    ("shaped-grpo-baseline-sif", 100),
    "GRPO_s1": ("shaped-grpo-baseline-s1-sif", 200),
    "GRPO_s3": ("shaped-grpo-baseline-s3-sif", 100),
    "GRPO_s4": ("shaped-grpo-baseline-s4-sif", 300),
    "GRPO_s5": ("shaped-grpo-baseline-s5-sif", 300),
    "SHAPED":   ("shaped-grpo-SHAPED-sif", 150),
}


def parse_log(path: Path) -> dict[int, float]:
    """Return {step: mean_val_acc} including step 0 for the initial validation."""
    out = {}
    txt = path.read_text(errors="ignore")
    # initial validation line: "Initial validation metrics: {...}" (dict-ish)
    m = re.search(r"[Ii]nitial validation metrics[^\n]*", txt)
    if m:
        accs = [float(x) for c in CATS
                for x in re.findall(rf"val-core/{c}/acc/mean@1['\":\s]+([0-9.]+)", m.group(0))]
        if len(accs) == len(CATS):
            out[0] = sum(accs) / len(accs)
    # step lines
    for sm in re.finditer(r"step:(\d+) - .*", txt):
        line = sm.group(0)
        accs = []
        for c in CATS:
            am = re.search(rf"val-core/{c}/acc/mean@1:([0-9.]+)", line)
            if am:
                accs.append(float(am.group(1)))
        if len(accs) == len(CATS):
            out[int(sm.group(1))] = sum(accs) / len(accs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logdir = Path(args.log_dir)

    results = {}
    for run, (prefix, step) in RUNS.items():
        best = {}
        best_file = None
        for f in sorted(logdir.glob(f"{prefix}-*.out")):
            # exclude longer prefixes swallowing shorter ones (baseline-sif vs baseline-s1-sif)
            stem = f.name.rsplit("-", 1)[0]
            if stem != prefix:
                continue
            cur = parse_log(f)
            if len(cur) > len(best):
                best, best_file = cur, f.name
        if not best:
            results[run] = {"error": f"no parsable log for prefix {prefix}"}
            continue
        init = best.get(0)
        # value at audited step (or nearest step <= audited step)
        steps = sorted(s for s in best if s > 0)
        at = max((s for s in steps if s <= step), default=None)
        results[run] = {
            "log": best_file,
            "audited_step": step,
            "initial_val_acc": init,
            "val_acc_at_step": best.get(at) if at else None,
            "at_step_used": at,
            "behavioral_delta": (round(best[at] - init, 4)
                                 if (init is not None and at is not None) else None),
            "curve": {s: best[s] for s in sorted(best)},
        }
        print(f"{run:<8} log={best_file} init={init} at{at}={best.get(at)} "
              f"delta={results[run]['behavioral_delta']}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
