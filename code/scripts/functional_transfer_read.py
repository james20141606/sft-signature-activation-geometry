#!/usr/bin/env python3
"""Score the differential-edge necessity test against PREREGISTRATION_functional_transfer.md
(+AMENDMENT 1) and its 2026-09-24 correction notes/PREREG_functional_transfer_full_20260924.md.

REFUSES PARTIAL INPUT. Reads the FAITHFULNESS GATE before the decisive statistic:
each own-condition differential set must move the output more than its size- and
frequency-matched random control in the same model. If it does not, no J is read.

Branch table (EXHAUSTIVE; revised 2026-09-24 before any full-sweep receipt existed):

  F4 NOT-A-CIRCUIT   neither own-condition cell beats its matched random control
  F3 ASYMMETRIC      exactly one cell beats its matched random control
  -- gate passed in both cells; J with paired-prompt bootstrap 95% CI [lo, hi] and a
     two-sided within-prompt label-permutation p (B = 10,000 each, seed 0) --
  F1 STAGE-SPECIFIC  J > 0, lo > 0 and p < 0.05
  F5 REVERSED        J < 0, hi < 0 and p < 0.05       (added 2026-09-24; see below)
  F2 PHANTOM         lo <= 0 <= hi and p >= 0.05      (CI covers zero, permutation agrees)
  U  DISCORDANT      every remaining case: the CI and the permutation test disagree
                     (CI excludes zero but p >= 0.05, or CI covers zero but p < 0.05),
                     or the point estimate lies outside its own percentile CI

WHY THIS FILE CHANGED (2026-09-24). The committed version ended in a catch-all `else`
that printed "F2 PHANTOM" for every gate-passing outcome that was not F1. The pilot
(notes/functional_transfer_k5000.json: J = -0.02701, CI [-0.05544, -0.00120],
p = 0.0501) is NOT F2 as registered (F2 requires the CI to cover zero), yet the reader
labelled it F2 (ELN entry at ~L4290). A registered reader must name an outcome outside
its table, never default it into a branch. The table is now exhaustive, each branch
has an explicit condition, and classify() asserts that at most one condition holds.
Under this reader the pilot reads U (DISCORDANT).

The arithmetic is unchanged: J_of(idx) is the mean over idx of the per-prompt
interaction j_i, and the bootstrap and the permutation consume the same RNG stream in
the same order as the committed version, so the pilot's J, CI and p reproduce to the
printed digits (--selftest checks this).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
B = 10_000
SEED = 0
ALPHA = 0.05
NEEDED = ["D_SFT", "D_base", "R_SFT", "R_base"]
BRANCHES = ("F1", "F2", "F3", "F4", "F5", "U")

SENTENCE = {
    "F1": "F1 STAGE-SPECIFIC: the SFT-displaced sources matter more in the SFT model than "
          "the base-displaced sources do, relative to the base model.",
    "F2": "F2 PHANTOM: the differential sources are load-bearing but NOT stage-specific "
          "(CI covers zero, permutation agrees).",
    "F3": "F3 ASYMMETRIC: only one own-condition cell beats its matched random control -- "
          "extraction-quality asymmetry, not specialization. No transfer claim.",
    "F4": "F4 NOT-A-CIRCUIT: neither own-condition differential set moves the output more "
          "than its matched random control. Only the STRUCTURAL claim is retained.",
    "F5": "F5 REVERSED: the SFT-displaced sources matter LESS in the SFT model than the "
          "base-displaced sources do, relative to the base model -- the opposite of stage "
          "specificity.",
    "U": "U DISCORDANT: the bootstrap CI and the label-permutation test disagree; no "
         "direction is read.",
}


def load(path):
    p = Path(path)
    if not p.exists():
        print(f"REFUSED: receipt missing: {path}")
        sys.exit(2)
    d = json.load(open(p))
    for arm in ("base", "SFT"):
        if arm not in d.get("results", {}):
            print(f"REFUSED: receipt lacks the {arm!r} arm")
            sys.exit(2)
        for s in NEEDED:
            if s not in d["results"][arm]["per_prompt"]:
                print(f"REFUSED: {arm} arm lacks set {s!r}; present: "
                      f"{sorted(d['results'][arm]['per_prompt'])}")
                sys.exit(2)
    lens = {len(d["results"][a]["per_prompt"][s]) for a in ("base", "SFT") for s in NEEDED}
    if len(lens) != 1 or 0 in lens:
        print(f"REFUSED: per-prompt lists have unequal or zero lengths {sorted(lens)}")
        sys.exit(2)
    return d


def mean(v):
    return sum(v) / len(v)


def classify(gate_sft, gate_base, J=None, lo=None, hi=None, p=None):
    """The registered branch table as a pure function. Exhaustive by construction:
    F4/F3 on the gate, then at most one of F1/F5/F2, else U."""
    if not gate_sft and not gate_base:
        return "F4"
    if gate_sft != gate_base:
        return "F3"
    conds = {
        "F1": J > 0 and lo > 0 and p < ALPHA,
        "F5": J < 0 and hi < 0 and p < ALPHA,
        "F2": lo <= 0 <= hi and p >= ALPHA,
    }
    hits = [k for k, v in conds.items() if v]
    if len(hits) > 1:
        raise AssertionError(f"branch conditions overlap: {hits}")
    return hits[0] if hits else "U"


def decide(pp, b=B, seed=SEED):
    """pp[arm][set] -> per-prompt KL lists. Returns a dict with the branch and the
    statistics; the gate is read before J is computed."""
    n = len(pp["SFT"]["D_SFT"])
    out = {"n": n}
    out["means"] = {a: {s: mean(pp[a][s]) for s in NEEDED} for a in ("base", "SFT")}
    gate_sft = out["means"]["SFT"]["D_SFT"] > out["means"]["SFT"]["R_SFT"]
    gate_base = out["means"]["base"]["D_base"] > out["means"]["base"]["R_base"]
    out["gate"] = {"SFT": gate_sft, "base": gate_base}
    if not (gate_sft and gate_base):
        out["branch"] = classify(gate_sft, gate_base)
        return out

    # per-prompt interaction; J over an index multiset is the mean of j over it
    j = [((pp["SFT"]["D_SFT"][i] - pp["SFT"]["D_base"][i])
          - (pp["base"]["D_SFT"][i] - pp["base"]["D_base"][i])) for i in range(n)]
    J = mean(j)
    rng = random.Random(seed)
    boots = sorted(sum(j[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    lo, hi = boots[int(0.025 * b)], boots[int(0.975 * b) - 1]
    # within-prompt arm-label permutation == a random sign flip of j_i
    # (same rng.random() draw per prompt, in the same order, as the committed version)
    perm = []
    for _ in range(b):
        tot = 0.0
        for i in range(n):
            tot += j[i] if rng.random() < 0.5 else -j[i]
        perm.append(tot / n)
    p = (sum(1 for v in perm if abs(v) >= abs(J)) + 1) / (b + 1)
    out.update({"J": J, "ci": [lo, hi], "p_perm": p, "B": b, "seed": seed})
    out["branch"] = classify(True, True, J, lo, hi, p)
    return out


def report(d, res):
    print(f"candidate set seen: 2 arms x {NEEDED} over {res['n']} held-out prompts "
          f"(k={d['k']}, edge Jaccard {d['edge_jaccard']:.4f}, sizes {d['set_sizes']})")
    for arm in ("base", "SFT"):
        print(f"  {arm:5s} " + "  ".join(f"{s}={res['means'][arm][s]:.5f}" for s in NEEDED))
    print(f"  gate: SFT own-condition beats matched random = {res['gate']['SFT']}; "
          f"base = {res['gate']['base']}")
    if "J" in res:
        print(f"  J = {res['J']:+.5f}  95% CI [{res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}]  "
              f"(paired bootstrap, B={res['B']}, seed {res['seed']});  "
              f"label-permutation p = {res['p_perm']:.4f}")
    print(f"\n>>> {SENTENCE[res['branch']]}")


def selftest():
    """Every branch must be reachable and named, and the pilot must reproduce."""
    ok = total = 0
    # (1) the table itself, including the pilot-shaped case and boundaries
    table = [
        ((False, False), "F4"), ((True, False), "F3"), ((False, True), "F3"),
        ((True, True, +0.03, +0.01, +0.05, 0.001), "F1"),
        ((True, True, -0.03, -0.05, -0.01, 0.001), "F5"),
        ((True, True, +0.01, -0.01, +0.03, 0.40), "F2"),
        ((True, True, -0.027, -0.055, -0.001, 0.0501), "U"),   # the pilot's shape
        ((True, True, +0.03, +0.01, +0.05, 0.06), "U"),        # CI excl. 0, p >= .05
        ((True, True, +0.01, -0.01, +0.03, 0.01), "U"),        # CI covers 0, p < .05
        ((True, True, 0.0, 0.0, 0.0, 1.0), "F2"),              # degenerate: all zero
        ((True, True, -0.01, +0.01, +0.05, 0.001), "U"),       # J outside its own CI
    ]
    for args, want in table:
        got = classify(*args)
        total += 1
        ok += got == want
        if got != want:
            print(f"  TABLE MISS {args}: {got} != {want}")
    print(f"  branch table: {ok}/{total}")

    # (2) synthetics through decide()
    rng = random.Random(1)
    n = 200

    def synth(j_mean, j_sd, gate=(True, True)):
        pp = {a: {s: [] for s in NEEDED} for a in ("base", "SFT")}
        for _ in range(n):
            base_sft, base_base = 0.5 + rng.gauss(0, 0.05), 0.03
            jj = rng.gauss(j_mean, j_sd)
            pp["base"]["D_SFT"].append(base_sft)
            pp["base"]["D_base"].append(base_base)
            pp["SFT"]["D_SFT"].append(base_sft + jj)
            pp["SFT"]["D_base"].append(base_base)
            pp["base"]["R_base"].append(0.01 if gate[1] else 0.5)
            pp["base"]["R_SFT"].append(0.01)
            pp["SFT"]["R_SFT"].append(0.01 if gate[0] else 5.0)
            pp["SFT"]["R_base"].append(0.01)
        return pp

    for want, pp in [("F1", synth(+0.05, 0.1)), ("F5", synth(-0.05, 0.1)),
                     ("F2", synth(0.0, 0.1)), ("F3", synth(0.05, 0.1, (True, False))),
                     ("F4", synth(0.05, 0.1, (False, False)))]:
        got = decide(pp, b=2000)["branch"]
        total += 1
        ok += got == want
        print(f"  synthetic {want}: reader says {got}")

    # (3) the pilot: committed digits reproduce, and it is no longer called F2
    pilot = ROOT / "notes/functional_transfer_k5000.json"
    d = load(pilot)
    res = decide({a: d["results"][a]["per_prompt"] for a in ("base", "SFT")})
    rep = f"{res['J']:+.5f} [{res['ci'][0]:+.5f}, {res['ci'][1]:+.5f}] p={res['p_perm']:.4f}"
    want = "-0.02701 [-0.05544, -0.00120] p=0.0501"
    print(f"  pilot digits: {rep}  (committed reader printed {want})")
    print(f"  pilot branch under the exhaustive table: {res['branch']} (want U)")
    total += 2
    ok += (rep == want) + (res["branch"] == "U")
    print(f"selftest {ok}/{total}")
    return 0 if ok == total else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = args[0] if args else "notes/functional_transfer_k5000.json"
    d = load(path)
    res = decide({a: d["results"][a]["per_prompt"] for a in ("base", "SFT")})
    report(d, res)
    if "--json" in argv:
        print("JSON " + json.dumps({k: v for k, v in res.items() if k != "means"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
