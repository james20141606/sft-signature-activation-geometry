#!/usr/bin/env python3
"""Pin the main-text Table 1 (tab:main), Table 2 (tab:tulu) and the abstract's
receipt-derived numbers to the committed receipts.

check_rewiring.py compares receipts against constants and never reads main.tex, so a
typo in a Table 1 cell passed every gate (final review 2026-09-24, mutation M2: SFT
contrast 0.0588 -> 0.0598 left all checkers green). This script recomputes each cell
from notes/*.json, formats it the way the table prints it, and requires that exact
string inside the named table row of main.tex.

    python3 code/scripts/check_table_main.py            # exit 1 on any mismatch
    python3 code/scripts/check_table_main.py --selftest # mutate the tex, must FAIL
"""
import json
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N = os.path.join(ROOT, "notes")


def load(name):
    return json.load(open(os.path.join(N, name)))


def band(receipt, prefix):
    d = load(receipt)
    seeds = [x["edge_jaccard"] for x in d["seed_null"]]
    cons = [x["edge_jaccard"] for x in d["post_contrasts"] if x["pair"].startswith(prefix)]
    return seeds, cons


def f4(x):
    return "$%.4f$" % x


def cells(line):
    """Split one LaTeX table row into stripped cells (drops the trailing \\\\)."""
    line = line.split("\\\\")[0]
    return [c.strip() for c in line.split("&")]


def fz(z, fmt):
    return ("+" if z >= 0 else "-") + (fmt % abs(z))


def arm_cells(seeds, cons, k, outside_fmt="$%d/%d$", zfmt="%.2f", z=True):
    """Expected cells 2..7 of a Table 1 row, in column order."""
    mu, sd, lo = st.mean(seeds), st.stdev(seeds), min(seeds)
    out = sum(c < lo for c in cons)
    zs = sorted((c - mu) / sd for c in cons)
    return ["%d" % k,
            "%s--%s" % (f4(1 - max(seeds)), f4(1 - min(seeds))),
            "$%.4f$" % sd,
            "%s--%s" % (f4(1 - max(cons)), f4(1 - min(cons))),
            outside_fmt % (out, len(cons)),
            ("$%s$ to $%s$" % (fz(zs[0], zfmt), fz(zs[-1], zfmt))) if z else "---$^{b}$"]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import samearch  # noqa: E402  (same-architecture + archived recompute from graph JSONs)


def row_expectations():
    """(arm-cell text, expected cells 2..7) for every Table 1 row. Cell-exact, so a
    swap of two columns inside one row fails even when the value set is unchanged.
    Revision R1: Table 1 reads graphs built on ONE GPU architecture (samearch.SAME),
    recomputed here from the edge lists at full precision (z from unrounded values)."""
    F = samearch.same_arch()
    aud_out = F["sft_audited"]["k_out"]
    rows = [("LoRA-SFT, 29 steps", samearch.table_cells(F["sft"], outside_fmt="$%d/%d^{a}$",
                                                        zfmt="%.1f", extra_out=aud_out, extra_k=1)),
            ("LoRA-SFT", samearch.table_cells(F["llama"], z=False))]
    for arm, key in [("LoRA-DPO, 12 steps", "dpo12"), ("LoRA-DPO, 29 steps", "dpo29"),
                     ("LoRA-DPO, 60 steps", "dpo60")]:
        rows.append((arm, samearch.table_cells(F[key])))
    rows.append(("GRPO family$^{c}$", samearch.table_cells(F["grpo"], zfmt="%.1f")))
    return rows


def archived_expectations():
    """Rows of the appendix table tab:archived (the submitted readings, recomputed at full
    precision from notes/), cells 3..8 after the 'built on' column."""
    A = samearch.archived()
    aud_out = A["sft_audited"]["k_out"]
    return [("Qwen LoRA-SFT, 29 steps", samearch.table_cells(A["sft"], zfmt="%.1f",
                                                             extra_out=aud_out, extra_k=1)),
            ("Llama LoRA-SFT", samearch.table_cells(A["llama"], z=False)[:5] + ["---"]),
            ("Qwen LoRA-DPO, 12 steps", samearch.table_cells(A["dpo12"])),
            ("Qwen LoRA-DPO, 29 steps", samearch.table_cells(A["dpo29"])),
            ("Qwen LoRA-DPO, 60 steps", samearch.table_cells(A["dpo60"])),
            ("Qwen DPO, matched construction", samearch.table_cells(A["dpomc29"])),
            ("Qwen GRPO family", samearch.table_cells(A["grpo"]))]


def matched_dose_line(fams=None, fmt="same"):
    g = samearch.gap_line(fams or samearch.same_arch())
    sft, dpo = g["sft"], g["dpo"]
    ra, rb = g["ratio_sft"], g["ratio_dpo"]
    return [f4(sft[0]) + "--" + f4(sft[1]), f4(dpo[0]) + "--" + f4(dpo[1]),
            "gap $%.4f=%.1f\\times$ and $%.1f\\times$" % (g["gap"], ra, rb),
            "($%.4f$, $%.4f$)" % (g["spread_sft"], g["spread_dpo"])], (ra, rb)


def check(tex):
    bad = []
    m = re.search(r"\\label\{tab:main\}(.*?)\\end\{tabular\}", tex, re.S)
    if not m:
        return ["tab:main block not found"]
    lines = [l for l in m.group(1).splitlines() if "&" in l]
    for arm, exp in row_expectations():
        hits = [l for l in lines if len(cells(l)) >= 8 and cells(l)[1] == arm]
        if len(hits) != 1:
            bad.append("row %r: %d matching lines" % (arm, len(hits)))
            continue
        got = cells(hits[0])[2:8]
        for j, (g, e) in enumerate(zip(got, exp)):
            if g != e:
                bad.append("row %r col %d: tex %r != receipt %r" % (arm, j + 2, g, e))
    md, ratio = matched_dose_line()
    mdl = [l for l in m.group(1).splitlines() if "Matched dose" in l]
    for e in md:
        if not mdl or e not in mdl[0]:
            bad.append("matched-dose line lacks %r" % e)
    # Llama replication margin quoted in the body (same-architecture rebuild, branch LL1)
    margin = samearch.same_arch()["llama"]["r_min_over_max"]
    if "the nearest at $%.1f\\times$ the band maximum" % margin not in tex:
        bad.append("body Llama margin != %.1fx" % margin)

    # Appendix table tab:archived (archived build, full precision), cell-exact
    ta = re.search(r"\\label\{tab:archived\}(.*?)\\end\{tabular\}", tex, re.S)
    if not ta:
        bad.append("tab:archived block not found")
    else:
        alines = [cells(l) for l in ta.group(1).splitlines() if "&" in l]
        for arm, exp in archived_expectations():
            hit = [c for c in alines if c and c[0] == arm]
            if len(hit) != 1 or len(hit[0]) < 8:
                bad.append("tab:archived row %r not found" % arm); continue
            for j, (g, e) in enumerate(zip(hit[0][2:8], exp)):
                if g != e:
                    bad.append("tab:archived %r col %d: tex %r != recompute %r" % (arm, j + 2, g, e))
        amd, _ = matched_dose_line(samearch.archived())
        amdl = [l for l in ta.group(1).splitlines() if "Matched dose" in l]
        for e in amd:
            if not amdl or e not in amdl[0]:
                bad.append("tab:archived matched-dose line lacks %r" % e)

    # Table 2, both halves, cell-exact (v1 receipts)
    t = re.search(r"\\label\{tab:tulu\}(.*?)\\end\{tabular\}", tex, re.S)
    if not t:
        bad.append("tab:tulu block not found")
    else:
        rows = load("tulu_basediff_summary.json")["rows"]
        inc = {x["pair"]: x for x in load("tulu_stage_increments.json")["stage_increments"]}
        want = [("SFT", "SFT", "base$\\to$SFT", "base->SFT"),
                ("DPO", "DPO", "SFT$\\to$DPO", "SFT->DPO"),
                ("RLVR-PPO", "RLVR_PPO", "DPO$\\to$PPO", "DPO->PPO"),
                ("RLVR-GRPO", "RLVR_GRPO", "DPO$\\to$GRPO", "DPO->GRPO")]
        tl = [cells(l) for l in t.group(1).splitlines() if "&" in l]
        for lab, key, ilab, ikey in want:
            hit = [c for c in tl if c and c[0] == lab]
            if len(hit) != 1 or len(hit[0]) < 6:
                bad.append("tab:tulu row %s not found" % lab); continue
            r, x = rows[key], inc[ikey]
            exp = ["$+%.3f$" % r["dGini_vs_base"], "$%.2f$" % r["hub_jaccard_vs_base"], ilab,
                   "$%.3f$" % x["edge_jaccard"], "$%.2f$" % x["hub_jaccard"]]
            for j, (g, e) in enumerate(zip(hit[0][1:6], exp)):
                if g != e:
                    bad.append("tab:tulu %s col %d: tex %r != receipt %r" % (lab, j + 1, g, e))
        base_gini = rows["base"]["gini"]
        if "base Gini $%.3f$" % base_gini not in tex[t.start() - 900:t.start()]:
            bad.append("tab:tulu caption base Gini != %.3f" % base_gini)
        rp = load("tulu_stage_increments.json")["run_pair_null"]["edge_jaccard"]
        if "run pair ($%.3f$)" % rp not in tex[t.start() - 900:t.start()]:
            bad.append("tab:tulu caption run-pair reference != %.3f" % rp)

    # abstract numbers that come straight from receipts
    a = tex[tex.index(r"\begin{abstract}"):tex.index(r"\end{abstract}")]
    # Revision R1: the abstract's SFT z and gap ratios are the same-architecture readings.
    aud_z = samearch.same_arch()["sft_audited"]["z"][0]
    # 2026-09-24: the functional-transfer ratios (formerly the k=5000 pilot's 65x / 13.5x)
    # are pinned by check_functional_transfer.py against the registered sweep receipts.
    for label, s in [("SFT z", "$%.1f$ pairwise-distance" % abs(aud_z)),
                     ("gap/spread", "a gap $%.1f\\times$ and $%.1f\\times$ the spreads of the two arms" % ratio)]:
        if s not in " ".join(a.split()):
            bad.append("abstract %s: expected %r" % (label, s))
    return bad


def main():
    tex = open(os.path.join(ROOT, "main.tex")).read()
    if "--selftest" in sys.argv:
        plants = [("$0.0571$--$0.0615$ & $5/5^{a}$", "$0.0581$--$0.0615$ & $5/5^{a}$"),
                  ("$1/4$ & $-2.76$", "$2/4$ & $-2.76$"),
                  # reverting a Table 1 row to its archived (cross-architecture) count
                  ("& $3/4$ & $-1.86$ to $-0.06$", "& $0/4$ & $-1.86$ to $-0.06$"),
                  ("& $0/5$ & $-1.0$ to $+3.2$", "& $1/5$ & $-1.0$ to $+3.2$"),
                  ("& $+0.004$ & $0.60$", "& $+0.004$ & $0.66$"),
                  ("by $17.8$ pairwise-distance", "by $19.9$ pairwise-distance"),
                  ("a gap $5.0\\times$ and $5.6\\times$ the two", "a gap $5.3\\times$ and $5.6\\times$ the two"),
                  # in-row column swap (value set preserved): DPO-29 seed-pair <-> contrast
                  ("& $0.0303$--$0.0334$ & $0.0012$ & $0.0317$--$0.0356$ &",
                   "& $0.0317$--$0.0356$ & $0.0012$ & $0.0303$--$0.0334$ &"),
                  # matched-dose footnote and the archived table
                  ("gap $0.0216=5.0\\times$", "gap $0.0223=5.0\\times$"),
                  ("& $2/4$ & $-3.58$ to $+0.03$", "& $2/4$ & $-3.61$ to $-0.01$"),
                  ("& $1/6$ & $-1.95$ to $+0.82$", "& $0/6$ & $-1.95$ to $+0.82$"),
                  # Table 2 right half
                  ("& $0.669$ & $0.73$", "& $0.699$ & $0.73$"),
                  ("& $0.958$ & $0.94$", "& $0.958$ & $0.84$"),
                  # Llama margin in the body
                  ("the nearest at $1.7\\times$", "the nearest at $1.9\\times$")]
        caught = 0
        for o, n in plants:
            if o not in tex:
                print("PLANT NOT APPLICABLE:", o); continue
            if check(tex.replace(o, n, 1)):
                caught += 1
            else:
                print("NOT CAUGHT:", o)
        print("selftest caught %d/%d" % (caught, len(plants)))
        return 0 if caught == len(plants) else 1
    bad = check(tex)
    for b in bad:
        print("MISMATCH", b)
    print("tab:main / tab:tulu / abstract receipt pins: %s" % ("FAIL" if bad else "PASS"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
