#!/usr/bin/env python3
"""Emit, from the sweep receipts only, every string the paper quotes about the registered
functional-transfer sweep (notes/PREREG_functional_transfer_full_20260924.md, Sec. 6):
the appendix table rows and the main-text numbers. check_functional_transfer.py
recomputes the same strings and requires each to appear in main.tex.

Formats fixed in advance (not chosen after the result): ratios as an integer if >= 10,
else one decimal; medians one decimal; J and CI bounds 4 decimals; p 4 decimals;
Delta in nats 3 decimals; n with a thousands separator.
"""
from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from functional_transfer_read import decide, load  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
KS = (2000, 5000, 10000)
ROWS = [("A", k) for k in KS] + [("B", k) for k in KS] + [("C", 5000)]


def rfmt(x):
    return f"{round(x):d}" if x >= 10 else f"{x:.1f}"


def sgn(x, d=4):
    return f"{x:+.{d}f}"


def thousands(n):
    return f"{n:,}".replace(",", "{,}")


def cell(cfg, k):
    p = NOTES / f"functional_transfer_full_{cfg}_k{k}.json"
    d = load(p)
    res = decide({a: d["results"][a]["per_prompt"] for a in ("base", "SFT")})
    r = d["results"]
    own = r["SFT"]["per_prompt"]["D_SFT"]
    ctl = r["SFT"]["per_prompt"]["R_SFT"]
    ownb = r["base"]["per_prompt"]["D_base"]
    ctlb = r["base"]["per_prompt"]["R_base"]
    out = {
        "cfg": cfg, "k": k, "n": d["n_prompts"],
        "sizes": (d["set_sizes"]["D_SFT"], d["set_sizes"]["D_base"]),
        "delta_sft": sum(own) / len(own),
        "ratio_sft": (sum(own) / len(own)) / (sum(ctl) / len(ctl)),
        "med_sft": st.median(own) / st.median(ctl),
        "ratio_base": (sum(ownb) / len(ownb)) / (sum(ctlb) / len(ctlb)),
        "res": res,
    }
    return out


def strings():
    cells = {(c, k): cell(c, k) for c, k in ROWS}
    s = {}
    for (c, k), x in cells.items():
        res = x["res"]
        if "J" in res:
            jtxt = (f"${sgn(res['J'])}$ & $[{sgn(res['ci'][0])}, {sgn(res['ci'][1])}]$ & "
                    f"${res['p_perm']:.4f}$")
        else:
            jtxt = "--- & --- & ---"
        role = {"A": "A", "B": "B$^{\\dagger}$", "C": "C$^{\\dagger}$"}[c]
        s[f"row:{c}:{k}"] = (
            f"{role} & ${thousands(k)}$ & ${x['sizes'][0]}/{x['sizes'][1]}$ & "
            f"${x['delta_sft']:.3f}$ & ${rfmt(x['ratio_sft'])}\\times$ & "
            f"${rfmt(x['ratio_base'])}\\times$ & {jtxt} & {res['branch']} \\\\")
    a10 = cells[("A", 10000)]
    s["abstract_ratio"] = (f"${rfmt(a10['ratio_sft'])}\\times$ a matched random control in "
                           f"mean, ${a10['med_sft']:.1f}\\times$ in median")
    s["n"] = f"${thousands(a10['n'])}$ held-out prompts"
    s["body_ratio"] = (f"${rfmt(a10['ratio_sft'])}\\times$ in mean, ${a10['med_sft']:.1f}\\times$ "
                       f"in median at $10{{,}}000$ edges")
    js = [cells[("A", k)]["res"] for k in KS if "J" in cells[("A", k)]["res"]]
    if js:
        s["main_J"] = "$J=" + "$ and $".join(sgn(r["J"]) for r in js) + "$"
        # 2026-09-25 de-numbering: the body states the direction of J in words; the values
        # live in tab:functional (pinned row by row). The sentence is DERIVED from the
        # receipts: sign of every J and whether every CI excludes zero.
        ks_j = [k for k in KS if "J" in cells[("A", k)]["res"]]
        signs = {("negative" if r["J"] < 0 else "positive") for r in js}
        excl = all(r["ci"][1] < 0 or r["ci"][0] > 0 for r in js)
        s["body_J_dir"] = (f"the interaction $J$ is {'/'.join(sorted(signs))} at "
                           + " and ".join(f"${thousands(k)}$" for k in ks_j)
                           + " edges, with CIs " + ("excluding" if excl else "not all excluding")
                           + " zero")
    rs = sorted(round(cells[("A", k)]["ratio_sft"]) for k in KS)
    s["app_range"] = f"${rs[0]}$--${rs[-1]}\\times$ their random control in SFT"
    return s, cells


def main():
    s, cells = strings()
    for k, v in s.items():
        print(f"{k:16s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
