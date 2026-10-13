#!/usr/bin/env python3
"""Body Figure 2 (overhaul 2026-09-24): own-run calibration of the displacement audit.

(a) SFT vs its own four-run band on two backbones, with DPO at SFT's own 29 steps on the
    same axis, so the null-free SFT/DPO disjointness is visible next to the band readings.
(b) DPO dose ladder with dose on the x axis: the contrast mean is flat while the seed-pair
    band moves (its spread narrows from 12 to 29 steps, then holds) -- the band reading turns on the denominator.

Revision R1 (2026-10-07): every arm is read from graphs built on ONE GPU architecture (H200),
via code/scripts/samearch.py (graph JSONs under results/sftband_20260926 and
results/graph_build_audit_20261005). The submitted version read notes/circuit_rewiring_*.json,
whose base and run graphs crossed architectures (archived readings: Table archived, App. build).
The expected DPO counts are parsed from main.tex's own dose sentence, not hard-coded.

    ~/.venvs/figtools/bin/python code/analysis/fig_main_seednull_v2.py
"""
import os, sys, statistics as st
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

OUT = os.path.join(S.FIGS, "overhaul_20260924"); os.makedirs(OUT, exist_ok=True)
MINUS = "−"


sys.path.insert(0, os.path.join(S.ROOT, "code", "scripts"))
import re
import samearch
F = samearch.same_arch()


def arm(key):
    f = F[key]
    assert len(f["pairs"]) == 6 and len(f["contrasts"]) == 4, (key, len(f["pairs"]), len(f["contrasts"]))
    return np.array(f["pairs"]), np.array(f["contrasts"])


sft = arm("sft")
lla = arm("llama")
dpo = {12: arm("dpo12"), 29: arm("dpo29"), 60: arm("dpo60")}


def outside(a):  # contrast more displaced than every seed pair
    null, pts = a
    return int(sum(p > null.max() for p in pts))


# ---- assertions against the manuscript (expected values parsed from main.tex, not literals)
TEX = open(os.path.join(S.ROOT, "main.tex")).read()
m = re.search(r"Under one architecture \$K_\{\\mathrm\{out\}\}\(\\tau\)=(\d+),(\d+),(\d+)\$ of \$4\$", TEX)
assert m, "main.tex dose sentence not found"
assert [outside(dpo[t]) for t in (12, 29, 60)] == [int(m.group(i)) for i in (1, 2, 3)]
assert outside(sft) == 4 and outside(lla) == 4
gap = sft[1].min() - dpo[29][1].max()
spread_s, spread_d = np.ptp(sft[1]), np.ptp(dpo[29][1])
ratio = gap / max(spread_s, spread_d)
assert ("$%.1f\\times$ the wider arm's range ($%.4f$ vs.\\ $%.4f$)" % (ratio, gap, max(spread_s, spread_d))) in TEX, ratio
means = [dpo[t][1].mean() for t in (12, 29, 60)]
assert ("4-run means $%.4f$/$%.4f$/$%.4f$" % tuple(means)) in TEX, means
assert ("the nearest at $%.1f\\times$ the band maximum" % (lla[1].min() / lla[0].max())) in TEX

fs.use()
fig = plt.figure(figsize=(fs.TEXTWIDTH_IN, 2.45))
gs = fig.add_gridspec(1, 2, width_ratios=[1.28, 1.0], left=0.155, right=0.985,
                      bottom=0.25, top=0.84, wspace=0.30)
a = fig.add_subplot(gs[0]); b = fig.add_subplot(gs[1])

# ------------------------------------------------------------------ (a)
rows = [(2.0, "Qwen3 SFT", sft, S.SFT), (1.0, "Qwen3 DPO", dpo[29], S.DPO),
        (0.0, "Llama-3.1 SFT", lla, S.SFT)]
for y, _, (null, pts), c in rows:
    lo, hi = null.min(), null.max()
    a.add_patch(Rectangle((lo, y - 0.30), hi - lo, 0.60, fc=fs.tint(S.NULL, 0.20), ec="none", zorder=1))
    a.scatter(null, y + np.linspace(-0.13, 0.13, len(null)), s=13, facecolor=fs.SURF,
              edgecolor=S.NULL, lw=0.7, zorder=3)
    a.scatter(pts, y + np.linspace(-0.16, 0.16, len(pts)), s=24, color=c, edgecolor=fs.SURF,
              lw=0.5, zorder=4)
    a.text(0.0705, y, f"{outside((null, pts))}/4", va="center", ha="right",
           fontsize=fs.SIZE_TICK, color=c)
a.text(0.0705, 2.62, "outside", va="center", ha="right", fontsize=fs.SIZE_TICK, color=fs.INK3)
# null-free comparison at 29 steps: gap between the two contrast ranges
yb = 1.5
x0, x1 = dpo[29][1].max(), sft[1].min()
a.annotate("", xy=(x1, yb), xytext=(x0, yb),
           arrowprops=dict(arrowstyle="<->", color=fs.INK2, lw=0.8, shrinkA=0, shrinkB=0))
a.text((x0 + x1) / 2, yb + 0.07, f"gap {ratio:.1f}× spread", ha="center", va="bottom",
       fontsize=fs.SIZE_TICK, color=fs.INK2)
a.set_yticks([2, 1, 0]); a.set_yticklabels([r[1] + ("\n29 steps" if i < 2 else "\n8B") for i, r in enumerate(rows)],
                                          linespacing=1.0)
for lab, r in zip(a.get_yticklabels(), rows): lab.set_color(r[3])
a.set_xlim(0.019, 0.071); a.set_ylim(-0.55, 2.75)
a.set_xticks([0.02, 0.03, 0.04, 0.05, 0.06, 0.07]); a.set_xticklabels(["0.02", "", "0.04", "", "0.06", ""])
a.set_xlabel(f"displacement from base, 1 {MINUS} edge Jaccard")
a.grid(True, axis="x"); a.grid(False, axis="y")
a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
fs.panel_label(a, "a", x=-0.30, y=1.02)
a.set_title("SFT clears its own band on two backbones", fontsize=fs.SIZE_LABEL, loc="left", pad=4)

# ------------------------------------------------------------------ (b)
xs = {12: 0.0, 29: 1.0, 60: 2.0}
bmeans = []
for t, (null, pts) in dpo.items():
    x = xs[t]; lo, hi = null.min(), null.max()
    b.add_patch(Rectangle((x - 0.26, lo), 0.52, hi - lo, fc=fs.tint(S.NULL, 0.20), ec="none", zorder=1))
    b.scatter(x - 0.12 + np.linspace(-0.05, 0.05, len(null)), null, s=13, facecolor=fs.SURF,
              edgecolor=S.NULL, lw=0.7, zorder=3)
    b.scatter(x + 0.12 + np.linspace(-0.04, 0.04, len(pts)), pts, s=24, color=S.DPO,
              edgecolor=fs.SURF, lw=0.5, zorder=4)
    bmeans.append(null.mean())
    b.text(x + 0.12, 0.0374, f"{outside((null, pts))}/4", ha="center", va="bottom",
           fontsize=fs.SIZE_TICK, color=S.DPO)
X = [0.0, 1.0, 2.0]
b.plot([x + 0.12 for x in X], means, color=S.DPO, lw=1.0, zorder=2)
b.plot([x - 0.12 for x in X], bmeans, color=S.NULL, lw=1.0, ls=(0, (3, 2)), zorder=2)
b.text(2.42, means[-1], "contrast\nmean", va="center", ha="left", fontsize=fs.SIZE_TICK,
       color=S.DPO, linespacing=1.0)
b.text(2.42, bmeans[-1] - 0.0004, "seed-pair\nmean", va="center", ha="left", fontsize=fs.SIZE_TICK,
       color=fs.INK2, linespacing=1.0)
b.set_xticks(X); b.set_xticklabels(["12", "29", "60"])
b.set_xlabel("DPO optimizer steps")
b.text(2.42, 0.0374, "outside", ha="left", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK3)
b.set_xlim(-0.45, 3.05); b.set_ylim(0.0285, 0.0386)
b.set_yticks([0.030, 0.032, 0.034, 0.036])
b.set_ylabel(f"1 {MINUS} edge Jaccard")
b.grid(True, axis="y")
fs.panel_label(b, "b", x=-0.22, y=1.02)
b.set_title("DPO: flat, at its band's edge", fontsize=fs.SIZE_LABEL, loc="left", pad=4)

fig.legend([Line2D([], [], marker="o", ls="", mfc=fs.SURF, mec=S.NULL, ms=4.5),
            Rectangle((0, 0), 1, 1, fc=fs.tint(S.NULL, 0.20), ec="none"),
            Line2D([], [], marker="o", ls="", color=S.SFT, mec=fs.SURF, ms=5),
            Line2D([], [], marker="o", ls="", color=S.DPO, mec=fs.SURF, ms=5)],
           ["seed-pair distance (6 per arm)", "seed-pair range", "base→SFT run", "base→DPO run"],
           loc="lower center", ncol=4, bbox_to_anchor=(0.53, 0.0), handletextpad=0.3, columnspacing=1.1)

svg = os.path.join(OUT, "fig2_seednull.svg")
fig.savefig(svg, bbox_inches=None)
print("wrote", svg, "gap/spread", round(ratio, 2), "DPO means", [round(m, 4) for m in means],
      "band means", [round(m, 4) for m in bmeans])
