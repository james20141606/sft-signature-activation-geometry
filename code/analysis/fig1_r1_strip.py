#!/usr/bin/env python3
"""Figure 1 (R1 redraw, 2026-10-10): each arm's base->post displacement against its own
seed-pair band, one row per Table 1 arm (FIG1_GUIDE.md section 6, entry 01; type 4A strip plot).

Every mark is read from code/scripts/samearch.same_arch(): edge lists of graphs built on ONE
GPU architecture (H200) under results/sftband_20260926/run/graphs and
results/graph_build_audit_20261005/run/graphs, the same source as Table 1. The archived
cross-architecture readings (Table 4) are deliberately not plotted.

Grey band = range of the arm's seed-pair distances; open grey circles = the seed pairs;
filled accent marks = base->post contrasts (diamond = the audited SFT checkpoint, which is
scored against the seeded band but is not one of its k runs). Right column = Table 1 "outside".

    ~/.venvs/figtools/bin/python code/analysis/fig1_r1_strip.py
      -> figures/fig1_r1_20261010/fig1.{svg,pdf,png} (compose.export: audit 0 errors, no Type 3)
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
sys.path.insert(0, os.path.join(ROOT, "code", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "code", "figstyle"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory as blend

from fig1_rc import apply, C
import samearch
import compose

OUT = os.path.join(ROOT, "figures", "fig1_r1_20261010")
os.makedirs(OUT, exist_ok=True)
MINUS = "−"

F = samearch.same_arch()

# (key, row label, group); top to bottom
ROWS = [("sft", "Qwen3 SFT, 29 steps", "Qwen3-1.7B"),
        ("dpo12", "Qwen3 DPO, 12 steps", None),
        ("dpo29", "Qwen3 DPO, 29 steps", None),
        ("dpo60", "Qwen3 DPO, 60 steps", None),
        ("grpo", "Qwen3 GRPO, 100–300 steps", None),
        ("llama", "Llama-3.1-8B SFT", "Llama-3.1-8B")]

# ---- gate: the counts drawn must equal Table 1's "outside" column (parsed, not typed)
TEX = open(os.path.join(ROOT, "main.tex")).read()
tab = TEX[TEX.index("\\label{tab:main}"):TEX.index("\\end{tabular}", TEX.index("\\label{tab:main}"))]
table_out = [(int(a), int(b)) for a, b in re.findall(r"& \$(\d)/(\d)(?:\^\{a\})?\$ &", tab)]
aud = F["sft_audited"]
counts = {k: (F[k]["k_out"], F[k]["k"]) for k, _, _ in ROWS}
counts["sft"] = (F["sft"]["k_out"] + aud["k_out"], F["sft"]["k"] + 1)   # Table 1 footnote a
want = [counts[k] for k in ("sft", "llama", "dpo12", "dpo29", "dpo60", "grpo")]
assert table_out == want, (table_out, want)
z_aud = aud["z"][0]
assert ("$z=%s%.1f$" % ("-", abs(z_aud))) in TEX, z_aud          # body quotes the same z

apply(8.5)   # axis label 8.5 pt, ticks and row labels 8 pt at print (width = \\textwidth)
W, H = 5.5, 2.3
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0.255, 0.175, 0.645, 0.735])
ys = {}
y = 0.0
for i, (k, lab, grp) in enumerate(ROWS):
    if grp and i > 0:
        y -= 0.55                                 # gap between backbone groups
    ys[k] = y
    y -= 1.0

def dodge(x, w=0.00075, step=0.095):
    """Vertical offsets so that no two marks within w (data units, ~one marker width) coincide."""
    order, off, placed = np.argsort(x), np.zeros(len(x)), []
    for i in order:
        for cand in (0, step, -step, 2 * step, -2 * step):
            if all(abs(x[i] - x[j]) >= w or abs(cand - off[j]) >= step * 0.99 for j in placed):
                off[i] = cand
                break
        placed.append(i)
    return off


for k, lab, grp in ROWS:
    y = ys[k]
    pairs = np.sort(np.array(F[k]["pairs"]))
    cons = np.array(F[k]["contrasts"])
    lo, hi = pairs.min(), pairs.max()
    ax.add_patch(Rectangle((lo, y - 0.32), hi - lo, 0.64, fc=C.BAND, ec="none", zorder=1))
    # two lanes per row: reruns (seed pairs) above the row centre, base->post contrasts below;
    # within a lane, points closer than one marker width are dodged vertically (no data change)
    ax.scatter(pairs, y + 0.13 + dodge(pairs), s=16, facecolor="white", edgecolor=C.REF, lw=0.75, zorder=3)
    ax.scatter(cons, y - 0.13 + dodge(cons), s=24, color=C.ACC, lw=0, zorder=4)
    n_out, n = counts[k]
    ax.text(1.015, y, f"{n_out}/{n}", transform=blend(ax.transAxes, ax.transData),
            ha="left", va="center", color=C.INK)

# audited SFT checkpoint (scored against the seeded band; Table 1 footnote a)
ya = ys["sft"]
ca = aud["contrasts"][0]
ax.scatter([ca], [ya + 0.32], s=26, marker="D", color=C.ACC, lw=0, zorder=5)
# "audited run, z = -17.8" with an italic z (as in the body), as plain text runs (no mathtext:
# STIX math glyphs come out of the Chrome export as Type 3)
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
tp = dict(fontsize=8, color=C.INK)
lab = HPacker(children=[TextArea("audited run,", textprops=tp),
                        TextArea("z", textprops=dict(tp, style="italic")),
                        TextArea("=", textprops=tp),
                        TextArea(f"{MINUS}{abs(z_aud):.1f}", textprops=tp)], pad=0, sep=2.0)
ax.add_artist(AnnotationBbox(lab, (ca - 0.0010, ya + 0.32), box_alignment=(1.0, 0.5), frameon=False,
                             pad=0, zorder=5))

# direct labels for the two encodings, once, on the top row
s = F["sft"]
ax.text((min(s["pairs"]) + max(s["pairs"])) / 2, ya + 0.70, "same-recipe reruns", ha="center",
        va="bottom", fontsize=8, color="#595959")
ax.text(max(s["contrasts"]), ya + 0.70, "base → post-trained", ha="right",
        va="bottom", fontsize=8, color=C.ACC)
ax.text(1.015, ya + 0.70, "outside", transform=blend(ax.transAxes, ax.transData),
        ha="left", va="bottom", fontsize=8, color="#595959")

ax.set_yticks([ys[k] for k, _, _ in ROWS])
ax.set_yticklabels([lab for _, lab, _ in ROWS])
ax.tick_params(axis="y", length=0, pad=3)
ax.spines["left"].set_visible(False)
# thin separator between the two backbones (no floating group words; backbone is in the label)
ysep = (ys["grpo"] + ys["llama"]) / 2
ax.plot([-0.37, 1.0], [ysep, ysep], transform=blend(ax.transAxes, ax.transData), color="#bfbfbf",
        lw=0.5, clip_on=False, zorder=0)
ax.set_ylim(ys["llama"] - 0.55, ya + 1.10)
ax.set_xlim(0.020, 0.064)
ax.set_xticks([0.02, 0.03, 0.04, 0.05, 0.06])
ax.set_xticklabels(["0.02", "0.03", "0.04", "0.05", "0.06"])
ax.set_xlabel(f"displacement from base (1 {MINUS} edge Jaccard)")

plt.rcParams["savefig.bbox"] = None
svg = os.path.join(OUT, "fig1.svg")
fig.savefig(svg, bbox_inches=None)
pdf, png = compose.export(svg, width_in=5.5)
print("wrote", pdf, png)
print(compose.check_fonts(pdf))
for k, lab, grp in ROWS:
    f = F[k]
    print(f"{k:6} band {min(f['pairs']):.4f}-{max(f['pairs']):.4f}  contrasts "
          f"{min(f['contrasts']):.4f}-{max(f['contrasts']):.4f}  outside {counts[k][0]}/{counts[k][1]}")
print(f"audited SFT D={ca:.4f} z={z_aud:.2f}")
