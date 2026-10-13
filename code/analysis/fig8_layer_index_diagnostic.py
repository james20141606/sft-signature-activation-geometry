#!/usr/bin/env python3
"""Layer-local diagnostic from the committed 15-index-slot receipt.

The qwen-scope SAEs are trained independently by layer, so an integer index at
one destination is not the same feature at another destination. This figure
collapses the receipt within each destination layer and never draws a cross-
layer feature row or recurrence count.
"""
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt

d = S.receipt("fig8_backbone_recurrence_data.json")
hubs, dsts = d["hubs"], d["dst_layers"]
M = np.array([[d["share_pct_max"][str(t)][str(h)] for t in dsts] for h in hubs])
present = np.array([[d["in_top30_any"][str(t)][str(h)] for t in dsts]
                    for h in hubs], dtype=bool)
slot_count = present.sum(axis=0)
max_share = np.array([M[:, j][present[:, j]].max() if present[:, j].any() else 0
                      for j in range(len(dsts))])
n_graphs = np.array([d["n_graphs_per_dst"][str(t)] for t in dsts])
assert len(dsts) == 31 and len(hubs) == 15
assert int((present.any(axis=1)).sum()) == 13
assert round(max_share[dsts.index(35)]) == 46

fs.use()
fig = plt.figure(figsize=(S.TEXTW, 3.0))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 0.58], hspace=0.18,
                      left=0.105, right=0.985, top=0.95, bottom=0.14)
ax, bx, cx = [fig.add_subplot(gs[i]) for i in range(3)]

for a in (ax, bx, cx):
    S.tidy(a)
    a.axvspan(34.5, 35.5, color=fs.tint(S.GEMMA, 0.13), zorder=0)
    a.set_xlim(0.3, 35.7)

ax.bar(dsts, slot_count, width=0.72, color=fs.tint(S.QWEN, 0.72), zorder=2)
ax.scatter(dsts, slot_count, s=18, color=S.QWEN, edgecolor=fs.SURF,
           linewidth=0.5, zorder=3)
ax.set_ylim(0, max(4, int(slot_count.max())+1))
ax.set_ylabel("audited slots\nin any top-30")
ax.tick_params(labelbottom=False)
fs.panel_label(ax, "(a)", x=-0.07, y=1.0)
ax.text(1.0, 1.02, "layer-local counts; index numbers are not aligned features",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.SIZE_TICK,
        color=fs.INK2, fontweight="bold")

colors = [S.GEMMA if t == 35 else S.SFT for t in dsts]
bx.vlines(dsts, 0, max_share, color=[fs.tint(c, 0.72) for c in colors],
          lw=1.7, zorder=2)
bx.scatter(dsts, max_share, s=30, color=colors, edgecolor=fs.SURF,
           linewidth=0.6, zorder=3)
bx.set_ylim(0, max_share.max()*1.14)
bx.set_ylabel("strongest layer-local\ninbound share (%)")
bx.tick_params(labelbottom=False)
fs.panel_label(bx, "(b)", x=-0.07, y=1.0)
bx.annotate(f"l35  {max_share[dsts.index(35)]:.0f}%\nunembedding-dominated",
            (35, max_share[dsts.index(35)]), (-8, -4), textcoords="offset points",
            ha="right", va="top", fontsize=fs.SIZE_TICK, color=S.GEMMA,
            fontweight="bold", linespacing=1.0)

cx.bar(dsts, n_graphs, width=0.72, color=fs.INK3, zorder=2)
cx.set_ylim(0, int(n_graphs.max())+1)
cx.set_ylabel("graphs")
cx.set_xlabel("destination layer")
cx.set_xticks(dsts); cx.set_xticklabels(dsts, fontsize=fs.SIZE_TICK)
fs.panel_label(cx, "(c)", x=-0.07, y=0.96)

S.finish(fig, "fig8_layer_index_diagnostic")
print("destinations", len(dsts), "coverage", int(n_graphs.min()), int(n_graphs.max()),
      "l35 max share", max_share[dsts.index(35)])
