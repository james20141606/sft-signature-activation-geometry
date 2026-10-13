#!/usr/bin/env python3
"""Available Qwen3-8B depth scan with explicit source-layer coverage.

Panel (a) recomputes top-1%-of-active inbound-weight share from every committed
layer-pair receipt. It shows every graph, the per-destination mean and range,
and the number of contributing source layers. Panel (b) makes the unbalanced
source/destination coverage visible. kt*/np200 sweeps and terminal layer 35 are
excluded exactly as in the manuscript.
"""
import glob, os, re
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

def top1_active(path):
    g = S.receipt(os.path.basename(path))
    dst = np.array(g["edges_dst"]); w = np.abs(np.array(g["weights"], float))
    feats, inv = np.unique(dst, return_inverse=True)
    inb = np.zeros(len(feats)); np.add.at(inb, inv, w)
    ranked = np.sort(inb)[::-1]; k = max(1, len(feats) // 100)
    return 100 * ranked[:k].sum() / inb.sum()

pts = []
for f in sorted(glob.glob(os.path.join(S.NOTES, "attribution_graph_v1_qwen3-8b_l*to*.json"))):
    b = os.path.basename(f)
    if "kt" in b or "np200" in b: continue
    m = re.search(r"l(\d+)to(\d+)\.json", b)
    if not m: continue
    src, dst = map(int, m.groups())
    if dst >= 35: continue
    pts.append((src, dst, top1_active(f)))

dsts = sorted({p[1] for p in pts}); srcs = sorted({p[0] for p in pts})
groups = {d: [p for p in pts if p[1] == d] for d in dsts}
means = np.array([np.mean([p[2] for p in groups[d]]) for d in dsts])
lo = np.array([min(p[2] for p in groups[d]) for d in dsts])
hi = np.array([max(p[2] for p in groups[d]) for d in dsts])
ns = np.array([len(groups[d]) for d in dsts])
assert 20 <= means[0] <= 25 and 45 <= means[-1] <= 55
assert ns.min() == 1 and ns.max() == 8

coverage = np.zeros((len(srcs), len(dsts)), dtype=int)
for src, dst, _ in pts:
    coverage[srcs.index(src), dsts.index(dst)] = 1

fs.use()
fig = plt.figure(figsize=(S.TEXTW, 4.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.4], hspace=0.38,
                      left=0.14, right=0.985, top=0.88, bottom=0.14)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1], sharex=ax)

S.tidy(ax)
for d in dsts:
    vals = [p[2] for p in groups[d]]
    offsets = np.linspace(-0.22, 0.22, len(vals)) if len(vals) > 1 else [0]
    ax.scatter(np.array(offsets) + d, vals, s=18, color=fs.tint(S.QWEN, 0.5),
               edgecolor="none", zorder=2)
ax.vlines(dsts, lo, hi, color=fs.INK3, lw=0.8, zorder=1)
ax.plot(dsts, means, color=S.QWEN, lw=1.15, alpha=0.75, zorder=2)
ax.scatter(dsts, means, s=28, marker="D", color=S.QWEN, edgecolor=fs.SURF,
           linewidth=0.6, zorder=3)
for d, n in zip(dsts, ns):
    ax.text(d, 58.3, str(n), ha="center", va="center", fontsize=fs.SIZE_TICK,
            color=fs.INK2)
ax.text(2.35, 58.3, "n", ha="right", va="center", fontsize=fs.SIZE_TICK,
        color=fs.INK2, fontweight="bold")
ax.annotate(f"{means[0]:.0f}%", (dsts[0], means[0]), (-5, 7),
            textcoords="offset points", ha="right", fontsize=fs.SIZE_TICK,
            color=S.QWEN, fontweight="bold")
ax.annotate(f"{means[-1]:.0f}%", (dsts[-1], means[-1]), (5, 2),
            textcoords="offset points", ha="left", fontsize=fs.SIZE_TICK,
            color=S.QWEN, fontweight="bold")
ax.axvspan(34.2, 36, color=fs.GRID, zorder=0)
ax.text(35.0, 21.0, "l35", ha="center", va="center",
        fontsize=fs.SIZE_TICK, color=fs.INK3, linespacing=1.05)
ax.set_xlim(2.5, 36); ax.set_ylim(15, 61); ax.set_yticks([20, 30, 40, 50])  # no 60 tick: the n-row header sits there
ax.set_ylabel("top-1%-of-active\ninbound share (%)")
ax.tick_params(labelbottom=False)
fs.panel_label(ax, "(a)", x=-0.075, y=1.02)
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=fs.tint(S.QWEN, 0.5), ms=4,
           label="one source→destination graph"),
    Line2D([], [], marker="D", ls="-", color=S.QWEN, ms=4,
           label="destination mean; rule = min–max")],
    loc="upper left", bbox_to_anchor=(0, 1.38), fontsize=fs.SIZE_TICK, handlelength=1.4,
    labelspacing=0.25, borderaxespad=0.15)

bx.grid(False)
for si,src in enumerate(srcs):
    for di,dst in enumerate(dsts):
        bx.add_patch(plt.Rectangle((dst-.7,si-.40),1.4,.8,facecolor=fs.tint_hex(S.QWEN,.82) if coverage[si,di] else '#EEF1F4',edgecolor='white',lw=.4))
bx.set_ylim(len(srcs)-.5,-.5);bx.set_xlim(2.5,36)
bx.set_yticks(range(len(srcs)));bx.set_yticklabels(srcs,fontsize=fs.SIZE_TICK)
bx.set_xticks(dsts);bx.set_xticklabels(dsts,fontsize=fs.SIZE_TICK)
bx.tick_params(length=0)
for s in bx.spines.values(): s.set_visible(False)
bx.set_xlabel("destination layer")
bx.set_ylabel("source layer")
fs.panel_label(bx, "(b)", x=-0.075, y=1.0)
bx.text(1.0, 1.04, "available layer-pair receipts", transform=bx.transAxes,
        ha="right", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK2,
        fontweight="bold")

S.finish(fig, "fig7_concentration_vs_depth")
print(f"{len(pts)} graphs; n/destination {ns.min()}..{ns.max()}; endpoint means {means[0]:.1f} -> {means[-1]:.1f}")
