#!/usr/bin/env python3
"""Tulu-3 staircase: levels and stage displacements against one public pair.

All values come from committed receipts. The PPO<->GRPO quantity is drawn as
one vertical reference line, never as a band or distribution.
"""
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Q, Gm, T = (S.receipt(n)["rows"] for n in
            ("basediff_qwen_summary.json", "basediff_gemma_summary.json", "tulu_basediff_summary.json"))
fam = [
    ("Qwen3-1.7B\nlight GRPO", S.QWEN,
     [("base", Q["base"]), ("SFT", Q["SFT"]), ("GRPO", Q["GRPO"]),
      ("s4", Q["GRPO_s4"]), ("s5", Q["GRPO_s5"]), ("shaped", Q["SHAPED_l0.1"])]),
    ("Gemma-2-2B\nheavy instruct", S.GEMMA,
     [("base", Gm["base"]), ("-it", Gm["IT"])]),
    ("Llama-3.1-8B\nTulu-3 staircase", S.LLAMA,
     [("base", T["base"]), ("SFT", T["SFT"]), ("DPO", T["DPO"]),
      ("PPO", T["RLVR_PPO"]), ("GRPO", T["RLVR_GRPO"])])]
inc = S.receipt("tulu_stage_increments.json")
stages = [(r["pair"], 1 - r["edge_jaccard"]) for r in inc["stage_increments"]]
reference = 1 - inc["run_pair_null"]["edge_jaccard"]
assert [round(1-v, 3) for _, v in stages] == [0.669, 0.796, 0.958, 0.906]
assert round(1-reference, 3) == 0.903

fs.use()
fig = plt.figure(figsize=(S.TEXTW, 3.0))
outer = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], wspace=0.34,
                         left=0.13, right=0.985, top=0.80, bottom=0.29)
ga = outer[0, 0].subgridspec(1, 3, width_ratios=[6, 2.5, 5], wspace=0.12)
axes = [fig.add_subplot(ga[0, i]) for i in range(3)]
bx = fig.add_subplot(outer[0, 1])

for k, (ax, (name, color, rows)) in enumerate(zip(axes, fam)):
    S.tidy(ax)
    x = np.arange(len(rows)); yv = [r[1]["gini"] for r in rows]
    ax.plot(x, yv, color=color, lw=1.1, zorder=2)
    ax.scatter(x, yv, s=25, color=color, edgecolor=fs.SURF, linewidth=0.6, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=fs.SIZE_TICK,
                                         rotation=90)
    ax.set_xlim(-0.55, len(rows)-0.45); ax.set_ylim(0.60, 0.85)
    ax.text(0.5, 1.04, ["Qwen3\nlight recipe", "Gemma\nheavy recipe", "Llama3\nTulu3"][k], transform=ax.transAxes, ha="center", va="bottom",
            fontsize=fs.SIZE_TICK, color=color, fontweight="bold", linespacing=1.0)
    if k: ax.tick_params(labelleft=False)
axes[0].set_ylabel("attribution-graph Gini\n(single-checkpoint level)")
axes[1].set_xlabel("post-training stage", labelpad=24)  # clear the rotated tick labels of the neighbouring panel ("shaped")
axes[0].text(2.55, 0.653, "levels alone do not\nidentify an RL effect", ha="center",
             va="center", fontsize=fs.SIZE_TICK, color=fs.INK2, style="italic",
             linespacing=1.05)
fs.panel_label(axes[0], "(a)", x=0, y=1.03)

S.tidy(bx, grid="x")
labels = ["base→SFT", "SFT→DPO", "DPO→PPO", "DPO→GRPO"]
colors = [S.SFT, S.DPO, S.RL, S.RL]
y = np.arange(4)[::-1]
for yi, (_, value), color in zip(y, stages, colors):
    bx.hlines(yi, 0, value, color=fs.tint(color, 0.7), lw=2.2, zorder=2)
    bx.scatter(value, yi, s=42, color=color, edgecolor=fs.SURF, linewidth=0.7, zorder=3)
    # white backing so the dashed reference line passes behind a label that sits across it
    bx.text(value + 0.009, yi, f"{value:.3f}", ha="left", va="center", zorder=4,
            fontsize=fs.SIZE_TICK, color=color, fontweight="bold",
            bbox=dict(facecolor=fs.SURF, edgecolor="none", pad=0.6))
bx.axvline(reference, color=fs.INK2, ls=(0, (3, 2)), lw=1.0, zorder=1)
bx.text(reference + 0.006, 4.07, "one public pair\nPPO↔GRPO  0.097",
        ha="left", va="top", fontsize=fs.SIZE_TICK, color=fs.INK2,
        linespacing=1.05)
bx.set_yticks(y); bx.set_yticklabels(labels, fontsize=fs.SIZE_TICK)
bx.tick_params(axis="y", length=0)
bx.set_xlim(0, 0.38); bx.set_ylim(-0.55, 4.15)
bx.set_xlabel("stage displacement (1 − edge Jaccard)")
fs.panel_label(bx, "(b)", x=-0.18, y=1.03)
bx.legend(handles=[
    Line2D([], [], marker="o", ls="", color=S.SFT, ms=5, label="SFT stage"),
    Line2D([], [], marker="o", ls="", color=S.DPO, ms=5, label="DPO stage"),
    Line2D([], [], marker="o", ls="", color=S.RL, ms=5, label="RLVR increment")],
    loc="lower right", fontsize=fs.SIZE_TICK, labelspacing=0.22,
    handletextpad=0.4, borderaxespad=0.15)

S.finish(fig, "fig_staircase_money")
print("stage displacement", [round(v, 4) for _, v in stages], "single reference", round(reference, 4))
