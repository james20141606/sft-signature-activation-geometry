#!/usr/bin/env python3
"""Behavioural sanity check (App. C, 0.42\\textwidth): held-out BFCL validation accuracy, 100 items
(40/20/20/20 over four categories; accuracy = unweighted category mean), behaviour ONLY -- there is
deliberately no internal axis (the dissociation claim was withdrawn, L5).
Receipt: notes/bfcl_behavior_ci.json (from code/scripts/bfcl_behavior_ci.py over
notes/behavioral_calibration.json). Intervals: stratified normal 95% per arm; unpaired 95% for the
difference vs base. 2026-09-26: corrected from the n=80 (4 x 20) Wilson/Wald version.
The script asserts the two gains and their CIs the caption prints (+0.36 [+0.21,+0.50]; +0.11 [-0.03,+0.24])."""
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt

d = S.receipt("bfcl_behavior_ci.json"); n = d["n_items"]
arms = [("base", d["base"], fs.INK2), ("SFT (LoRA)", d["sft"], S.SFT), ("GRPO (RL)", d["grpo_merge_eval"], S.RL)]
rows = [(name, a["acc"], c) for name, a, c in arms]
ci = {name: tuple(a["ci95"]) for name, a, _ in arms}
gains = {"SFT (LoRA)": (d["delta_sft_vs_base"]["delta"],) + tuple(d["delta_sft_vs_base"]["ci95"]),
         "GRPO (RL)": (d["delta_grpo_merge_vs_base"]["delta"],) + tuple(d["delta_grpo_merge_vs_base"]["ci95"])}
assert n == 100
assert [round(x, 2) for x in gains["GRPO (RL)"]] == [0.36, 0.21, 0.50], gains
assert [round(x, 2) for x in gains["SFT (LoRA)"]] == [0.11, -0.03, 0.24], gains

fs.use()
fig, ax = plt.subplots(figsize=(0.42 * S.TEXTW, 1.52)); S.tidy(ax, grid="x")
ys = [3.0, 1.5, 0]
for y, (name, p, c) in zip(ys, rows):
    lo, hi = ci[name]
    ax.hlines(y, lo, hi, color=c, lw=1.2, zorder=2)
    ax.plot([p], [y], "o", ms=6, color=c, mec=fs.SURF, mew=0.8, zorder=3)
    ax.annotate(f"{p:.3f}", (hi, y), (4, 0), textcoords="offset points", ha="left", va="center", fontsize=fs.SIZE_TICK, color=c, fontweight="bold")
g, lo, hi = gains["SFT (LoRA)"]
ax.annotate(f"gain {g:+.2f} [{lo:+.2f}, {hi:+.2f}]\nnot resolved from 0", (0.40, ys[1]), (0, -6), textcoords="offset points", ha="left", va="top", fontsize=7, color=fs.INK2, style="italic", linespacing=1.1)
g, lo, hi = gains["GRPO (RL)"]
ax.annotate(f"gain {g:+.2f} [{lo:+.2f}, {hi:+.2f}]", (1.07, ys[2]), (0, -6), textcoords="offset points", ha="right", va="top", fontsize=7, color=fs.INK2, style="italic")
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=fs.SIZE_TICK)
for lab, (_, _, c) in zip(ax.get_yticklabels(), rows): lab.set_color(c)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.95, 3.6); ax.set_xlim(0.38, 1.08); ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel(f"BFCL accuracy (n = {n}, 95% CI)", fontsize=fs.SIZE_TICK)
ax.set_title("Sanity check, behaviour only:\nGRPO learned its task", pad=4, linespacing=1.05)
fig.subplots_adjust(left=0.25, right=0.97, top=0.79, bottom=0.24)
S.finish(fig, "fig_dissociation")
