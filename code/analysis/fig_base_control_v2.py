#!/usr/bin/env python3
"""Body Figure 3 (overhaul 2026-09-24): the matched-base control as three small multiples
sharing one row per post-training checkpoint.

(a) level: top-1%-of-active inbound-weight share, against the checkpoint's own base (vertical
    rule) and, for Qwen, the degree-preserving permutation null of the base graph -- far above
    chance, yet equal to the base;
(b) change: Delta Gini vs base, with the observed five-run Qwen envelope (|dGini| <= 0.005);
(c) identity: top-32 hub Jaccard vs base.

Receipts (notes/): basediff_qwen_summary.json, basediff_gemma_summary.json,
h1_permutation_null_1p7b_l14to22.json. No new data.

    ~/.venvs/figtools/bin/python code/analysis/fig_base_control_v2.py
"""
import os
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.join(S.FIGS, "overhaul_20260924"); os.makedirs(OUT, exist_ok=True)
MINUS = "−"

Q, G = (S.receipt(n)["rows"] for n in ("basediff_qwen_summary.json", "basediff_gemma_summary.json"))
P = S.receipt("h1_permutation_null_1p7b_l14to22.json")
rows = [("SFT (LoRA)", Q["SFT"], Q["base"], S.QWEN),
        ("GRPO, accuracy-only", Q["GRPO"], Q["base"], S.QWEN),
        ("GRPO seed s4", Q["GRPO_s4"], Q["base"], S.QWEN),
        ("GRPO seed s5", Q["GRPO_s5"], Q["base"], S.QWEN),
        ("GRPO, shaped reward", Q["SHAPED_l0.1"], Q["base"], S.QWEN),
        ("Gemma-2-2B-it", G["IT"], G["base"], S.GEMMA)]
env = max(abs(r["dGini_vs_base"]) for _, r, _, c in rows if c == S.QWEN)
assert abs(env - 0.005) < 1e-12 and abs(G["IT"]["dGini_vs_base"] - 0.035) < 1e-12
assert abs(Q["base"]["t1"] - 0.427) < 1e-9 and abs(Q["GRPO"]["t1"] - 0.429) < 1e-9
assert round(P["null_mean"], 3) == 0.330 and round(P["z_score"], 1) == 20.0
ys = [6, 5, 4, 3, 2, 0.6]

fs.use()
fig, axs = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH_IN, 2.35), sharey=True,
                        gridspec_kw=dict(width_ratios=[1.25, 1.0, 0.8]))
fig.subplots_adjust(left=0.215, right=0.985, bottom=0.19, top=0.80, wspace=0.13)
a, b, c = axs
for ax in axs:
    ax.set_ylim(-0.1, 6.7); ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.axhline(1.3, color=fs.GRID, lw=0.8, zorder=0)

# (a) level vs base and vs permutation null
lo, hi = P["null_ci95"]
a.add_patch(Rectangle((lo, 1.55), hi - lo, 4.9, fc=fs.tint(S.NULL, 0.25), ec="none", zorder=0))
a.text((lo + hi) / 2, 6.55, "permuted\n(chance)", ha="center", va="bottom", fontsize=fs.SIZE_TICK,
       color=fs.INK2, linespacing=1.0)
a.plot([Q["base"]["t1"]] * 2, [1.55, 6.45], color=fs.INK2, lw=0.9, zorder=1)
a.text(Q["base"]["t1"], 6.55, "base", ha="center", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK2)
a.plot([G["base"]["t1"]] * 2, [0.15, 1.05], color=fs.INK2, lw=0.9, zorder=1)
a.text(G["base"]["t1"] - 0.003, 0.6, "base", ha="right", va="center", fontsize=fs.SIZE_TICK, color=fs.INK2)
for y, (_, r, base, col) in zip(ys, rows):
    a.hlines(y, min(base["t1"], r["t1"]), max(base["t1"], r["t1"]), color=fs.tint(col, 0.7), lw=2, zorder=2)
    a.scatter(r["t1"], y, s=30, color=col, edgecolor=fs.SURF, lw=0.6, zorder=3)
a.set_xlim(0.30, 0.45); a.set_xticks([0.32, 0.36, 0.40, 0.44])
a.set_xlabel("top-1% inbound share")
a.set_title("a   level", fontsize=fs.SIZE_LABEL, loc="left", pad=16)

# (b) change vs base with the observed Qwen envelope
b.add_patch(Rectangle((-env, 1.55), 2 * env, 4.9, fc=fs.tint(S.NULL, 0.25), ec="none", zorder=0))
b.text(0, 6.55, f"±{env:.3f}\nQwen range", ha="center", va="bottom", fontsize=fs.SIZE_TICK,
       color=fs.INK2, linespacing=1.0)
b.plot([0, 0], [-0.1, 6.45], color=fs.INK3, lw=0.8, zorder=1)
for y, (_, r, _, col) in zip(ys, rows):
    v = r["dGini_vs_base"]
    b.hlines(y, min(0, v), max(0, v), color=fs.tint(col, 0.7), lw=2, zorder=2)
    b.scatter(v, y, s=30, color=col, edgecolor=fs.SURF, lw=0.6, zorder=3)
b.text(G["IT"]["dGini_vs_base"], 0.6 + 0.35, f"+{G['IT']['dGini_vs_base']:.3f}", ha="center", va="bottom",
       fontsize=fs.SIZE_TICK, color=S.GEMMA)
b.set_xlim(-0.012, 0.045); b.set_xticks([0, 0.02, 0.04]); b.set_xticklabels(["0", "0.02", "0.04"])
b.set_xlabel("ΔGini vs base")
b.set_title("b   change", fontsize=fs.SIZE_LABEL, loc="left", pad=16)

# (c) hub identity
for y, (_, r, _, col) in zip(ys, rows):
    v = r["hub_jaccard_vs_base"]
    c.hlines(y, v, 1.0, color=fs.tint(col, 0.7), lw=2, zorder=2)
    c.scatter(v, y, s=30, color=col, edgecolor=fs.SURF, lw=0.6, zorder=3)
c.plot([1, 1], [-0.1, 6.45], color=fs.INK3, lw=0.8, zorder=1)
c.set_xlim(0.66, 1.04); c.set_xticks([0.7, 0.8, 0.9, 1.0]); c.set_xticklabels(["0.7", "0.8", "0.9", "1"])
c.set_xlabel("top-32 hub Jaccard")
c.set_title("c   hub identity", fontsize=fs.SIZE_LABEL, loc="left", pad=16)

a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows])
for lab, r in zip(a.get_yticklabels(), rows): lab.set_color(r[3])
fig.text(0.012, 0.47, "Qwen3-1.7B", rotation=90, ha="left", va="center", fontsize=fs.SIZE_TICK, color=S.QWEN)
fig.text(0.012, 0.215, "Gemma", rotation=90, ha="left", va="center", fontsize=fs.SIZE_TICK, color=S.GEMMA)

svg = os.path.join(OUT, "fig3_base_control.svg")
fig.savefig(svg, bbox_inches=None)
print("wrote", svg, "envelope", env, "perm null", round(P["null_mean"], 4), "ci", [round(x, 3) for x in P["null_ci95"]])
