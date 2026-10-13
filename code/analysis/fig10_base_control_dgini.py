#!/usr/bin/env python3
"""Base-control dot plot with an observed Qwen envelope (not a null band)."""
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

Q, Gm = (S.receipt(n)["rows"] for n in
         ("basediff_qwen_summary.json", "basediff_gemma_summary.json"))
rows = [
    ("SFT (LoRA)", Q["SFT"], S.QWEN, "Qwen"),
    ("GRPO (accuracy-only)", Q["GRPO"], S.QWEN, "Qwen"),
    ("GRPO seed s4", Q["GRPO_s4"], S.QWEN, "Qwen"),
    ("GRPO seed s5", Q["GRPO_s5"], S.QWEN, "Qwen"),
    ("shaped-λ0.1 GRPO", Q["SHAPED_l0.1"], S.QWEN, "Qwen"),
    ("gemma-2-2b-it\n(heavy SFT+RLHF)", Gm["IT"], S.GEMMA, "Gemma")]
qwen_envelope = max(abs(r["dGini_vs_base"]) for _, r, _, family in rows if family == "Qwen")
assert abs(qwen_envelope - 0.005) < 1e-12
assert abs(Gm["IT"]["dGini_vs_base"] - 0.035) < 1e-12

fs.use()
fig, ax = plt.subplots(figsize=(S.TEXTW, 2.65))
S.tidy(ax, grid="x")
y = [6, 5, 4, 3, 2, 0.45]
ax.fill_betweenx([1.55, 6.45], -qwen_envelope, qwen_envelope,
                 color=fs.tint(S.NULL, 0.22), zorder=0)
ax.plot([0, 0], [-0.2, 6.45], color=fs.INK3, lw=0.8, zorder=1)  # stop below the header so the zero line never crosses text
for yi, (label, row, color, _) in zip(y, rows):
    value = row["dGini_vs_base"]
    ax.hlines(yi, min(0, value), max(0, value), color=fs.tint(color, 0.75),
              lw=2.2, zorder=2)
    ax.scatter(value, yi, s=40, color=color, edgecolor=fs.SURF,
               linewidth=0.7, zorder=3)
    ax.text(max(value, 0) + 0.0022, yi,
            f"ΔGini {value:+.3f}   hub Jaccard {row['hub_jaccard_vs_base']:.2f}".replace("-", "\u2212"),
            va="center", ha="left", fontsize=fs.SIZE_TICK, color=fs.INK2)
ax.text(0, 6.58, "observed envelope across five Qwen light-recipe runs\n|ΔGini| ≤ 0.005",
        ha="center", va="bottom", fontsize=fs.SIZE_TICK, color=fs.INK2,
        fontweight="bold", linespacing=1.05)
ax.axhline(1.25, color=fs.GRID, lw=0.8)
ax.text(-0.0114, 1.1, "different family + heavy recipe", ha="left", va="top",
        fontsize=fs.SIZE_TICK, color=fs.INK3, style="italic")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=fs.SIZE_TICK,
                                     linespacing=1.0)
for lab, (_, _, color, _) in zip(ax.get_yticklabels(), rows): lab.set_color(color)
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.012, 0.075); ax.set_ylim(-0.2, 7.05)
ax.set_xlabel("base→post change in attribution-graph concentration (ΔGini)")
ax.legend(handles=[
    Line2D([], [], marker="o", ls="", color=S.QWEN, ms=5,
           label="Qwen3-1.7B / qwen-scope"),
    Line2D([], [], marker="o", ls="", color=S.GEMMA, ms=5,
           label="Gemma-2-2B / gemma-scope")],
    loc="upper right", fontsize=fs.SIZE_TICK, handletextpad=0.4,
    labelspacing=0.25, borderaxespad=0.2)
fig.subplots_adjust(left=0.25, right=0.985, top=0.89, bottom=0.19)
S.finish(fig, "fig10_base_control_dgini")
print("observed Qwen envelope", qwen_envelope)
