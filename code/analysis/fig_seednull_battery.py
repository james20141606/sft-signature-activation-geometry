#!/usr/bin/env python3
"""App. figure (\\textwidth): (a) the GRPO-family seed-noise-null strip on Qwen3-1.7B edge Jaccard;
(b) the narrowing battery as base->post relative deltas (%).

Revision R1 (2026-10-07): panel (a) reads graphs whose base and runs were built on ONE GPU
architecture (H200; results/sftband_20260926/run/graphs, recomputed by code/scripts/samearch.py):
the five GRPO-family runs that can still be rebuilt (the baseline@100 checkpoint is deleted), their
10 seed pairs, five base->RL contrasts and the audited base->SFT contrast. The submitted panel read
notes/circuit_rewiring_qwen_6seed.json, whose base and runs crossed architectures (archived; it
printed "s1: marginally below", an exceedance that lay within the build difference).
Panel (b) is unchanged: receipts notes/featstats_qwen_summary.json, featstats_gemma_summary.json,
tulu_featstats_summary.json (four metrics) and the per-checkpoint featstats_*_l22 / tulu_featstats_*_l25
files (activation mass per token; not recorded for the Gemma run).
Every number printed in panel (a) is asserted against the figure caption in main.tex."""
import os, re, sys
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(S.ROOT, "code", "scripts"))
import samearch

# ------------------------------------------------ (a) seed-noise-null strip, one architecture
F = samearch.same_arch()
G = F["grpo"]
null = np.array([1 - x for x in G["pairs"]]); assert len(null) == 10
names = [os.path.basename(r)[5:-5] for r in samearch.SAME["grpo"][1]]   # shaped150, s1_200, ...
rl = {n: 1 - c for n, c in zip(names, G["contrasts"])}
sft_j = 1 - F["sft_audited"]["contrasts"][0]
lo, hi = null.min(), null.max(); rng = hi - lo
sft_dist = lo - sft_j
n_below = sum(v < lo for v in rl.values())
z_sft = (sft_j - null.mean()) / null.std(ddof=1)
TEX = open(os.path.join(S.ROOT, "main.tex")).read()
want = ("on one architecture (H200), the $%d$ seed pairs of the five rebuildable GRPO-family runs, "
        "%d of the five base-to-RL contrasts below the band, and base-to-SFT at $z=%s$, "
        "$%.1f\\times$ the null range beyond it" % (len(null), n_below,
        ("%.1f" % z_sft).replace("-", "-"), sft_dist / rng))
assert " ".join(want.split()) in " ".join(TEX.split()), want

# ------------------------------------------------ (b) battery relative deltas (%)
def rel(a, b): return 100 * (b - a) / a
Q, Gm, T = (S.receipt(n)["rows"] for n in ("featstats_qwen_summary.json", "featstats_gemma_summary.json", "tulu_featstats_summary.json"))
mass = lambda n: S.receipt(n)["act_mass_per_token"]
metrics = ["mean_L0", "firing_freq_gini", "meanact_gini", "usage_entropy_norm"]
cols = [("Qwen\nSFT", Q["base"], Q["SFT"], "featstats_base_l22.json", "featstats_sft_l22.json"),
        ("Qwen\nGRPO", Q["base"], Q["GRPO"], "featstats_base_l22.json", "featstats_grpo_l22.json"),
        ("Gemma\n-it", Gm["base"], Gm["IT"], None, None),
        ("Tulu\nSFT", T["base"], T["SFT"], "tulu_featstats_base_l25.json", "tulu_featstats_sft_l25.json"),
        ("Tulu\nDPO", T["base"], T["DPO"], "tulu_featstats_base_l25.json", "tulu_featstats_dpo_l25.json"),
        ("Tulu\nPPO", T["base"], T["RLVR_PPO"], "tulu_featstats_base_l25.json", "tulu_featstats_rlvr_ppo_l25.json"),
        ("Tulu\nGRPO", T["base"], T["RLVR_GRPO"], "tulu_featstats_base_l25.json", "tulu_featstats_rlvr_grpo_l25.json")]
M = np.full((5, len(cols)), np.nan)
for j, (_, b, p, fb, fp) in enumerate(cols):
    for i, m in enumerate(metrics): M[i, j] = rel(b[m], p[m])
    if fb: M[4, j] = rel(mass(fb), mass(fp))
pinned = np.zeros_like(M, dtype=bool); pinned[0, :2] = True        # Qwen TopK k=50: mean L0 structurally pinned (Table battery, dagger)
assert abs(M[0, 2] - 7.45) < 0.05 and abs(M[4, 0] - 0.21) < 0.05 and abs(M[0, 3] + 0.27) < 0.05, M   # regression vs the previously hand-typed matrix
rows = ["mean L0", "firing-freq Gini", "mean-act Gini", "usage entropy", "act. mass / token"]

def fmt(v):
    """one decimal, true minus sign, and no signed zero (a value that rounds to 0.0 prints as 0.0)."""
    t = f"{v:+.1f}"
    return "0.0" if t in ("+0.0", "-0.0") else t.replace("-", "\u2212")

# ------------------------------------------------ draw
fs.use()
fig = plt.figure(figsize=(S.TEXTW, 3.15))
gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.12], wspace=0.46, left=0.065, right=0.93, top=0.9, bottom=0.30)
ax = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[0, 1])
S.tidy(ax, grid=None); ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.add_patch(plt.Rectangle((lo, -0.62), hi - lo, 2.37, facecolor=fs.tint(S.NULL, 0.25), edgecolor="none", zorder=0))   # stops above the ratio label
rs = np.random.default_rng(0); ax.scatter(null, rs.normal(0, 0.12, len(null)), s=14, color=S.NULL, alpha=0.8, edgecolor="none", zorder=2)
ax.text((lo + hi) / 2, 1.08, f"{len(null)} null pairs", ha="center", va="center", fontsize=fs.SIZE_TICK, color=fs.INK2, rotation=90)
for k, v in rl.items():
    c = S.SHAPED if k.startswith("shaped") else S.RL
    ax.vlines(v, -0.55, -0.15, color=c, lw=0.9, zorder=3)
ax.vlines(sft_j, -0.55, 0.55, color=S.SFT, lw=2.6, zorder=4)
ax.text(sft_j + 0.0012, 0.30, f"base→SFT\nz = {z_sft:.1f}".replace("-", "\u2212"), ha="left", va="center", fontsize=fs.SIZE_LABEL, color=S.SFT, fontweight="bold", linespacing=1.1)
yb = -0.98
ax.annotate("", (sft_j, yb), (lo, yb), arrowprops=dict(arrowstyle="<->", color=fs.INK2, lw=0.8, shrinkA=0, shrinkB=0))
ax.text((sft_j + lo) / 2, yb - 0.1, f"{sft_dist / rng:.1f}× the null range\n({n_below} of 5 RL contrasts below it)",
        ha="center", va="top", fontsize=fs.SIZE_TICK, color=fs.INK2, style="italic", linespacing=1.1)
ax.set_xlim(0.930, 0.972); ax.set_ylim(-1.75, 1.75); ax.set_xticks([0.93, 0.94, 0.95, 0.96, 0.97])
ax.set_xlabel("edge Jaccard vs. base (higher = less displaced)")
fs.panel_label(ax, "(a)", x=0.0, y=1.0)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=S.NULL, ms=3.5, label="null pair"),
                   Line2D([], [], color=S.RL, lw=1.2, label="base→RL"),
                   Line2D([], [], color=S.SHAPED, lw=1.2, label="shaped reward"),
                   Line2D([], [], color=S.SFT, lw=2.4, label="base→SFT")],
          loc="upper left", fontsize=fs.SIZE_TICK, handlelength=1.4, labelspacing=0.25, borderaxespad=0.1)

cmap = LinearSegmentedColormap.from_list("div", [fs.DIVERGING[0], fs.DIVERGING[1], fs.DIVERGING[2]])
im = bx.imshow(np.ma.masked_invalid(np.where(pinned, np.nan, M)), cmap=cmap, norm=TwoSlopeNorm(0, -8, 8), aspect="auto", interpolation="nearest")
bx.grid(False)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if pinned[i, j]: bx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fs.GRID, hatch="///", edgecolor=fs.SURF, lw=0)); bx.text(j, i, "TopK\nk=50", ha="center", va="center", fontsize=fs.SIZE_TICK, color=fs.INK3, linespacing=1.0)
        elif np.isnan(M[i, j]): bx.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fs.GRID, edgecolor=fs.SURF, lw=0)); bx.text(j, i, "n/a", ha="center", va="center", fontsize=fs.SIZE_TICK, color=fs.INK3)
        else: bx.text(j, i, fmt(M[i, j]), ha="center", va="center", fontsize=fs.SIZE_TICK, color=fs.SURF if abs(M[i, j]) > 4.5 else fs.INK)
bx.set_xticks(range(len(cols))); bx.set_xticklabels([c[0].split("\n")[1] for c in cols], fontsize=fs.SIZE_TICK)
# family names as group labels under the method names (seven two-line labels collided at print width)
import matplotlib.transforms as mtrans
_tr = mtrans.blended_transform_factory(bx.transData, bx.transAxes)
for _x0, _x1, _fam in ((0, 1, "Qwen"), (2, 2, "Gemma"), (3, 6, "Tulu")):
    bx.plot([_x0 - 0.38, _x1 + 0.38], [-0.13, -0.13], transform=_tr, color=fs.INK3, lw=0.6, clip_on=False)
    bx.text((_x0 + _x1) / 2, -0.155, _fam, transform=_tr, ha="center", va="top", fontsize=fs.SIZE_TICK, color=fs.INK2)
bx.set_yticks(range(len(rows))); bx.set_yticklabels(rows, fontsize=fs.SIZE_TICK)
bx.tick_params(length=0)
for s in bx.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=bx, fraction=0.05, pad=0.03, ticks=[-8, -4, 0, 4, 8]); cb.ax.tick_params(labelsize=fs.SIZE_TICK, length=2); cb.outline.set_visible(False)
cb.set_label("base→post relative Δ (%)", fontsize=fs.SIZE_TICK)
fs.panel_label(bx, "(b)", x=-0.1, y=1.0)
bx.set_xlabel("base→post checkpoint (Δ vs. its own base)", labelpad=24)
S.finish(fig, "fig_seednull_battery")
print(f"null [{lo:.4f}, {hi:.4f}] n={len(null)}; SFT {sft_j:.4f} z={z_sft:.2f}; dist/range={sft_dist/rng:.2f}; RL below band {n_below}/5")
