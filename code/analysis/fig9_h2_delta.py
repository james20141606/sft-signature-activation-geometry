#!/usr/bin/env python3
"""App. H2 per-feature delta diagnostic (\\textwidth), receipt notes/h2_diff_v0_cross_tokenizer.json (SFT vs
shaped-reward GRPO, shaped-lambda0.1 arm, n = 16 probes, 32 features). (a) pre vs post mean activation with
the four v0-panel top-attribution features highlighted (notes/h1_summary_v0.json top_features[:4]);
(b) histogram of post - pre deltas with those four marked."""
import numpy as np
import _sig as S; fs = S.fs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = S.receipt("h2_diff_v0_cross_tokenizer.json")
feats = d["f32_features"]; pre = np.array(d["mean_activation_pre"], float); post = np.array(d["mean_activation_post"], float)
delta = post - pre; cos = d["weighted_cosine"]; assert round(cos, 4) == 0.9998
tops = [t["feature_id"] for t in S.receipt("h1_summary_v0.json")["top_features"][:4]]; assert tops == [32255, 9666, 8286, 5158]
is_top = np.array([f in tops for f in feats]); assert is_top.sum() == 4 and (delta[is_top] < 0).all()

fs.use()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(S.TEXTW, 2.3), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.3})
S.tidy(ax); lim = max(pre.max(), post.max()) * 1.08
ax.plot([0, lim], [0, lim], ls=(0, (3, 2)), color=fs.INK3, lw=0.8, zorder=1)
ax.scatter(pre[~is_top], post[~is_top], s=16, color=S.QWEN, alpha=0.8, edgecolor="none", zorder=2)
ax.scatter(pre[is_top], post[is_top], s=34, marker="D", color=S.HEAVY, edgecolor=fs.INK, linewidth=0.5, zorder=3)
for f, x, y in zip(np.array(feats)[is_top], pre[is_top], post[is_top]):
    if x > 0.05: ax.annotate(f"#{f}", (x, y), (6, -9) if f != 9666 else (-7, -9), textcoords="offset points", ha="left" if f != 9666 else "right", fontsize=7, color=fs.INK2)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("mean activation, pre (SFT)"); ax.set_ylabel("mean activation, post (shaped GRPO)")
ax.set_title(f"(a) Pre vs. post per feature (cosine {cos:.4f})", pad=4)
ax.legend(handles=[Line2D([], [], ls="", marker="o", color=S.QWEN, ms=3.6, label="other top-32 features"),
                   Line2D([], [], ls="", marker="D", color=S.HEAVY, mec=fs.INK, mew=0.5, ms=4.2, label="v0-panel top-4 features"),
                   Line2D([], [], ls=(0, (3, 2)), color=fs.INK3, lw=0.8, label="y = x (no change)")],
          loc="upper left", fontsize=7, handlelength=1.4, labelspacing=0.3, borderaxespad=0.2)
S.tidy(bx)
bx.hist(delta[~is_top], bins=17, color=S.QWEN, alpha=0.85, edgecolor=fs.SURF, linewidth=0.5, zorder=2)
for x in delta[is_top]: bx.axvline(x, color=S.HEAVY, lw=1.4, zorder=3)
bx.axvline(0, color=fs.INK3, lw=0.8, ls=(0, (3, 2)), zorder=1)
bx.set_xlabel("Δ activation = post − pre"); bx.set_ylabel("# features")
bx.set_title(f"(b) Deltas, range [{delta.min():+.4f}, {delta.max():+.4f}]", pad=4)
bx.legend(handles=[Line2D([], [], color=S.HEAVY, lw=1.4, label="top-4 (all negative)")], loc="center right", fontsize=7, handlelength=1.4, borderaxespad=0.2)
fig.subplots_adjust(left=0.105, right=0.99, top=0.9, bottom=0.18)
S.finish(fig, "fig9_h2_delta")
print("cosine", cos, "delta range", delta.min(), delta.max(), "n_top_negative", int((delta[is_top] < 0).sum()))
