#!/usr/bin/env python3
"""Draw the four figures of the paper from the CSV files in results/.

Usage: python make_figures.py [--results results] [--out figures]
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--out", default="figures")
a = ap.parse_args()
R, FIG = a.results, a.out
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.6, "savefig.dpi": int(os.environ.get("FIG_DPI", 300))})
COLW = 3.23   # width of one journal column, inches
rd = lambda f: pd.read_csv(os.path.join(R, f))
e4 = rd("E4_attacks_chronological.csv")
e3s = rd("E3_fewshot_recalibration_summary.csv")

yr = rd("E0_year_by_class.csv")
fig, ax = plt.subplots(figsize=(COLW, 1.75))
x = np.arange(len(yr)); w = 0.4
ax.bar(x - w / 2, yr.ham, w, color="#9aa5b1", label="Ham")
ax.bar(x + w / 2, yr.spam, w, color="#2b5c8a", label="Spam")
ax.set_xticks(x, yr.date.astype(str)); ax.set_ylabel("Messages")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))
ax.set_ylim(0, 9800)
ax.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "years.png")); plt.close(fig)

# pipeline diagram
fig, ax = plt.subplots(figsize=(COLW, 2.7)); ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
def box(xc, yc, wd, ht, txt, fc="#eef2f6", ec="#2b5c8a", fs=8):
    ax.add_patch(FancyBboxPatch((xc - wd / 2, yc - ht / 2), wd, ht, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(xc, yc, txt, ha="center", va="center", fontsize=fs)
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#333333"))
box(5, 10.3, 4.2, 0.8, "E-mail (subject + body)", fc="white")
box(5, 8.9, 8.6, 1.1, "Normalisation: NFKC, homoglyph mapping,\njoining separated letters, leetspeak decoding")
box(2.6, 7.2, 4.2, 1.0, "Word 1-2-gram\nTF-IDF (200k)")
box(7.4, 7.2, 4.2, 1.0, "Char_wb 2-5-gram\nTF-IDF (300k)")
box(5, 5.55, 8.6, 1.1, "Logistic regression on concatenated features\n(trained with obfuscation-augmented spam)")
box(5, 3.85, 8.6, 1.1, "Recency-calibrated threshold θ\n(1% FPR on newest held-out ham)", fc="#fdf3e1", ec="#b7791f")
box(5, 2.3, 4.2, 0.8, "Spam / Ham", fc="white")
box(5, 0.7, 8.6, 1.0, "New mailbox: re-estimate θ from n legitimate\ne-mails (few-shot recalibration)", fc="#fdf3e1", ec="#b7791f", fs=7.5)
arr(5, 9.9, 5, 9.47); arr(3.6, 8.35, 2.8, 7.72); arr(6.4, 8.35, 7.2, 7.72)
arr(2.8, 6.68, 3.6, 6.12); arr(7.2, 6.68, 6.4, 6.12); arr(5, 5.0, 5, 4.42); arr(5, 3.3, 5, 2.72)
ax.add_patch(FancyArrowPatch((8.9, 1.2), (8.9, 3.4), arrowstyle="-|>", mutation_scale=7, lw=0.7, color="#b7791f",
                             connectionstyle="arc3,rad=0", linestyle="--"))
fig.tight_layout(pad=0.1); fig.savefig(os.path.join(FIG, "pipeline.png")); plt.close(fig)

# attacks figure
piv = e4.pivot(index="attack", columns="model", values="SpamRecall")
lines = [("LR (word)", "#7f7f7f", "o", "-"), ("NB (word)", "#bdbdbd", "s", "-"),
         ("HyRoS", "#2b5c8a", "D", "-"), ("HyRoS+GW", "#2b5c8a", "^", "--"), ("Monotone top-3", "#b7791f", "v", ":")]
fig, axs = plt.subplots(2, 1, figsize=(COLW, 2.8))
obf = ["clean", "obfuscation 10%", "obfuscation 20%", "obfuscation 30%", "obfuscation 50%"]
gw = ["clean", "good-word k=10", "good-word k=25", "good-word k=50"]
for axx, keys, xs, xl in ((axs[0], obf, [0, 10, 20, 30, 50], "Obfuscated words (%)"),
                          (axs[1], gw, [0, 10, 25, 50], "Appended good words (m)")):
    for mname, col, mk, ls in lines:
        axx.plot(xs, [100 * piv.loc[k, mname] for k in keys], color=col, marker=mk, ms=3, lw=1, ls=ls,
                 label=mname.replace(" (word)", ""))
    axx.set_xlabel(xl); axx.set_ylabel("Spam recall (%)"); axx.set_xticks(xs)
    axx.spines[["top", "right"]].set_visible(False); axx.grid(lw=0.3, alpha=0.5)
axs[0].set_ylim(0, 102); axs[1].set_ylim(0, 102)
axs[1].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "attacks.png")); plt.close(fig)

# few-shot recalibration figure
fs_ = e3s[e3s.model.isin(["LR (word)", "HyRoS"])]
fig, axs = plt.subplots(1, 2, figsize=(COLW, 1.75))
for mname, col, mk in (("LR (word)", "#7f7f7f", "o"), ("HyRoS", "#2b5c8a", "D")):
    d = fs_[fs_.model == mname]
    axs[0].errorbar(d.n_ham, 100 * d.FPR_mean, yerr=100 * d.FPR_std, color=col, marker=mk, ms=3, lw=1, capsize=2,
                    label=mname.replace(" (word)", ""))
    axs[1].errorbar(d.n_ham, 100 * d.Recall_mean, yerr=100 * d.Recall_std, color=col, marker=mk, ms=3, lw=1, capsize=2)
for axx, yl in ((axs[0], "Ham FPR (%)"), (axs[1], "Spam recall (%)")):
    axx.set_xscale("log"); axx.minorticks_off(); axx.set_xticks([25, 100, 500], ["25", "100", "500"])
    axx.set_xlabel("n (new-mailbox ham)"); axx.set_ylabel(yl)
    axx.spines[["top", "right"]].set_visible(False); axx.grid(lw=0.3, alpha=0.5)
axs[0].axhline(1, color="#b7791f", lw=0.8, ls="--"); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(pad=0.3); fig.savefig(os.path.join(FIG, "fewshot.png")); plt.close(fig)
print("figures written to", FIG)
