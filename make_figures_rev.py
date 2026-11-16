#!/usr/bin/env python3
"""Figs. 2 and 3 of the revised manuscript (Fig. 1 and Fig. S1 come from make_figures.py).

Fig. 2: few-shot recalibration on the strict external set (FPR, recall, precision) with the 2.5th-97.5th
percentiles of the shared draws. Fig. 3: spam recall under attack with every threshold set to flag 1.5% of
the clean E2 test ham, with joint-bootstrap 95% intervals; legend below the panels.

Usage: FIG_DPI=600 python make_figures_rev.py [--results results_rev2] [--out figures]
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results_rev2"); ap.add_argument("--out", default="figures")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.6, "savefig.dpi": int(os.environ.get("FIG_DPI", 300))})
rd = lambda f: pd.read_csv(os.path.join(a.results, f))
STY = {"LR (word)": ("LR (word)", "#8c8c8c", "o", "-"),
       "NB (word)": ("NB (word)", "#bdbdbd", "s", "-"),
       "LR (word+char)": ("A1: LR (word+char)", "#6baed6", "x", "-"),
       "LR (word+norm)": ("LR+norm", "#9e7bb5", "s", "-."),
       "HyRoS": ("HyRoS", "#2b5c8a", "D", "-"),
       "HyRoS+GW": ("HyRoS+GW", "#2b5c8a", "^", "--"),
       "Monotone top-3": ("Monotone top-3", "#b7791f", "v", ":")}

# ---------------- Fig. 3: few-shot recalibration with uncertainty ----------------
s = rd("R8_recalibration_summary.csv")
fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.1))
for m in ("LR (word)", "LR (word+char)", "HyRoS"):
    lab, col, mk, ls = STY[m]
    d = s[s.model == m].sort_values("n_ham"); x = d.n_ham.values.astype(int)
    for ax, k in zip(axs, ("FPR", "Recall", "Precision")):
        ax.plot(x, 100 * d[f"{k}_mean"], color=col, marker=mk, ms=3, lw=1, ls=ls, label=lab)
        if m != "LR (word+char)":   # bands for LR and HyRoS only, to keep the panels legible
            ax.fill_between(x, 100 * d[f"{k}_lo"], 100 * d[f"{k}_hi"], color=col, alpha=0.18 if m == "HyRoS" else 0.22, lw=0)
            ax.plot(x, 100 * d[f"{k}_lo"], color=col, lw=0.5, ls=":"); ax.plot(x, 100 * d[f"{k}_hi"], color=col, lw=0.5, ls=":")
for ax, yl in zip(axs, ("Ham false-positive rate (%)", "Spam recall (%)", "Precision (%)")):
    ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks([25, 50, 100, 200, 500], ["25", "50", "100", "200", "500"])
    ax.set_xlabel("n (calibration ham)"); ax.set_ylabel(yl)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(lw=0.3, alpha=0.5)
axs[0].axhline(1, color="#b7791f", lw=0.8, ls="--")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8)
fig.tight_layout(rect=(0, 0.1, 1, 1), pad=0.3); fig.savefig(os.path.join(a.out, "fewshot.png")); plt.close(fig)

# ---------------- Fig. 3: attacks at a matched FPR, legend outside ----------------
r4 = rd("R6_matched_fpr_joint.csv"); r4 = r4[np.isclose(r4.target_FPR, 0.015)]
obf = [("clean", 0), ("obfuscation 10%", 10), ("obfuscation 20%", 20), ("obfuscation 30%", 30), ("obfuscation 50%", 50)]
ho = [("clean", 0), ("held-out edits 50%", 50), ("held-out edits 100%", 100)]
gw = [("clean", 0), ("good-word k=10", 10), ("good-word k=25", 25), ("good-word k=50", 50)]
fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.5))
for ax, keys, xl, yl in ((axs[0], obf, "Words obfuscated, known edits (%)", (88, 100)),
                         (axs[1], ho, "Words obfuscated, held-out edits (%)", (88, 100)),
                         (axs[2], gw, "Appended good words (m)", (0, 102))):
    for m in ("LR (word)", "NB (word)", "LR (word+norm)", "LR (word+char)", "HyRoS", "HyRoS+GW", "Monotone top-3"):
        lab, col, mk, ls = STY[m]
        d = r4[r4.model == m].set_index("attack").loc[[k for k, _ in keys]]
        xs = [x for _, x in keys]; y = 100 * d.Recall.values
        err = np.vstack([y - 100 * d.CI_lo.values, 100 * d.CI_hi.values - y])
        ax.errorbar(xs, y, yerr=err, color=col, marker=mk, ms=3, lw=1, ls=ls, capsize=1.5, elinewidth=0.6, label=lab)
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel("Spam recall (%)"); ax.set_xticks([x for _, x in keys]); ax.set_ylim(*yl)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(lw=0.3, alpha=0.5)
axs[0].set_title("(a)", fontsize=9); axs[1].set_title("(b)", fontsize=9); axs[2].set_title("(c)", fontsize=9)
h, l = axs[2].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
fig.tight_layout(rect=(0, 0.14, 1, 1), pad=0.3); fig.savefig(os.path.join(a.out, "attacks.png")); plt.close(fig)
print("figures written to", a.out)
