#!/usr/bin/env python3
"""E7: sensitivity analysis for E4 at a MATCHED false-positive rate.

E4 compares models at their own recency-calibrated thresholds, which give slightly different
false-positive rates on the test ham. Here every model's threshold is set so that its FPR on the
clean chronological test ham is exactly the same (1.5%), and spam recall under each attack is
recomputed. This uses test ham to place the threshold, so it is an analysis, not a deployable rule.

Usage: python e7_matched_fpr.py --enron data/enron_spam_data.csv --out results
"""
import argparse, os
import numpy as np, pandas as pd
import hyros_experiments as H

ap = argparse.ArgumentParser()
ap.add_argument("--enron", required=True); ap.add_argument("--out", default="results")
ap.add_argument("--fpr", type=float, default=0.015)
a = ap.parse_args()

df = H.load_enron(a.enron)
grid = pd.read_csv(os.path.join(a.out, "grid_search.csv"))
C_word = float(grid[grid.model == "LR (word)"].sort_values("cv_F1").C.iloc[-1])
C_hyb = float(grid[grid.model == "LR (word+char+norm)"].sort_values("cv_F1").C.iloc[-1])
trc, tec = H.chrono_split(df)
fit_p = []
for _, g in trc.groupby("y"):
    g = g.sort_values("date"); fit_p.append(g.iloc[:int(len(g) * 0.9)])
fitc = pd.concat(fit_p)
yte = tec.y.values; ham = yte == 0; spam = yte == 1
gw = H.ham_word_list(list(trc.text), trc.y.values)          # same attacker list as E4
sets = H.attacked_sets(tec.text, yte, gw, 7)                 # same attacked texts as E4
rows = []
for name in H.MODELS + ["HyRoS+GW", "Monotone top-3"]:
    m, _ = H.fit(name, fitc.text, fitc.y.values, 0, C_word if name == "LR (word)" else C_hyb)
    s0 = H.scores(m, sets["clean"])
    thr = H.thr_at_fpr(s0[ham], a.fpr)
    for att, X in sets.items():
        s = s0 if att == "clean" else H.scores(m, X)
        rows.append({"attack": att, "model": name, "SpamRecall": (s[spam] >= thr).mean(),
                     "FPR": (s[ham] >= thr).mean()})
    print(name, "done", flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(a.out, "E7_matched_fpr.csv"), index=False)
piv = out.pivot(index="attack", columns="model", values="SpamRecall").loc[[n for n, _ in H.ATTACKS]]
piv.reset_index().to_csv(os.path.join(a.out, "E7_matched_fpr_pivot.csv"), index=False)
print((100 * piv).round(1).to_string())
print(out[out.attack == "clean"][["model", "FPR"]].to_string(index=False))
