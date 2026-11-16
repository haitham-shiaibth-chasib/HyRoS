#!/usr/bin/env python3
"""Three-fold grid search for the LR regularisation constant C on the E2 training (fit) set only.
Checks that C = 100, chosen on the seed-0 random split, is also selected when the search uses no message of the E2 test set.

Usage: python grid_search_e2.py --enron data/enron_spam_data.csv --parts results_rev/splits/E2_parts.csv --out results_rev3
"""
import argparse, os, time
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
import hyros_experiments as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enron", required=True); ap.add_argument("--parts", default="results_rev/splits/E2_parts.csv")
    ap.add_argument("--out", default="results_rev3")
    ap.add_argument("--models", nargs="+", default=["LR (word)", "LR (word+char+norm)"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    df = H.load_enron(a.enron)
    parts = pd.read_csv(a.parts)
    fitc = df[parts.E2_part.values == "fit"]
    rows = []
    for name in a.models:
        for C in (1.0, 10.0, 100.0):
            t = time.time()
            cv = StratifiedKFold(3, shuffle=True, random_state=0)
            f1 = cross_val_score(H.build(name, C), list(fitc.text), fitc.y, cv=cv, scoring="f1").mean()
            rows.append({"model": name, "C": C, "mean_F1_3fold": round(f1, 6), "n_fit": len(fitc)})
            print(name, C, round(f1, 6), f"{time.time() - t:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "R13_grid_search_e2.csv"), index=False)


if __name__ == "__main__":
    main()
