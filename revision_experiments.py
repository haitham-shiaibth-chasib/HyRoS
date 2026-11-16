#!/usr/bin/env python3
"""Additional analyses for the revised manuscript (MJCS-D-26-00031, major revision).

R0  de-duplication rules and the resulting Enron-Spam corpus size (comment 7)
R1  control with contemporaneous ham and spam: SpamAssassin, July-September 2002 (comment 1)
R2  exact and near-duplicate checks between Enron-Spam and SpamAssassin (comment 2)
R3  cross-corpus test on the stricter external set, default threshold and few-shot
    recalibration with 200 draws per calibration size (comments 3 and 8)
R4  chronological test set: every model at matched false-positive rates (1.0% and 1.5%)
    under every attack, bootstrap intervals, component effects with paired intervals and
    McNemar tests with Holm adjustment (comments 4, 5 and 8)
R5  LR (word) vs HyRoS at their recency-calibrated thresholds: full contingency tables,
    paired intervals for the FPR and recall differences, Holm adjustment (comment 8)

All randomness is seeded. Usage:
    python revision_experiments.py --enron data/enron_spam_data.csv \
        --spamassassin data/sa/package/data --out results_rev
"""
import argparse, email, glob, json, os, re, time, zlib
from email import policy
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import hyros_experiments as H

MAIN = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]
ALL8 = MAIN + ["HyRoS+GW", "Monotone top-3"]
PAIRS = [("LR (word+char)", "LR (word)", "A1 - LR: character n-grams"),
         ("LR (word+char+norm)", "LR (word+char)", "A2 - A1: normalisation"),
         ("HyRoS", "LR (word+char+norm)", "HyRoS - A2: obfuscation augmentation"),
         ("HyRoS", "LR (word)", "HyRoS - LR: all components")]
B = 2000


# ----------------------------------------------------------------------------- helpers
def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def boot_mean(v, rng, B=B):
    v = np.asarray(v, float)
    idx = rng.integers(0, len(v), (B, len(v)))
    m = v[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def boot_diff(a, b, rng, B=B):
    """Paired bootstrap interval of mean(a) - mean(b) over the same items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (B, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return np.percentile(d, [2.5, 97.5])


def mcnemar_full(correct_a, correct_b):
    ca, cb = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    both = int((ca & cb).sum()); a_only = int((ca & ~cb).sum())
    b_only = int((~ca & cb).sum()); neither = int((~ca & ~cb).sum())
    p = 1.0 if a_only + b_only == 0 else binomtest(a_only, a_only + b_only, 0.5).pvalue
    return both, a_only, b_only, neither, p


def thr_from_ham(s_ham, fpr):
    return H.thr_at_fpr(s_ham, fpr)


# ----------------------------------------------------------------------------- data
def load_sa_full(folder):
    """Same parsing and within-corpus de-duplication as H.load_spamassassin, but keeps the
    file id, the folder and the delivery time taken from the first Received header."""
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*", "*.json"))):
        group = os.path.basename(os.path.dirname(f))
        rec = json.load(open(f, encoding="utf-8"))
        try:
            text = H._email_to_text(rec["text"])
        except Exception:
            continue
        rd = pd.NaT
        try:
            m = email.message_from_string(rec["text"], policy=policy.compat32)
            rcv = m.get_all("Received") or []
            if rcv:
                x = parsedate_to_datetime(rcv[0].split(";")[-1].strip())
                rd = pd.Timestamp(x).tz_convert("UTC").tz_localize(None) if x.tzinfo else pd.Timestamp(x)
        except Exception:
            pass
        rows.append({"id": f"{group}/{os.path.basename(f)}", "text": text,
                     "y": int(group.startswith("spam")), "group": group, "received": rd})
    df = pd.DataFrame(rows)
    df = df[df.text.str.len() > 0].copy()
    df["key"] = df.text.str.lower().str.split().str.join(" ")
    df = df.drop_duplicates("key").drop(columns="key")
    return df.reset_index(drop=True)


TOK = re.compile(r"[a-z0-9]+")


def tokens(t):
    return TOK.findall(t.lower())


def shingles(toks, k=5):
    if len(toks) < k:
        return set()
    return {zlib.crc32(" ".join(toks[i:i + k]).encode()) for i in range(len(toks) - k + 1)}


# ----------------------------------------------------------------------------- R2
def near_duplicates(en, sa, say):
    from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble
    t0 = time.time()
    en_tok = [tokens(t) for t in en.text]; sa_tok = [tokens(t) for t in sa.text]
    en_sh = [shingles(t) for t in en_tok]; sa_sh = [shingles(t) for t in sa_tok]
    NP = 128

    def mh_all(sets):
        out = []
        for s in sets:
            m = MinHash(num_perm=NP, seed=1)
            if s:
                m.update_batch([x.to_bytes(4, "little") for x in s])
            out.append(m)
        return out
    en_mh, sa_mh = mh_all(en_sh), mh_all(sa_sh)
    say(f"R2 minhash built in {time.time() - t0:.0f}s")
    # exact match of the full token sequence, and the original 25-word prefix key
    en_full = {" ".join(t) for t in en_tok if t}
    exact = np.array([bool(t) and " ".join(t) in en_full for t in sa_tok])
    en_pref = {k for k in map(H.overlap_key, en.text) if k}
    prefix = np.array([H.overlap_key(t) in en_pref for t in sa.text])
    # Jaccard >= 0.5 on 5-word shingles
    lsh = MinHashLSH(threshold=0.5, num_perm=NP)
    for i, m in enumerate(en_mh):
        if en_sh[i]:
            lsh.insert(f"e{i}", m)
    jac = np.zeros(len(sa), bool); best_j = np.zeros(len(sa)); cont = np.zeros(len(sa), bool)
    for i, m in enumerate(sa_mh):
        if not sa_sh[i]:
            continue
        for key in lsh.query(m):
            j = int(key[1:]); a, b = sa_sh[i], en_sh[j]
            jj = len(a & b) / len(a | b)
            best_j[i] = max(best_j[i], jj)
            if jj >= 0.5:
                jac[i] = True
    # containment >= 0.8 in either direction (truncated, quoted or forwarded copies)
    for index_sets, index_mh, query_sets, query_mh, direction in (
            (en_sh, en_mh, sa_sh, sa_mh, "SA in Enron"), (sa_sh, sa_mh, en_sh, en_mh, "Enron in SA")):
        ens = MinHashLSHEnsemble(threshold=0.8, num_perm=NP, num_part=16)
        ens.index([(i, index_mh[i], len(s)) for i, s in enumerate(index_sets) if s])
        for q, s in enumerate(query_sets):
            if not s:
                continue
            for c in ens.query(query_mh[q], len(s)):
                inter = len(s & index_sets[c])
                if inter / len(s) >= 0.8 and len(s) >= 20:          # at least 20 shingles in common
                    if direction == "SA in Enron":
                        cont[q] = True
                    else:
                        cont[c] = True
        say(f"R2 containment {direction} done ({time.time() - t0:.0f}s)")
    return pd.DataFrame({"id": sa.id, "y": sa.y, "group": sa.group, "prefix25": prefix, "exact": exact,
                         "jaccard50": jac, "containment80": cont, "max_jaccard": best_j.round(3)})


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enron", required=True); ap.add_argument("--spamassassin", required=True)
    ap.add_argument("--out", default="results_rev"); ap.add_argument("--orig", default="results")
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "splits"), exist_ok=True)
    log = open(os.path.join(a.out, "revision_log.txt"), "w")

    def say(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    def save(d, n):
        d.to_csv(os.path.join(a.out, n), index=False)
    rng = np.random.default_rng(2026)
    grid = pd.read_csv(os.path.join(a.orig, "grid_search.csv"))
    C_word = float(grid[grid.model == "LR (word)"].sort_values("cv_F1").C.iloc[-1])
    C_hyb = float(grid[grid.model == "LR (word+char+norm)"].sort_values("cv_F1").C.iloc[-1])
    C_of = lambda n: C_word if n == "LR (word)" else C_hyb

    # ------------------------------------------------------------------ R0
    raw = pd.read_csv(a.enron)
    r = raw.copy(); r["Subject"] = r.Subject.fillna(""); r["Message"] = r.Message.fillna("")
    t = (r.Subject + " " + r.Message).str.strip(); nz = t.str.len() > 0
    rules = [
        ("rows in the single-file release", len(raw)),
        ("exact duplicates of subject+body removed", len(r.assign(k=r.Subject + "\n" + r.Message).drop_duplicates("k"))),
        ("empty messages removed, exact text duplicates removed", len(t[nz].drop_duplicates())),
        ("empty messages removed, duplicates after whitespace normalisation", len(t[nz].str.split().str.join(" ").drop_duplicates())),
        ("empty messages removed, duplicates after lower-casing and whitespace normalisation (this study)",
         len(t[nz].str.lower().str.split().str.join(" ").drop_duplicates())),
        ("empty messages removed, duplicates after keeping letters and digits only", len(t[nz].str.lower().str.replace(r"[^a-z0-9]", "", regex=True).drop_duplicates())),
        ("missing bodies dropped, duplicates of the body text removed", len(raw.dropna(subset=["Message"]).drop_duplicates("Message"))),
    ]
    save(pd.DataFrame(rules, columns=["rule", "messages"]), "R0_dedup_rules.csv")
    say("R0", rules)

    df = H.load_enron(a.enron)
    df["eid"] = np.arange(len(df))
    say(f"Enron-Spam: {len(df)}")

    # ------------------------------------------------------------------ split indices
    for seed in range(5):
        tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=seed)
        pd.DataFrame({"eid": te.eid.sort_values()}).to_csv(os.path.join(a.out, "splits", f"E1_seed{seed}_test_ids.csv"), index=False)
    trc, tec = H.chrono_split(df)
    fit_p, cal_p = [], []
    for _, g in trc.groupby("y"):
        g = g.sort_values("date"); c = int(len(g) * 0.9); fit_p.append(g.iloc[:c]); cal_p.append(g.iloc[c:])
    fitc, calc = pd.concat(fit_p), pd.concat(cal_p)
    part = pd.Series("test", index=df.index); part[fitc.index] = "fit"; part[calc.index] = "calibration"
    pd.DataFrame({"eid": df.eid, "y": df.y, "E2_part": part.values}).to_csv(os.path.join(a.out, "splits", "E2_parts.csv"), index=False)

    # ------------------------------------------------------------------ R2 near duplicates
    sa = load_sa_full(a.spamassassin)
    say(f"SpamAssassin after parsing and within-corpus de-duplication: {len(sa)}")
    nd = near_duplicates(df, sa, say)
    nd["any"] = nd[["prefix25", "exact", "jaccard50", "containment80"]].any(axis=1)
    save(nd, "R2_overlap_flags.csv")
    rows = []
    for col in ["prefix25", "exact", "jaccard50", "containment80", "any"]:
        rows.append({"check": col, "ham_flagged": int((nd[col] & (nd.y == 0)).sum()),
                     "spam_flagged": int((nd[col] & (nd.y == 1)).sum()),
                     "new_beyond_prefix25": int((nd[col] & ~nd.prefix25).sum())})
    r2 = pd.DataFrame(rows); save(r2, "R2_overlap_summary.csv"); say(r2.to_string(index=False))
    sa_strict = sa[~nd["any"].values].reset_index(drop=True)
    sa_prefix = sa[~nd["prefix25"].values].reset_index(drop=True)
    say(f"SA prefix-only set {len(sa_prefix)} (ham {int((sa_prefix.y==0).sum())}, spam {int((sa_prefix.y==1).sum())}); "
        f"strict set {len(sa_strict)} (ham {int((sa_strict.y==0).sum())}, spam {int((sa_strict.y==1).sum())}, "
        f"hard ham {int((sa_strict.group=='hard-ham-1').sum())})")
    pd.DataFrame({"id": sa_strict.id}).to_csv(os.path.join(a.out, "splits", "SA_strict_ids.csv"), index=False)

    # ------------------------------------------------------------------ R1 contemporaneous control
    win = sa[(sa.received >= "2002-07-01") & (sa.received < "2002-10-01")].copy().reset_index(drop=True)
    say(f"R1 window Jul-Sep 2002: {len(win)} (ham {int((win.y==0).sum())}, spam {int((win.y==1).sum())})")
    r1 = []

    def run_split(name_split, fit_df, cal_df, te_df, rep):
        for name in MAIN:
            m, _ = H.fit(name, fit_df.text, fit_df.y.values, rep, C_of(name))
            s_cal = H.scores(m, cal_df.text); s = H.scores(m, te_df.text); y = te_df.y.values
            thr_c = H.thr_at_fpr(s_cal[cal_df.y.values == 0], 0.01)
            for mode, thr in (("default", H.default_thr(m)), ("calibrated", thr_c)):
                p = (s >= thr).astype(int)
                r1.append({"split": name_split, "rep": rep, "model": name, "threshold": mode,
                           "FPR": p[y == 0].mean(), "Recall": p[y == 1].mean(), "AUC": roc_auc_score(y, s),
                           "n_test_ham": int((y == 0).sum()), "n_test_spam": int((y == 1).sum())})
    w = win.sort_values("received")
    cut = int(len(w) * 0.8); trw, tew = w.iloc[:cut], w.iloc[cut:]
    c2 = int(len(trw) * 0.9)
    run_split("joint chronological", trw.iloc[:c2], trw.iloc[c2:], tew, 0)
    say(f"R1 joint chronological: train to {trw.received.max()}, test from {tew.received.min()}, "
        f"test ham {int((tew.y==0).sum())}, spam {int((tew.y==1).sum())}")
    trs, tes, fs, cs = [], [], [], []
    for _, g in w.groupby("y"):
        cut = int(len(g) * 0.8); tr_, te_ = g.iloc[:cut], g.iloc[cut:]
        c2 = int(len(tr_) * 0.9); fs.append(tr_.iloc[:c2]); cs.append(tr_.iloc[c2:]); tes.append(te_)
    run_split("class-wise chronological", pd.concat(fs), pd.concat(cs), pd.concat(tes), 0)
    for seed in range(5):
        tr_, te_ = train_test_split(w, test_size=0.2, stratify=w.y, random_state=seed)
        f_, c_ = train_test_split(tr_, test_size=0.1, stratify=tr_.y, random_state=seed)
        run_split("random", f_, c_, te_, seed)
    r1 = pd.DataFrame(r1); save(r1, "R1_contemporaneous_all.csv")
    r1s = r1.groupby(["split", "model", "threshold"], sort=False)[["FPR", "Recall", "AUC"]].agg(["mean", "std"])
    r1s.columns = [f"{x}_{y}" for x, y in r1s.columns]; r1s = r1s.reset_index(); save(r1s, "R1_contemporaneous_summary.csv")
    say((r1s.set_index(["split", "model", "threshold"]) * 100).round(2).to_string())
    pd.DataFrame({"id": win.id, "received": win.received}).to_csv(os.path.join(a.out, "splits", "R1_window_ids.csv"), index=False)

    # ------------------------------------------------------------------ R3 cross-corpus on the strict set
    hard = (sa_strict.group == "hard-ham-1").values; ysa = sa_strict.y.values
    ham_idx = np.where(ysa == 0)[0]
    e3, e3r, e3m = [], [], []
    sa_scores = {}
    for name in MAIN:
        m, _ = H.fit(name, df.text, df.y.values, 0, C_of(name))
        s = H.scores(m, sa_strict.text); sa_scores[name] = s
        met, pred = H.metrics(ysa, s, H.default_thr(m), False)
        met["FPR_hard_ham"] = pred[hard].mean(); met["model"] = name; e3.append(met)
        for n in (25, 50, 100, 200, 500):
            for rep in range(a.draws):
                g = np.random.default_rng(10_000 * n + rep)
                ci = g.choice(ham_idx, n, replace=False)
                mask = np.ones(len(sa_strict), bool); mask[ci] = False
                thr = H.thr_at_fpr(s[ci], 0.01)
                yy, pp = ysa[mask], (s[mask] >= thr).astype(int)
                tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
                e3r.append({"model": name, "n_ham": n, "rep": rep, "FPR": fp / max((yy == 0).sum(), 1),
                            "Recall": tp / max((yy == 1).sum(), 1), "Precision": tp / max(tp + fp, 1),
                            "FPR_hard_ham": pp[hard[mask]].mean()})
        # matched-FPR analysis on the external corpus (threshold placed on all SA ham: analysis only)
        thr = H.thr_at_fpr(s[ysa == 0], 0.01)
        e3m.append({"model": name, "FPR": (s[ysa == 0] >= thr).mean(), "Recall": (s[ysa == 1] >= thr).mean(),
                    "pred_spam": (s[ysa == 1] >= thr)})
        say(f"R3 {name} done")
    e3 = pd.DataFrame(e3); save(e3, "R3_cross_corpus_strict.csv")
    say(e3[["model", "Accuracy", "Precision", "Recall", "FPR", "FPR_hard_ham", "AUC"]].round(4).to_string(index=False))
    e3r = pd.DataFrame(e3r); save(e3r, "R3_recalibration_all_draws.csv")
    q = lambda p: (lambda s: s.quantile(p))
    agg = e3r.groupby(["model", "n_ham"], sort=False).agg(
        FPR_mean=("FPR", "mean"), FPR_lo=("FPR", q(.025)), FPR_hi=("FPR", q(.975)),
        FPR_share_above_2pct=("FPR", lambda s: (s > 0.02).mean()),
        Recall_mean=("Recall", "mean"), Recall_lo=("Recall", q(.025)), Recall_hi=("Recall", q(.975)),
        Precision_mean=("Precision", "mean"), Precision_lo=("Precision", q(.025)), Precision_hi=("Precision", q(.975)),
        FPRhard_mean=("FPR_hard_ham", "mean")).reset_index()
    save(agg, "R3_recalibration_summary.csv"); say((agg.set_index(["model", "n_ham"]) * 100).round(2).to_string())
    m3 = pd.DataFrame([{k: v for k, v in d.items() if k != "pred_spam"} for d in e3m])
    preds3 = {d["model"]: d["pred_spam"] for d in e3m}
    rows = []
    for d_ in e3m:
        lo, hi = boot_mean(d_["pred_spam"], rng); rows.append({"model": d_["model"], "FPR": d_["FPR"], "Recall": d_["Recall"], "CI_lo": lo, "CI_hi": hi})
    for a_, b_, lab in PAIRS:
        lo, hi = boot_diff(preds3[a_], preds3[b_], rng)
        rows.append({"model": lab, "FPR": np.nan, "Recall": preds3[a_].mean() - preds3[b_].mean(), "CI_lo": lo, "CI_hi": hi})
    m3 = pd.DataFrame(rows); save(m3, "R3_cross_corpus_matched_fpr1.csv"); say((m3.set_index("model") * 100).round(2).to_string())

    # ------------------------------------------------------------------ R4 / R5 chronological test set
    yte = tec.y.values; ham = yte == 0; spam = yte == 1
    gw = H.ham_word_list(list(trc.text), trc.y.values)
    sets = H.attacked_sets(tec.text, yte, gw, 7)
    att_names = [n for n, _ in H.ATTACKS]
    S, thr_cal = {}, {}
    for name in ALL8:
        m, _ = H.fit(name, fitc.text, fitc.y.values, 0, C_of(name))
        s_cal = H.scores(m, calc.text); thr_cal[name] = H.thr_at_fpr(s_cal[calc.y.values == 0], 0.01)
        S[name] = {att: H.scores(m, X) for att, X in sets.items()}
        say(f"R4 {name} scored ({time.strftime('%H:%M:%S')})")
    r4, r4p = [], []
    for target in (0.01, 0.015):
        P = {}
        for name in ALL8:
            thr = H.thr_at_fpr(S[name]["clean"][ham], target)
            for att in att_names:
                p = S[name][att] >= thr; P[(name, att)] = p
                lo, hi = boot_mean(p[spam], rng)
                r4.append({"target_FPR": target, "model": name, "attack": att, "Recall": p[spam].mean(),
                           "CI_lo": lo, "CI_hi": hi, "FPR_clean_ham": (S[name]["clean"][ham] >= thr).mean()})
        for a_, b_, lab in PAIRS:
            ps = []
            for att in att_names:
                pa, pb = P[(a_, att)][spam], P[(b_, att)][spam]
                lo, hi = boot_diff(pa, pb, rng)
                both, ao, bo, nei, p = mcnemar_full(pa, pb)          # correct = detected spam
                ps.append(p)
                r4p.append({"target_FPR": target, "comparison": lab, "attack": att,
                            "diff_pp": 100 * (pa.mean() - pb.mean()), "CI_lo_pp": 100 * lo, "CI_hi_pp": 100 * hi,
                            "both_detect": both, "first_only": ao, "second_only": bo, "neither": nei, "p": p})
            adj = holm(ps)
            for k in range(len(att_names)):
                r4p[-len(att_names) + k]["p_holm"] = adj[k]
    r4 = pd.DataFrame(r4); save(r4, "R4_matched_fpr_all.csv")
    r4p = pd.DataFrame(r4p); save(r4p, "R4_component_effects.csv")
    say(r4[r4.target_FPR == 0.015].pivot(index="attack", columns="model", values="Recall").loc[att_names, ALL8].mul(100).round(1).to_string())
    say(r4p[r4p.target_FPR == 0.015].round(4).to_string(index=False))

    # R5: LR vs HyRoS at their calibrated thresholds (the operating points of Table 2 / old Table 4)
    r5 = []
    ps = []
    for att in att_names:
        pl = S["LR (word)"][att] >= thr_cal["LR (word)"]; ph = S["HyRoS"][att] >= thr_cal["HyRoS"]
        cl, ch = pl == (yte == 1), ph == (yte == 1)
        both, lo_, ho_, nei, p = mcnemar_full(cl, ch); ps.append(p)
        dr = boot_diff(ph[spam], pl[spam], rng)
        r5.append({"attack": att, "n_paired": len(yte), "both_correct": both, "LR_only": lo_, "HyRoS_only": ho_,
                   "both_wrong": nei, "p": p, "Recall_LR": pl[spam].mean(), "Recall_HyRoS": ph[spam].mean(),
                   "Recall_diff_pp": 100 * (ph[spam].mean() - pl[spam].mean()), "Recall_diff_lo": 100 * dr[0], "Recall_diff_hi": 100 * dr[1],
                   "FPR_LR": pl[ham].mean(), "FPR_HyRoS": ph[ham].mean()})
    adj = holm(ps)
    for k, row in enumerate(r5):
        row["p_holm"] = adj[k]
    pl = S["LR (word)"]["clean"] >= thr_cal["LR (word)"]; ph = S["HyRoS"]["clean"] >= thr_cal["HyRoS"]
    df_ = boot_diff(ph[ham], pl[ham], rng)
    r5 = pd.DataFrame(r5); save(r5, "R5_lr_vs_hyros_calibrated.csv")
    extra = pd.DataFrame([{"quantity": "FPR difference HyRoS - LR, clean chronological test ham (pp)",
                           "value": 100 * (ph[ham].mean() - pl[ham].mean()), "CI_lo": 100 * df_[0], "CI_hi": 100 * df_[1]}])
    for name in ALL8:
        p = S[name]["clean"] >= thr_cal[name]
        f_lo, f_hi = boot_mean(p[ham], rng); r_lo, r_hi = boot_mean(p[spam], rng)
        extra = pd.concat([extra, pd.DataFrame([
            {"quantity": f"{name}: FPR at calibrated threshold (%)", "value": 100 * p[ham].mean(), "CI_lo": 100 * f_lo, "CI_hi": 100 * f_hi},
            {"quantity": f"{name}: recall at calibrated threshold (%)", "value": 100 * p[spam].mean(), "CI_lo": 100 * r_lo, "CI_hi": 100 * r_hi}])])
    save(extra, "R5_intervals.csv")
    say(r5.round(4).to_string(index=False)); say(extra.round(3).to_string(index=False))
    say("done")


if __name__ == "__main__":
    main()
