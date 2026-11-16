#!/usr/bin/env python3
"""Second set of revision analyses (R6-R10), added after an internal review of the first revision.

R6  E2 attacks at matched false-positive rates (1.5% and 1%) with a JOINT bootstrap that resamples
    both test ham and test spam and re-places every threshold in each replicate; HyRoS is trained
    with five seeds and each replicate draws one of them, so the intervals include training
    variability. Adds a held-out family of character edits that neither the augmentation nor the
    normaliser was built for, and a word-level LR with normalisation (LR+norm).
R7  E2 at default and recency-calibrated thresholds: exact (Clopper-Pearson) intervals, false
    positives by Enron mailbox, and results without the automated "HourAhead" messages.
R8  E3 on the strict SpamAssassin set: matched 1% FPR with the joint bootstrap, paired
    recalibration draws (HyRoS - LR), exact near-duplicate re-check of the remaining spam.
R9  E2c: false positives by SpamAssassin folder and without the RSS-feed sender that appears only
    in the test weeks; sensitivity to the 135 messages without a Received header.
R10 Beta(r, n - r + 1) distribution of the FPR given by the recalibration rule (continuous scores).

Usage: python revision_experiments2.py --enron data/enron_spam_data.csv \
          --spamassassin data/sa/package/data --out results_rev2
Model scores are cached in <out>/scores/ so that the analysis step can be re-run quickly.
"""
import argparse, email, glob, json, os, random, re, time
from email import policy
from email.utils import parsedate_to_datetime
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import beta, binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
import hyros_experiments as H
import revision_experiments as R

SEEDS = [0, 1, 2, 3, 4]
B = 2000
BASE = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+norm)", "LR (word+char)", "LR (word+char+norm)"]
HY = [f"HyRoS s{s}" for s in SEEDS]
EXTRA = ["HyRoS+GW", "Monotone top-3"]
HELD = ["held-out edits 50%", "held-out edits 100%", "held-out edits 30% + good-word k=25"]
PAIRS = [("LR (word+char)", "LR (word)", "A1 - LR: character n-grams"),
         ("LR (word+char+norm)", "LR (word+char)", "A2 - A1: normalisation after character n-grams"),
         ("HyRoS", "LR (word+char+norm)", "HyRoS - A2: obfuscation augmentation"),
         ("HyRoS", "LR (word)", "HyRoS - LR: all components"),
         ("LR (word+norm)", "LR (word)", "LR+norm - LR: normalisation alone"),
         ("LR (word+char+norm)", "LR (word+norm)", "A2 - LR+norm: character n-grams after normalisation")]


# ----------------------------------------------------------------------------- helpers
def fit_model(name, X, y):
    if name == "LR (word+norm)":
        m = Pipeline([("v", H.word_vec_norm()), ("c", LogisticRegression(C=100.0, solver="liblinear", max_iter=2000))])
        m.fit(list(X), list(y)); return m
    if name.startswith("HyRoS s"):
        m, _ = H.fit("HyRoS", X, y, int(name.split("s")[-1]), 100.0); return m
    m, _ = H.fit(name, X, y, 0, 100.0); return m


def thr(s_ham, fpr):
    return H.thr_at_fpr(s_ham, fpr)


def cp(k, n):
    """Exact Clopper-Pearson 95% interval for k successes in n trials (as percentages)."""
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return 100 * lo, 100 * hi


def holm(p):
    return R.holm(p)


def boot_p(d):
    """Two-sided bootstrap p-value of a difference (share of replicates on the other side of 0)."""
    d = np.asarray(d)
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / len(d))


ACC = {"a": "á", "e": "é", "i": "í", "o": "ö", "u": "ü"}
ALT = {"a": "ɑ", "i": "ı", "g": "ɡ", "l": "ӏ", "h": "һ", "n": "ո"}
ZW = "​"


def ho_word(w, r):
    """One edit from a held-out family: vowel deletion, accent, zero-width space or a look-alike
    glyph that the normaliser does not map."""
    op = r.choice(["vowel_drop", "accent", "zwsp", "alt_glyph"])
    inner = list(range(1, len(w) - 1))
    if op == "vowel_drop":
        v = [i for i in inner if w[i] in "aeiou"]
        if v:
            i = r.choice(v); return w[:i] + w[i + 1:]
        op = "accent"
    if op == "accent":
        v = [i for i in range(len(w)) if w[i] in ACC]
        if v:
            i = r.choice(v); return w[:i] + ACC[w[i]] + w[i + 1:]
        op = "zwsp"
    if op == "alt_glyph":
        v = [i for i in range(len(w)) if w[i] in ALT]
        if v:
            i = r.choice(v); return w[:i] + ALT[w[i]] + w[i + 1:]
    i = r.choice(inner); return w[:i] + ZW + w[i:]


def ho_obf(text, rate, r):
    return " ".join(ho_word(t, r) if (len(t) >= 4 and t.isalpha() and r.random() < rate) else t for t in text.split())


def sa_headers(folder, ids):
    """From address and Date header of the given SpamAssassin message ids."""
    out = {}
    for i in ids:
        rec = json.load(open(os.path.join(folder, i.split("/")[0], i.split("/")[1]), encoding="utf-8"))
        m = email.message_from_string(rec["text"], policy=policy.compat32)
        fr = str(m.get("From", "")); mm = re.search(r"[\w.\-+]+@[\w.\-]+", fr)
        d = pd.NaT
        try:
            x = parsedate_to_datetime(m.get("Date"))
            d = pd.Timestamp(x).tz_convert("UTC").tz_localize(None) if x.tzinfo else pd.Timestamp(x)
        except Exception:
            pass
        out[i] = (mm.group(0).lower() if mm else fr.lower(), d)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enron", required=True); ap.add_argument("--spamassassin", required=True)
    ap.add_argument("--out", default="results_rev2"); ap.add_argument("--prev", default="results_rev")
    ap.add_argument("--analysis-only", action="store_true")
    a = ap.parse_args()
    SC = os.path.join(a.out, "scores"); os.makedirs(SC, exist_ok=True)
    log = open(os.path.join(a.out, "revision2_log.txt"), "a")

    def say(*x):
        s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n"); log.flush()

    def save(d, n):
        d.to_csv(os.path.join(a.out, n), index=False)
    t0 = time.time()

    # ================================================================ data
    df = H.load_enron(a.enron); df["eid"] = np.arange(len(df))
    trc, tec = H.chrono_split(df)
    fp_, cp_ = [], []
    for _, g in trc.groupby("y"):
        g = g.sort_values("date"); c = int(len(g) * 0.9); fp_.append(g.iloc[:c]); cp_.append(g.iloc[c:])
    fitc, calc = pd.concat(fp_), pd.concat(cp_)
    parts = pd.read_csv(os.path.join(a.prev, "splits", "E2_parts.csv"))
    chk = pd.Series("test", index=df.index); chk[fitc.index] = "fit"; chk[calc.index] = "calibration"
    assert (chk.values == parts.E2_part.values).all(), "E2 split differs from splits/E2_parts.csv"
    yte = tec.y.values; ham = yte == 0; spam = yte == 1
    gw = H.ham_word_list(list(trc.text), trc.y.values)
    sets = H.attacked_sets(tec.text, yte, gw, 7)
    X = list(tec.text); si = np.where(spam)[0]; hi = np.where(ham)[0]
    SP = {n: [sets[n][i] for i in si] for n, _ in H.ATTACKS}
    r = random.Random(99); SP[HELD[0]] = [ho_obf(X[i], 0.5, r) for i in si]
    r = random.Random(101); SP[HELD[1]] = [ho_obf(X[i], 1.0, r) for i in si]
    r = random.Random(103); SP[HELD[2]] = [H.good_word(ho_obf(X[i], 0.3, r), gw, 25, r) for i in si]
    ATT = list(SP)
    hamX = [X[i] for i in hi]; calham = list(calc.text[calc.y.values == 0]); calspam = list(calc.text[calc.y.values == 1])
    say(f"E2: fit {len(fitc)}, calibration {len(calc)}, test ham {len(hamX)}, test spam {len(si)}; attacks {len(ATT)}")

    # ================================================================ E2 scores
    for name in BASE + HY + EXTRA:
        f = os.path.join(SC, "e2_" + re.sub(r"[^A-Za-z0-9]+", "_", name) + ".npz")
        if os.path.exists(f) or a.analysis_only:
            continue
        t1 = time.time(); m = fit_model(name, fitc.text, fitc.y.values)
        d = {"ham": H.scores(m, hamX), "calham": H.scores(m, calham), "calspam": H.scores(m, calspam)}
        for att in ATT:
            d["spam|" + att] = H.scores(m, SP[att])
        dt = H.default_thr(m); d["default_thr"] = np.array(np.nan if dt is None else dt)
        np.savez(f, **d); del m
        say(f"E2 scored {name}: {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)")

    def load_e2(name):
        z = np.load(os.path.join(SC, "e2_" + re.sub(r"[^A-Za-z0-9]+", "_", name) + ".npz"))
        return {k: z[k] for k in z.files}
    E2 = {n: load_e2(n) for n in BASE + HY + EXTRA}

    # ================================================================ E3 scores
    sa = R.load_sa_full(a.spamassassin)
    flags = pd.read_csv(os.path.join(a.prev, "R2_overlap_flags.csv"))
    assert (flags.id.values == sa.id.values).all()
    st = sa[~flags["any"].values].reset_index(drop=True)
    ysa = st.y.values
    for name in BASE + HY:
        f = os.path.join(SC, "e3_" + re.sub(r"[^A-Za-z0-9]+", "_", name) + ".npz")
        if os.path.exists(f) or a.analysis_only:
            continue
        t1 = time.time(); m = fit_model(name, df.text, df.y.values)
        dt = H.default_thr(m)
        np.savez(f, s=H.scores(m, st.text), default_thr=np.array(np.nan if dt is None else dt)); del m
        say(f"E3 scored {name}: {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)")
    E3 = {}
    for n in BASE + HY:
        z = np.load(os.path.join(SC, "e3_" + re.sub(r"[^A-Za-z0-9]+", "_", n) + ".npz")); E3[n] = {"s": z["s"], "dt": float(z["default_thr"])}

    # ================================================================ E2c scores (original window and with
    # the messages that have no Received header, dated by their Date header)
    hdr = sa_headers(a.spamassassin, list(sa.id))
    sa["from"] = [hdr[i][0] for i in sa.id]; sa["date_hdr"] = [hdr[i][1] for i in sa.id]
    win0 = sa[(sa.received >= "2002-07-01") & (sa.received < "2002-10-01")].copy()
    sa["when"] = sa.received.fillna(sa.date_hdr)
    win1 = sa[(sa.when >= "2002-07-01") & (sa.when < "2002-10-01")].copy(); win1["received"] = win1.when
    say(f"E2c window {len(win0)}; with Date-header fallback {len(win1)} (+{len(win1) - len(win0)})")
    E2C = {}
    for tag, win in (("orig", win0), ("datefallback", win1)):
        f = os.path.join(SC, f"e2c_{tag}.pkl")
        if os.path.exists(f) or a.analysis_only:
            E2C[tag] = pd.read_pickle(f); continue
        w = win.sort_values("received")          # same ordering as R1
        cut = int(len(w) * 0.8); trw, tew = w.iloc[:cut], w.iloc[cut:]; c2 = int(len(trw) * 0.9)
        fit_, cal_ = trw.iloc[:c2], trw.iloc[c2:]
        rows = []
        for name in ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS s0"]:
            m = fit_model(name, fit_.text, fit_.y.values)
            s_cal = H.scores(m, cal_.text); s = H.scores(m, tew.text)
            tc = thr(s_cal[cal_.y.values == 0], 0.01); dt = H.default_thr(m)
            for j, (idx, row) in enumerate(tew.iterrows()):
                rows.append({"model": name.replace(" s0", ""), "id": row.id, "group": row.group, "from": row["from"], "y": row.y,
                             "pred_default": int(s[j] >= dt), "pred_calibrated": int(s[j] >= tc)})
        E2C[tag] = pd.DataFrame(rows); E2C[tag].to_pickle(f)
        say(f"E2c {tag}: test {len(tew)} (ham {(tew.y == 0).sum()}), calibration ham {(cal_.y == 0).sum()}, "
            f"train to {trw.received.max()} ({time.time() - t0:.0f}s)")

    # ================================================================ R6 matched FPR, joint bootstrap
    rng = np.random.default_rng(2027)
    nh, ns = len(hamX), len(si)
    models6 = BASE + ["HyRoS"] + EXTRA
    r6, r6p, seedrows = [], [], []
    for target in (0.015, 0.01):
        # point estimates (HyRoS: mean over seeds)
        def recall_at(d, ih=None, is_=None):
            h = d["ham"] if ih is None else d["ham"][ih]
            t = thr(h, target)
            return {att: ((d["spam|" + att] if is_ is None else d["spam|" + att][is_]) >= t).mean() for att in ATT}, (h >= t).mean()
        point = {n: recall_at(E2[n])[0] for n in BASE + EXTRA}
        per_seed = {s: recall_at(E2[f"HyRoS s{s}"])[0] for s in SEEDS}
        point["HyRoS"] = {att: np.mean([per_seed[s][att] for s in SEEDS]) for att in ATT}
        for s in SEEDS:
            for att in ATT:
                seedrows.append({"target_FPR": target, "seed": s, "attack": att, "Recall_HyRoS": per_seed[s][att],
                                 "Recall_LR": point["LR (word)"][att], "Recall_A2": point["LR (word+char+norm)"][att]})
        # joint bootstrap with seed mixture
        BR = {n: np.zeros((B, len(ATT))) for n in models6}
        for b in range(B):
            ih = rng.integers(0, nh, nh); is_ = rng.integers(0, ns, ns); s = SEEDS[rng.integers(0, len(SEEDS))]
            for n in models6:
                d = E2[f"HyRoS s{s}"] if n == "HyRoS" else E2[n]
                rec, _ = recall_at(d, ih, is_)
                BR[n][b] = [rec[att] for att in ATT]
        for n in models6:
            for k, att in enumerate(ATT):
                lo, hi_ = np.percentile(BR[n][:, k], [2.5, 97.5])
                r6.append({"target_FPR": target, "model": n, "attack": att, "Recall": point[n][att], "CI_lo": lo, "CI_hi": hi_,
                           "n_seeds": 5 if n == "HyRoS" else 1})
        for a_, b_, lab in PAIRS:
            ps, rows_ = [], []
            for k, att in enumerate(ATT):
                dd = BR[a_][:, k] - BR[b_][:, k]
                lo, hi_ = np.percentile(dd, [2.5, 97.5]); p = boot_p(dd); ps.append(p)
                rows_.append({"target_FPR": target, "comparison": lab, "attack": att,
                              "diff_pp": 100 * (point[a_][att] - point[b_][att]), "CI_lo_pp": 100 * lo, "CI_hi_pp": 100 * hi_, "p_boot": p})
            for row, ph in zip(rows_, holm(ps)):
                row["p_holm"] = ph
            r6p += rows_
        say(f"R6 target {target} done ({time.time() - t0:.0f}s)")
    r6 = pd.DataFrame(r6); save(r6, "R6_matched_fpr_joint.csv")
    r6p = pd.DataFrame(r6p); save(r6p, "R6_component_effects_joint.csv")
    sd = pd.DataFrame(seedrows); save(sd, "R6_hyros_seeds.csv")
    say((r6[r6.target_FPR == 0.015].pivot(index="attack", columns="model", values="Recall").loc[ATT] * 100).round(1).to_string())
    say(r6p[r6p.target_FPR == 0.015].round(3).to_string(index=False))

    # ================================================================ R7 E2 at default / calibrated thresholds
    test_ham = tec[tec.y == 0]
    hourahead = test_ham.text.str.lower().str.contains("hourahead").values
    fit_ha = int(fitc[fitc.y == 0].text.str.lower().str.contains("hourahead").sum())
    cal_ha = int(calc[calc.y == 0].text.str.lower().str.contains("hourahead").sum())
    say(f"HourAhead messages: test ham {hourahead.sum()} of {len(test_ham)}; fit ham {fit_ha}; calibration ham {cal_ha}")
    rel = pd.read_csv(os.path.join(a.prev, "splits", "enron_eid_to_release_row.csv"))
    rowmap = dict(zip(rel.eid, rel.release_row))
    bounds = [0, 5172, 11029, 16541, 22541, 27716, 33716]
    boxes = ["farmer", "kaminski", "kitchen", "williams", "beck", "lokay"]
    mailbox = np.array([boxes[np.searchsorted(bounds, rowmap[e], side="right") - 1] for e in test_ham.eid])
    r7, r7box = [], []
    for n in BASE + HY + EXTRA:
        d = E2[n]; dt = float(d["default_thr"]); tc = thr(d["calham"], 0.01)
        for mode, t in (("default", dt), ("calibrated", tc)):
            if np.isnan(t):
                continue
            fpv = d["ham"] >= t; tpv = d["spam|clean"] >= t
            k, kn = int(fpv.sum()), int(fpv[~hourahead].sum())
            r7.append({"model": n, "threshold": mode, "FP": k, "n_ham": len(fpv), "FPR": 100 * k / len(fpv),
                       "FPR_lo": cp(k, len(fpv))[0], "FPR_hi": cp(k, len(fpv))[1],
                       "TP": int(tpv.sum()), "n_spam": len(tpv), "Recall": 100 * tpv.mean(),
                       "Recall_lo": cp(int(tpv.sum()), len(tpv))[0], "Recall_hi": cp(int(tpv.sum()), len(tpv))[1],
                       "FP_hourahead": k - kn, "FPR_without_hourahead": 100 * kn / int((~hourahead).sum())})
            for bx in boxes:
                mk = mailbox == bx
                r7box.append({"model": n, "threshold": mode, "mailbox": bx, "n_ham": int(mk.sum()), "FP": int(fpv[mk].sum())})
    r7 = pd.DataFrame(r7)
    hy = r7[r7.model.isin(HY)].groupby("threshold")[["FPR", "Recall", "FPR_without_hourahead", "FP_hourahead"]].agg(["mean", "min", "max"])
    save(r7, "R7_e2_thresholds.csv"); save(pd.DataFrame(r7box), "R7_e2_fp_by_mailbox.csv")
    hy.columns = [f"{x}_{y}" for x, y in hy.columns]; save(hy.reset_index(), "R7_e2_hyros_seed_summary.csv")
    say(r7[["model", "threshold", "FP", "FPR", "FPR_lo", "FPR_hi", "Recall", "FP_hourahead", "FPR_without_hourahead"]].round(2).to_string(index=False))
    # McNemar LR vs HyRoS at calibrated thresholds, all test messages, with and without HourAhead
    rows = []
    for s in SEEDS:
        dl, dh = E2["LR (word)"], E2[f"HyRoS s{s}"]
        tl, th = thr(dl["calham"], 0.01), thr(dh["calham"], 0.01)
        for subset, keep in (("all test ham", np.ones(len(hourahead), bool)), ("without HourAhead", ~hourahead)):
            cl = np.concatenate([dl["ham"][keep] < tl, dl["spam|clean"] >= tl]); ch = np.concatenate([dh["ham"][keep] < th, dh["spam|clean"] >= th])
            both, lo_, ho_, nei, p = R.mcnemar_full(cl, ch)
            rows.append({"seed": s, "ham_subset": subset, "n_paired": len(cl), "both_correct": both, "LR_only": lo_,
                         "HyRoS_only": ho_, "both_wrong": nei, "p": p})
    save(pd.DataFrame(rows), "R7_mcnemar_hourahead.csv"); say(pd.DataFrame(rows).to_string(index=False))

    # ================================================================ R8 E3 strict set
    hamI = np.where(ysa == 0)[0]; spamI = np.where(ysa == 1)[0]; hard = (st.group == "hard-ham-1").values
    r8 = []
    for n in BASE + HY:
        s, dt = E3[n]["s"], E3[n]["dt"]
        p = s >= dt; tp = int(p[spamI].sum()); fp = int(p[hamI].sum())
        r8.append({"model": n, "FPR": 100 * fp / len(hamI), "Recall": 100 * tp / len(spamI), "Precision": 100 * tp / max(tp + fp, 1),
                   "FPR_hard": 100 * p[hard].mean(), "AUC": roc_auc_score(ysa, s)})
    save(pd.DataFrame(r8), "R8_e3_default.csv")
    # recalibration draws (same calibration samples as R3), paired across models
    rec_rows = []
    for nsz in (25, 50, 100, 200, 500):
        for rep in range(200):
            g = np.random.default_rng(10_000 * nsz + rep); ci = g.choice(hamI, nsz, replace=False)
            mask = np.ones(len(st), bool); mask[ci] = False; yy = ysa[mask]
            row = {"n_ham": nsz, "rep": rep}
            for n in BASE + HY:
                s = E3[n]["s"]; t = thr(s[ci], 0.01); pp = s[mask] >= t
                tp = int((pp & (yy == 1)).sum()); fp = int((pp & (yy == 0)).sum())
                row[f"{n}|FPR"] = fp / (yy == 0).sum(); row[f"{n}|Recall"] = tp / (yy == 1).sum()
                row[f"{n}|Precision"] = tp / (tp + fp) if tp + fp > 0 else np.nan
            rec_rows.append(row)
    rc = pd.DataFrame(rec_rows); rc.to_csv(os.path.join(a.out, "R8_recalibration_draws.csv"), index=False)
    summ = []
    for nsz, g in rc.groupby("n_ham"):
        for n in BASE + ["HyRoS"]:
            cols = HY if n == "HyRoS" else [n]
            row = {"model": n, "n_ham": nsz}
            for k in ("FPR", "Recall", "Precision"):
                v = np.concatenate([g[f"{c}|{k}"].values for c in cols]); v = v[~np.isnan(v)]
                row[f"{k}_mean"], row[f"{k}_lo"], row[f"{k}_hi"] = v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5)
            v = np.concatenate([g[f"{c}|FPR"].values for c in cols]); row["FPR_share_above_2pct"] = (v > 0.02).mean()
            v = np.concatenate([g[f"{c}|Precision"].values for c in cols]); row["draws_flagging_nothing"] = int(np.isnan(v).sum())
            summ.append(row)
        d = np.concatenate([g[f"HyRoS s{s}|Recall"].values - g["LR (word)|Recall"].values for s in SEEDS])
        df_ = np.concatenate([g[f"HyRoS s{s}|FPR"].values - g["LR (word)|FPR"].values for s in SEEDS])
        summ.append({"model": "HyRoS - LR (paired draws)", "n_ham": nsz, "Recall_mean": d.mean(), "Recall_lo": np.percentile(d, 2.5),
                     "Recall_hi": np.percentile(d, 97.5), "FPR_mean": df_.mean(), "share_HyRoS_ahead": (d > 0).mean()})
    summ = pd.DataFrame(summ); save(summ, "R8_recalibration_summary.csv")
    say((summ.set_index(["model", "n_ham"]) * 100).round(2).to_string())
    # matched 1% FPR on all SpamAssassin ham, joint bootstrap with seed mixture
    def rec1(s, ih=None, is_=None):
        h = s[hamI] if ih is None else s[hamI][ih]; t = thr(h, 0.01)
        sp_ = s[spamI] if is_ is None else s[spamI][is_]
        return (sp_ >= t).mean()
    pt = {n: rec1(E3[n]["s"]) for n in BASE}; pt["HyRoS"] = np.mean([rec1(E3[h]["s"]) for h in HY])
    BR3 = {n: np.zeros(B) for n in BASE + ["HyRoS"]}
    for b in range(B):
        ih = rng.integers(0, len(hamI), len(hamI)); is_ = rng.integers(0, len(spamI), len(spamI)); s = SEEDS[rng.integers(0, 5)]
        for n in BASE + ["HyRoS"]:
            BR3[n][b] = rec1(E3[f"HyRoS s{s}"]["s"] if n == "HyRoS" else E3[n]["s"], ih, is_)
    m8 = []
    for n in BASE + ["HyRoS"]:
        lo, hi_ = np.percentile(BR3[n], [2.5, 97.5]); m8.append({"model": n, "Recall": 100 * pt[n], "CI_lo": 100 * lo, "CI_hi": 100 * hi_})
    for a_, b_, lab in PAIRS:
        dd = BR3[a_] - BR3[b_]; lo, hi_ = np.percentile(dd, [2.5, 97.5])
        m8.append({"model": lab, "Recall": 100 * (pt[a_] - pt[b_]), "CI_lo": 100 * lo, "CI_hi": 100 * hi_, "p_boot": boot_p(dd)})
    m8 = pd.DataFrame(m8); save(m8, "R8_e3_matched_fpr1_joint.csv"); say(m8.round(2).to_string(index=False))
    for s in SEEDS:
        say(f"E3 seed {s}: HyRoS recall at 1% = {100 * rec1(E3[f'HyRoS s{s}']['s']):.2f}")
    # exact near-duplicate re-check of the strict set
    sa_sh = [R.shingles(R.tokens(t)) for t in st.text]; voc = {}; rr, cc = [], []
    for i, s_ in enumerate(sa_sh):
        for h in s_:
            cc.append(voc.setdefault(h, len(voc))); rr.append(i)
    A = sp.csr_matrix((np.ones(len(rr), np.int32), (rr, cc)), shape=(len(st), len(voc))); asz = np.array([len(s_) for s_ in sa_sh])
    br, bc, bsz = [], [], []
    for j, t in enumerate(df.text):
        s_ = R.shingles(R.tokens(t)); bsz.append(len(s_))
        for h in s_:
            k = voc.get(h)
            if k is not None:
                br.append(j); bc.append(k)
    Bm = sp.csr_matrix((np.ones(len(br), np.int32), (br, bc)), shape=(len(df), len(voc))); bsz = np.array(bsz)
    I = (A @ Bm.T).tocsr(); maxJ = np.zeros(len(st)); cont = np.zeros(len(st), bool)
    for i in range(len(st)):
        js = I.indices[I.indptr[i]:I.indptr[i + 1]]; inter = I.data[I.indptr[i]:I.indptr[i + 1]].astype(float)
        if len(js) == 0 or asz[i] == 0:
            continue
        maxJ[i] = (inter / (asz[i] + bsz[js] - inter)).max()
        cont[i] = (asz[i] >= 20 and (inter / asz[i] >= 0.8).any()) or ((inter / np.maximum(bsz[js], 1) >= 0.8) & (bsz[js] >= 20)).any()
    nd = pd.DataFrame({"id": st.id, "y": ysa, "maxJ_exact": maxJ, "containment80_exact": cont}); save(nd, "R8_exact_near_duplicates.csv")
    resid = (maxJ >= 0.3) | cont
    rows = [{"check": "exact Jaccard >= 0.5", "ham": int(((maxJ >= 0.5) & (ysa == 0)).sum()), "spam": int(((maxJ >= 0.5) & (ysa == 1)).sum())},
            {"check": "exact Jaccard >= 0.3", "ham": int(((maxJ >= 0.3) & (ysa == 0)).sum()), "spam": int(((maxJ >= 0.3) & (ysa == 1)).sum())},
            {"check": "exact containment >= 0.8", "ham": int((cont & (ysa == 0)).sum()), "spam": int((cont & (ysa == 1)).sum())},
            {"check": "Jaccard >= 0.3 or containment", "ham": int((resid & (ysa == 0)).sum()), "spam": int((resid & (ysa == 1)).sum())}]
    keep_sp = spamI[~resid[spamI]]
    for n in ["LR (word)", "LR (word+char)", "LR (word+char+norm)"] + ["HyRoS"]:
        vals = []
        for h in (HY if n == "HyRoS" else [n]):
            s = E3[h]["s"]; t = thr(s[hamI], 0.01); vals.append(((s[keep_sp] >= t).mean(), (s[spamI] >= t).mean()))
        rows.append({"check": f"recall at 1% FPR, {n}: all strict spam / without residual overlap",
                     "ham": round(100 * np.mean([v[1] for v in vals]), 2), "spam": round(100 * np.mean([v[0] for v in vals]), 2)})
    save(pd.DataFrame(rows), "R8_residual_overlap.csv"); say(pd.DataFrame(rows).to_string(index=False))

    # ================================================================ R9 E2c by folder and sender
    r9 = []
    for tag, d in E2C.items():
        hm = d[d.y == 0]; rss = hm["from"].str.contains("rssfeeds@", regex=False)
        for (model, mode), g in [((m_, md), hm[hm.model == m_]) for m_ in hm.model.unique() for md in ("default", "calibrated")]:
            col = f"pred_{mode}"; k = int(g[col].sum()); n_ = len(g)
            gr = g[~g["from"].str.contains("rssfeeds@", regex=False)]
            row = {"window": tag, "model": model, "threshold": mode, "FP": k, "n_ham": n_, "FPR": 100 * k / n_,
                   "FPR_lo": cp(k, n_)[0], "FPR_hi": cp(k, n_)[1], "n_rss": int(len(g) - len(gr)),
                   "FP_rss": int(k - gr[col].sum()), "FPR_without_rss": 100 * gr[col].sum() / len(gr)}
            for grp in ("easy-ham-1", "easy-ham-2", "hard-ham-1"):
                gg = g[g.group == grp]; row[f"n_{grp}"] = len(gg); row[f"FP_{grp}"] = int(gg[col].sum())
            sp_ = d[(d.y == 1) & (d.model == model)]
            row["Recall"] = 100 * sp_[col].mean(); row["Recall_lo"], row["Recall_hi"] = cp(int(sp_[col].sum()), len(sp_))
            r9.append(row)
    r9 = pd.DataFrame(r9); save(r9, "R9_e2c_joint_chronological.csv")
    say(r9[["window", "model", "threshold", "FP", "n_ham", "FPR", "FPR_lo", "FPR_hi", "FP_rss", "FPR_without_rss", "FP_hard-ham-1", "n_hard-ham-1"]].round(2).to_string(index=False))
    # Table 2 intervals: exact CIs for the published E2c splits (counts from R1_contemporaneous_all.csv)
    r1 = pd.read_csv(os.path.join(a.prev, "R1_contemporaneous_all.csv"))
    rows = []
    for _, x in r1.iterrows():
        kf = int(round(x.FPR * x.n_test_ham)); kr = int(round(x.Recall * x.n_test_spam))
        rows.append({"split": x.split, "rep": x.rep, "model": x.model, "threshold": x.threshold,
                     "FP": kf, "n_ham": x.n_test_ham, "FPR_lo": cp(kf, x.n_test_ham)[0], "FPR_hi": cp(kf, x.n_test_ham)[1],
                     "TP": kr, "n_spam": x.n_test_spam, "Recall_lo": cp(kr, x.n_test_spam)[0], "Recall_hi": cp(kr, x.n_test_spam)[1]})
    save(pd.DataFrame(rows), "R9_e2c_exact_intervals.csv")

    # ================================================================ R10 recalibration rule, Beta distribution of the FPR
    rows = []
    for nsz in (25, 50, 100, 200, 500):
        r_ = nsz - int(np.ceil(0.99 * (nsz - 1)))      # rank from the top selected by quantile(method="higher")
        b_ = beta(r_, nsz - r_ + 1)
        rows.append({"n_ham": nsz, "rank_from_top": r_, "expected_FPR_pct": 100 * r_ / (nsz + 1),
                     "q2.5_pct": 100 * b_.ppf(0.025), "q97.5_pct": 100 * b_.ppf(0.975), "P_FPR_above_2pct": b_.sf(0.02)})
    save(pd.DataFrame(rows), "R10_recalibration_rule.csv"); say(pd.DataFrame(rows).round(3).to_string(index=False))
    say(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
