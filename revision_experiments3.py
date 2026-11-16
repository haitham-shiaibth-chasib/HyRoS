#!/usr/bin/env python3
"""Third set of revision analyses (sensitivity checks). Uses the cached scores of revision_experiments2.py.

R11  How the HourAhead notices affect the matched-FPR component effects of Table 5:
     number of HourAhead notices among the ham flagged at the matched 1.5% and 1% rates, and the component
     effects recomputed (a) with a cluster bootstrap in which the 774 HourAhead notices form one cluster and
     (b) as recall averaged over matched false-positive rates from 0.5% to 3% (11 points).
     Paired interval for the E2 calibrated false-positive difference HyRoS - LR.
R12  Overlap checks: the de-duplication rule stated by the replicated study, and a subject plus word-set
     check for further Enron-Spam copies among the strict SpamAssassin spam, with external recall at a
     matched 1% false-positive rate after excluding them. Rebuilds the eid -> release-row map.

Usage:
  python revision_experiments3.py --enron data/enron_spam_data.csv --spamassassin data/sa/package/data \
         --rev results_rev --rev2 results_rev2 --out results_rev3
"""
import argparse, email, json, os, re, time
from collections import defaultdict
from email import policy

import numpy as np
import pandas as pd

import hyros_experiments as H
import revision_experiments as R

B_CLUSTER, B_AVG = 2000, 1000
GRID = np.linspace(0.005, 0.03, 11)
BASE = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+norm)", "LR (word+char)", "LR (word+char+norm)"]
HY = [f"HyRoS s{s}" for s in range(5)]
PAIRS = [("LR (word+char)", "LR (word)", "A1 - LR: character n-grams"),
         ("LR (word+char+norm)", "LR (word+char)", "A2 - A1: normalisation after character n-grams"),
         ("HyRoS", "LR (word+char+norm)", "HyRoS - A2: obfuscation augmentation"),
         ("HyRoS", "LR (word)", "HyRoS - LR: all components"),
         ("LR (word+norm)", "LR (word)", "LR+norm - LR: normalisation alone"),
         ("LR (word+char+norm)", "LR (word+norm)", "A2 - LR+norm: character n-grams after normalisation")]
TOK = re.compile(r"[a-z0-9]+")
SUBJ_STOP = {"re", "fw", "fwd", "adv", "spam", "sa", "ilug", "social"}


def fname(n):
    return re.sub(r"[^A-Za-z0-9]+", "_", n)


def boot_p(d):
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1.0 / len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enron", required=True); ap.add_argument("--spamassassin", required=True)
    ap.add_argument("--rev", default="results_rev"); ap.add_argument("--rev2", default="results_rev2")
    ap.add_argument("--out", default="results_rev3")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    SC = os.path.join(a.rev2, "scores"); t0 = time.time()
    save = lambda d, f: d.to_csv(os.path.join(a.out, f), index=False)

    # ================================================================ eid -> release row (provenance of the split file)
    raw = pd.read_csv(a.enron)
    sub, msg = raw["Subject"].fillna(""), raw["Message"].fillna("")
    text = (sub + " " + msg).str.strip()
    keep = text.str.len() > 0
    key = text[keep].str.lower().str.split().str.join(" ")
    rows_kept = key[~key.duplicated()].index.values
    emap = pd.DataFrame({"eid": np.arange(len(rows_kept)), "release_row": rows_kept, "Message ID": raw["Message ID"].values[rows_kept]})
    f = os.path.join(a.rev, "splits", "enron_eid_to_release_row.csv")
    if os.path.exists(f):
        assert (pd.read_csv(f).values == emap.values).all(), "eid map differs"
    else:
        os.makedirs(os.path.dirname(f), exist_ok=True); emap.to_csv(f, index=False)

    # ================================================================ R12a de-duplication rule of the replicated study
    both_empty = (sub.str.strip() == "") & (msg.str.strip() == "")
    d = raw[~both_empty]
    c = ((d["Subject"].fillna("") + " " + d["Message"].fillna("")).str.lower()
         .str.replace(r"[^\w\s\$%]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip())
    u = d.assign(c=c).drop_duplicates("c")
    u2 = u[u.c.str.len() > 0]
    rows = [{"rule": "replicated study: lower-case, keep letters, digits, $ and %, collapse whitespace", "kept": len(u),
             "ham": int((u["Spam/Ham"] == "ham").sum()), "spam": int((u["Spam/Ham"] == "spam").sum())},
            {"rule": "same, also dropping messages left empty after cleaning", "kept": len(u2),
             "ham": int((u2["Spam/Ham"] == "ham").sum()), "spam": int((u2["Spam/Ham"] == "spam").sum())}]
    save(pd.DataFrame(rows), "R12_dedup_replicated_rule.csv"); print(pd.DataFrame(rows).to_string(index=False))

    # ================================================================ E2 scores and HourAhead mask
    df = H.load_enron(a.enron)
    trc, tec = H.chrono_split(df)
    hamtxt = tec.text[tec.y.values == 0]
    ha = hamtxt.str.lower().str.contains("hourahead").values
    E = {}
    for n in BASE + HY:
        z = np.load(os.path.join(SC, "e2_" + fname(n) + ".npz")); E[n] = {k: z[k] for k in z.files}
    assert len(E["LR (word)"]["ham"]) == len(ha)
    ATT = [k[5:] for k in E["LR (word)"] if k.startswith("spam|")]
    S = {n: np.vstack([E[n]["spam|" + t] for t in ATT]) for n in BASE + HY}
    Hm = {n: E[n]["ham"] for n in BASE + HY}
    thr = H.thr_at_fpr

    rows = []
    for tgt in (0.015, 0.01):
        for n in BASE + HY:
            fl = Hm[n] >= thr(Hm[n], tgt)
            rows.append({"target_FPR": tgt, "model": n, "ham_flagged": int(fl.sum()), "hourahead_flagged": int((fl & ha).sum())})
    save(pd.DataFrame(rows), "R11_hourahead_flagged.csv"); print(pd.DataFrame(rows).to_string(index=False))

    # ================================================================ R11 sensitivity of the component effects
    MODELS = BASE + ["HyRoS"]
    nh, ns = len(ha), S["LR (word)"].shape[1]
    nonha, haidx = np.where(~ha)[0], np.where(ha)[0]

    def stat(m, hidx, is_, target):
        h = Hm[m][hidx]; Ss = S[m] if is_ is None else S[m][:, is_]
        if target == "avg":
            return np.mean([(Ss >= thr(h, g)).mean(1) for g in GRID], 0)
        return (Ss >= thr(h, target)).mean(1)

    def run(label, target, sampler, B, seed=2027):
        rng = np.random.default_rng(seed); allh = np.arange(nh)
        point = {n: stat(n, allh, None, target) for n in BASE + HY}; point["HyRoS"] = np.mean([point[h] for h in HY], 0)
        BR = {n: np.zeros((B, len(ATT))) for n in MODELS}
        for b in range(B):
            ih = sampler(rng); is_ = rng.integers(0, ns, ns); s = rng.integers(0, 5)
            for n in MODELS:
                BR[n][b] = stat(f"HyRoS s{s}" if n == "HyRoS" else n, ih, is_, target)
        out = []
        for x, y, lab in PAIRS:
            rr, ps = [], []
            for k, att in enumerate(ATT):
                dd = BR[x][:, k] - BR[y][:, k]; lo, hi = np.percentile(dd, [2.5, 97.5]); p = boot_p(dd); ps.append(p)
                rr.append({"analysis": label, "comparison": lab, "attack": att, "diff_pp": 100 * (point[x][k] - point[y][k]),
                           "CI_lo_pp": 100 * lo, "CI_hi_pp": 100 * hi, "p_boot": p})
            for r_, ph in zip(rr, R.holm(ps)):
                r_["p_holm"] = ph
            out += rr
        print(f"R11 {label} done ({time.time() - t0:.0f}s)", flush=True)
        return pd.DataFrame(out)

    def cluster(rng):
        k = rng.integers(0, len(nonha) + 1, len(nonha) + 1)
        return np.concatenate([nonha[k[k < len(nonha)]]] + [haidx] * int((k == len(nonha)).sum()))

    iid = lambda rng: rng.integers(0, nh, nh)
    res = pd.concat([run("cluster bootstrap, HourAhead as one cluster, matched 1.5%", 0.015, cluster, B_CLUSTER),
                     run("recall averaged over matched FPR 0.5-3%", "avg", iid, B_AVG)])
    save(res, "R11_component_sensitivity.csv")

    # paired interval for the E2 calibrated FPR difference HyRoS - LR (thresholds from the calibration ham)
    rng = np.random.default_rng(2028); rows = []
    for subset, idx in (("all test ham", np.arange(nh)), ("without HourAhead", nonha)):
        fl = {n: Hm[n] >= thr(E[n]["calham"], 0.01) for n in ["LR (word)"] + HY}
        pt = np.mean([fl[h][idx].mean() for h in HY]) - fl["LR (word)"][idx].mean()
        bd = []
        for _ in range(B_CLUSTER):
            ib = idx[rng.integers(0, len(idx), len(idx))]; s = rng.integers(0, 5)
            bd.append(fl[f"HyRoS s{s}"][ib].mean() - fl["LR (word)"][ib].mean())
        bd = np.array(bd)
        rows.append({"ham_subset": subset, "n_ham": len(idx), "diff_pp": 100 * pt, "CI_lo_pp": 100 * np.percentile(bd, 2.5),
                     "CI_hi_pp": 100 * np.percentile(bd, 97.5), "p_boot": boot_p(bd)})
    save(pd.DataFrame(rows), "R11_e2_calibrated_fpr_difference.csv"); print(pd.DataFrame(rows).round(3).to_string(index=False))

    # McNemar of R7 split into ham and spam (calibrated thresholds, HyRoS seed 0)
    rows = []
    dl, dh = E["LR (word)"], E["HyRoS s0"]
    tl, th = thr(dl["calham"], 0.01), thr(dh["calham"], 0.01)
    for part, cl, ch in (("test ham without HourAhead", dl["ham"][~ha] < tl, dh["ham"][~ha] < th),
                         ("test ham, all", dl["ham"] < tl, dh["ham"] < th),
                         ("test spam", dl["spam|clean"] >= tl, dh["spam|clean"] >= th)):
        both, lo_, ho_, nei, p = R.mcnemar_full(cl, ch)
        rows.append({"part": part, "n": len(cl), "both_correct": both, "LR_only": lo_, "HyRoS_only": ho_, "both_wrong": nei, "p": p})
    save(pd.DataFrame(rows), "R11_mcnemar_by_class.csv"); print(pd.DataFrame(rows).to_string(index=False))

    # ================================================================ R12b subject + word-set check on the strict SpamAssassin spam
    st = pd.read_csv(os.path.join(a.rev, "splits", "SA_strict_ids.csv"))
    nd = pd.read_csv(os.path.join(a.rev2, "R8_exact_near_duplicates.csv"))
    assert (st.id.values == nd.id.values).all()
    sa = R.load_sa_full(a.spamassassin).set_index("id").loc[st.id]

    def sa_subject(i):
        rec = json.load(open(os.path.join(a.spamassassin, i), encoding="utf-8"))
        m = email.message_from_string(rec["text"], policy=policy.compat32)
        return str(m.get("Subject", "") or "")

    def skey(s):
        return " ".join(t for t in TOK.findall(str(s).lower()) if t not in SUBJ_STOP)

    def wset(t):
        return {w for w in TOK.findall(t.lower()) if len(w) >= 3 and not w.isdigit()}

    rel = emap.set_index("eid").release_row
    ens = df[df.y == 1]
    en_sk = [skey(raw.Subject.fillna("").values[rel[e]]) for e in ens.index]
    en_ws = [wset(t) for t in ens.text]
    bysk = {k for k in en_sk if len(k.split()) >= 2}
    inv = defaultdict(set)
    for j, ws in enumerate(en_ws):
        for w in ws:
            inv[w].add(j)
    rows = []
    for i, yv in zip(st.id, nd.y):
        if yv != 1:
            continue
        ws = wset(sa.loc[i, "text"]); cnt = defaultdict(int)
        for w in ws:
            if len(inv[w]) < 3000:
                for j in inv[w]:
                    cnt[j] += 1
        bJ = max((c_ / (len(ws) + len(en_ws[j]) - c_) for j, c_ in cnt.items()), default=0.0)
        k = skey(sa_subject(i))
        rows.append({"id": i, "subject_match": len(k.split()) >= 2 and k in bysk, "wordset_jaccard": bJ})
    ov = pd.DataFrame(rows).merge(nd[nd.y == 1][["id", "maxJ_exact", "containment80_exact"]], on="id")
    ov["residual_shingle"] = (ov.maxJ_exact >= 0.3) | ov.containment80_exact
    ov["subject_and_words"] = ov.subject_match & (ov.wordset_jaccard >= 0.3)
    ov["words_only"] = ov.wordset_jaccard >= 0.5
    ov["any_suspect"] = ov.residual_shingle | ov.subject_and_words | ov.words_only
    save(ov, "R12_overlap_extended_flags.csv")

    yv = nd.y.values; hamI, spamI = np.where(yv == 0)[0], np.where(yv == 1)[0]
    E3 = {n: np.load(os.path.join(SC, "e3_" + fname(n) + ".npz"))["s"] for n in BASE + HY}
    sets = {"all strict spam": np.zeros(len(spamI), bool), "without shingle residual": ov.residual_shingle.values,
            "without any suspected copy": ov.any_suspect.values}
    rows = []
    for lab, ex in sets.items():
        r_ = {}
        for n in BASE + HY:
            s = E3[n]; t = thr(s[hamI], 0.01); r_[n] = 100 * (s[spamI][~ex] >= t).mean()
        r_["HyRoS"] = np.mean([r_[h] for h in HY])
        rows.append({"spam_set": lab, "n_spam": int((~ex).sum()), **{m: r_[m] for m in BASE + ["HyRoS"]},
                     "HyRoS - LR": r_["HyRoS"] - r_["LR (word)"], "A1 - LR": r_["LR (word+char)"] - r_["LR (word)"]})
    keep_ = spamI[~ov.any_suspect.values]; bd = []
    rng = np.random.default_rng(2029)
    for _ in range(B_CLUSTER):
        ih = rng.choice(hamI, len(hamI)); isp = rng.choice(keep_, len(keep_)); s = rng.integers(0, 5)
        r1 = lambda z: (z[isp] >= thr(z[ih], 0.01)).mean()
        bd.append(100 * (r1(E3[f"HyRoS s{s}"]) - r1(E3["LR (word)"])))
    rows[-1]["HyRoS - LR CI_lo"], rows[-1]["HyRoS - LR CI_hi"] = np.percentile(bd, [2.5, 97.5])
    summ = pd.DataFrame(rows); save(summ, "R12_overlap_extended_recall.csv")
    print({k: int(ov[k].sum()) for k in ["residual_shingle", "subject_and_words", "words_only", "any_suspect"]})
    print(summ.round(2).to_string(index=False)); print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
