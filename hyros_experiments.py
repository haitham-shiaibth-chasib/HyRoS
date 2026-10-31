#!/usr/bin/env python3
"""
HyRoS: Hybrid, Robust, lightweight Spam filter.

Reproducible experiments for the paper
  "Beyond 99% Accuracy: Temporal Drift, Cross-Corpus Shift and Evasion Robustness
   of Lightweight TF-IDF Spam Filters"

It (1) reproduces the word-TF-IDF baselines of Hadi & Adil (WJCMS 5(3), 2026,
doi:10.31185/wjcms.562) on a random stratified split, then evaluates the same
models and the proposed HyRoS model under
  E2  a class-wise chronological split (older e-mails -> train, newer -> test),
      with a recency-calibrated decision threshold,
  E3  cross-corpus transfer (train Enron-Spam -> test SpamAssassin) and
      few-shot threshold recalibration on the new mailbox,
  E4  obfuscation and good-word evasion attacks,
  E5  efficiency (training time, latency, model size),
  E6  significance tests (exact McNemar, bootstrap CIs).

Usage:
  python hyros_experiments.py --enron enron_spam_data.csv \
                              --spamassassin path/to/spam-assassin/data \
                              --out results [--seeds 5]

Data:
  Enron-Spam (Metsis et al., 2006) as a single CSV: github.com/MWiechmann/enron_spam_data
  SpamAssassin public corpus (npm package @stdlib/datasets-spam-assassin, folder data/)
"""
import argparse
import email
import glob
import json
import os
import pickle
import random
import re
import time
import unicodedata
import warnings
from email import policy

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
MAX_CHARS = 3000          # characters of each e-mail seen by the char analyser
GW_LIST_SIZE = 200        # size of the attacker's / defender's "good word" list

# --------------------------------------------------------------------------- #
# 1. Data loading                                                             #
# --------------------------------------------------------------------------- #
def load_enron(path):
    df = pd.read_csv(path)
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")
    df["text"] = (df["Subject"] + " " + df["Message"]).str.strip()
    df = df[df.text.str.len() > 0].copy()
    df["key"] = df.text.str.lower().str.split().str.join(" ")
    df = df.drop_duplicates("key")
    df["y"] = (df["Spam/Ham"] == "spam").astype(int)
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df[["text", "y", "date"]].reset_index(drop=True)


_TAG = re.compile(r"<[^>]+>")


def _email_to_text(raw):
    msg = email.message_from_string(raw, policy=policy.default)
    subj = str(msg.get("Subject", "") or "")
    parts = []
    for part in (msg.walk() if msg.is_multipart() else [msg]):
        ctype = part.get_content_type()
        if ctype not in ("text/plain", "text/html"):
            continue
        try:
            body = part.get_content()
        except Exception:
            payload = part.get_payload(decode=True) or b""
            body = payload.decode("latin-1", "ignore")
        if ctype == "text/html":
            body = _TAG.sub(" ", body)
        parts.append(body)
    return (subj + " " + " ".join(parts)).strip()


def load_spamassassin(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "*", "*.json"))):
        group = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8") as fh:
            rec = json.load(fh)
        try:
            text = _email_to_text(rec["text"])
        except Exception:
            continue
        rows.append({"text": text, "y": int(group.startswith("spam")), "group": group})
    df = pd.DataFrame(rows)
    df = df[df.text.str.len() > 0].copy()
    df["key"] = df.text.str.lower().str.split().str.join(" ")
    df = df.drop_duplicates("key").drop(columns="key")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Text normalisation used by HyRoS (defence)                               #
# --------------------------------------------------------------------------- #
HOMO = str.maketrans({
    # Cyrillic / Greek look-alikes -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h",
    "α": "a", "ο": "o", "ρ": "p", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "τ": "t",
})
LEET = str.maketrans({"@": "a", "4": "a", "3": "e", "1": "i", "!": "i",
                      "0": "o", "$": "s", "5": "s", "7": "t", "|": "l"})
_SEP_PUNCT = re.compile(r"\b(?:[a-z][.\-_*~+]){2,}[a-z]\b")
_SEP_SPACE = re.compile(r"\b(?:[a-z] ){3,}[a-z]\b")
_MIXED = re.compile(r"(?<![\w@$!|])(?=[\w@$!|]*[a-z])(?=[\w@$!|]*[@013457$!|])[\w@$!|]{3,}(?![\w@$!|])")


def normalise(text):
    t = unicodedata.normalize("NFKC", text).translate(HOMO).lower()
    t = _SEP_PUNCT.sub(lambda m: re.sub(r"[.\-_*~+]", "", m.group(0)), t)
    t = _SEP_SPACE.sub(lambda m: m.group(0).replace(" ", ""), t)
    t = _MIXED.sub(lambda m: m.group(0).translate(LEET), t)   # de-leet mixed tokens only
    return t


def char_pre(text):
    return normalise(text)[:MAX_CHARS]


def char_pre_raw(text):
    return text.lower()[:MAX_CHARS]


# --------------------------------------------------------------------------- #
# 3. Attacks (applied to SPAM test messages only)                             #
# --------------------------------------------------------------------------- #
INV_HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
INV_LEET = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$", "t": "7"}


def _obf_word(w, rng):
    op = rng.choice(["swap", "leet", "sep", "homo", "dup"])
    if op == "swap" and len(w) > 3:
        i = rng.randrange(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "leet":
        return "".join(INV_LEET.get(c, c) if rng.random() < .5 else c for c in w)
    if op == "sep":
        return rng.choice([".", "-", "*", " "]).join(w)
    if op == "homo":
        return "".join(INV_HOMO.get(c, c) if rng.random() < .5 else c for c in w)
    i = rng.randrange(len(w))
    return w[:i] + w[i] + w[i:]


def obfuscate(text, rate, rng):
    toks = text.split()
    out = [(_obf_word(t, rng) if (len(t) >= 4 and t.isalpha() and rng.random() < rate) else t)
           for t in toks]
    return " ".join(out)


def good_word(text, words, k, rng):
    return text + " " + " ".join(rng.choice(words) for _ in range(k))


def ham_word_list(texts, y, n=GW_LIST_SIZE):
    """Surrogate model (word unigram MNB) -> most ham-indicative words."""
    v = TfidfVectorizer(lowercase=True, min_df=5, token_pattern=r"(?u)\b[a-z]{3,}\b")
    X = v.fit_transform(texts)
    nb = MultinomialNB(alpha=0.1).fit(X, y)
    score = nb.feature_log_prob_[0] - nb.feature_log_prob_[1]
    vocab = np.array(v.get_feature_names_out())
    return list(vocab[np.argsort(-score)[:n]])


# --------------------------------------------------------------------------- #
# 4. Models                                                                   #
# --------------------------------------------------------------------------- #
def word_vec():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, max_features=200_000)


def char_vec(norm):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=300_000,
                           preprocessor=char_pre if norm else char_pre_raw, lowercase=False)


def word_vec_norm():
    v = word_vec()
    v.set_params(preprocessor=normalise)
    return v


class MonotoneTopK:
    """Non-negative logistic weights on binary word presence; score = sum of the k largest
    weights present in an e-mail. Inserting words can only add present features, so the
    score of an e-mail can never decrease under any word-insertion (good-word) attack."""

    def __init__(self, k=3, C=1.0, maxiter=300):
        self.k, self.C, self.maxiter = k, C, maxiter
        self.vec = CountVectorizer(lowercase=True, min_df=2, max_df=0.95, max_features=200_000,
                                   binary=True, preprocessor=normalise)

    def _A(self, X, fit=False):
        A = (self.vec.fit_transform(X) if fit else self.vec.transform(X)).tocsr().astype(np.float64)
        A.data[:] = 1.0
        return A

    def fit(self, X, y):
        A = self._A(list(X), fit=True)
        s = 2 * np.asarray(y, float) - 1
        n, d = A.shape
        lam = 1.0 / (self.C * n)

        def f(p):
            w, b = p[:d], p[d]
            z = s * (A @ w + b)
            g = -s * expit(-z) / n
            return (np.logaddexp(0, -z).mean() + 0.5 * lam * w @ w,
                    np.concatenate([A.T @ g + lam * w, [g.sum()]]))
        res = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                       bounds=[(0, None)] * d + [(None, None)], options={"maxiter": self.maxiter})
        self.w = res.x[:d]
        return self

    def decision_function(self, X):
        A = self._A(list(X))
        out = np.empty(A.shape[0])
        for i in range(A.shape[0]):
            w = self.w[A.indices[A.indptr[i]:A.indptr[i + 1]]]
            if len(w) > self.k:
                w = np.partition(w, -self.k)[-self.k:]
            out[i] = w.sum()
        return out


def build(name, C=100.0):
    lr = LogisticRegression(C=C, solver="liblinear", max_iter=2000)
    if name == "NB (word)":
        return Pipeline([("v", word_vec()), ("c", MultinomialNB(alpha=0.01))])
    if name == "SVM (word)":
        return Pipeline([("v", word_vec()), ("c", LinearSVC(C=1.0))])
    if name == "LR (word)":                       # best model of Hadi & Adil (2026)
        return Pipeline([("v", word_vec()), ("c", lr)])
    if name == "LR (word+char)":                  # ablation 1: + character n-grams
        return Pipeline([("v", FeatureUnion([("w", word_vec()), ("c", char_vec(False))])), ("c", lr)])
    if name in ("LR (word+char+norm)", "HyRoS", "HyRoS+GW"):   # ablation 2 / proposed / negative ablation
        return Pipeline([("v", FeatureUnion([("w", word_vec_norm()), ("c", char_vec(True))])), ("c", lr)])
    if name == "Monotone top-3":
        return MonotoneTopK(k=3)
    raise ValueError(name)


MODELS = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]


def augment(texts, y, seed, gw_words=None, frac_obf=0.5, frac_gw=0.0):
    """Defender-side augmentation: obfuscated (and optionally good-word) copies of training spam."""
    rng = random.Random(seed)
    texts, y = list(texts), list(y)
    new = []
    for t, l in zip(texts, y):
        if l != 1:
            continue
        if rng.random() < frac_obf:
            new.append(obfuscate(t, rng.uniform(0.1, 0.4), rng))
        if gw_words and rng.random() < frac_gw:
            new.append(good_word(t, gw_words, rng.randint(5, 40), rng))
    return texts + new, y + [1] * len(new)


def defender_words(texts, y, n=GW_LIST_SIZE):
    """Defender's own ham-word list: most negative coefficients of a word-LR."""
    m = build("LR (word)").fit(list(texts), list(y))
    v, c = m.named_steps["v"], m.named_steps["c"]
    names = np.array(v.get_feature_names_out())
    uni = np.array([" " not in n_ and n_.isalpha() and len(n_) >= 3 for n_ in names])
    coef = c.coef_[0].copy()
    coef[~uni] = np.inf
    return list(names[np.argsort(coef)[:n]])


def fit(name, Xtr, ytr, seed, C=100.0):
    t0 = time.perf_counter()
    m = build(name, C)
    Xtr, ytr = list(Xtr), list(ytr)
    if name == "HyRoS":
        Xtr, ytr = augment(Xtr, ytr, seed)
    elif name == "HyRoS+GW":
        Xtr, ytr = augment(Xtr, ytr, seed, defender_words(Xtr, ytr), frac_obf=0.5, frac_gw=0.25)
    m.fit(Xtr, ytr)
    return m, time.perf_counter() - t0


def scores(m, X):
    if hasattr(m, "predict_proba"):
        return m.predict_proba(list(X))[:, 1]
    return m.decision_function(list(X))


def default_thr(m):
    if isinstance(m, MonotoneTopK):
        return None
    return 0.5 if hasattr(m, "predict_proba") else 0.0


def thr_at_fpr(s_ham, fpr=0.01):
    """Recency-calibrated threshold: smallest score above the (1-fpr) quantile of held-out ham."""
    return float(np.quantile(np.asarray(s_ham), 1 - fpr, method="higher")) + 1e-12


# --------------------------------------------------------------------------- #
# 5. Metrics                                                                  #
# --------------------------------------------------------------------------- #
def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if idx.any():
            e += idx.mean() * abs(y[idx].mean() - p[idx].mean())
    return e


def metrics(y, s, thr, proba):
    y = np.asarray(y)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
           "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
           "FPR": fp / max(fp + tn, 1), "MCC": matthews_corrcoef(y, pred)}
    try:
        out["AUC"] = roc_auc_score(y, s)
    except ValueError:
        out["AUC"] = np.nan
    if proba:
        out["ECE"], out["Brier"] = ece(y, s), brier_score_loss(y, s)
    return out, pred


def mcnemar(y, a, b):
    y, a, b = map(np.asarray, (y, a, b))
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return n01, n10, p


def boot_ci(v, B=1000, seed=0):
    """95% bootstrap CI of the mean of a 0/1 vector."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return np.percentile(means, [2.5, 97.5])


# --------------------------------------------------------------------------- #
# 6. Experiments                                                              #
# --------------------------------------------------------------------------- #
def overlap_key(text, n=25):
    """First n alphabetic words (>= 3 letters); None for very short messages."""
    w = re.findall(r"[a-z]{3,}", text.lower())
    return " ".join(w[:n]) if len(w) >= 10 else None


def chrono_split(df, frac=0.8):
    tr, te = [], []
    for _, g in df.groupby("y"):
        g = g.sort_values("date")
        cut = int(len(g) * frac)
        tr.append(g.iloc[:cut]); te.append(g.iloc[cut:])
    return pd.concat(tr), pd.concat(te)


ATTACKS = [("clean", None)] + \
    [(f"obfuscation {int(r * 100)}%", ("obf", r)) for r in (0.1, 0.2, 0.3, 0.5)] + \
    [(f"good-word k={k}", ("gw", k)) for k in (10, 25, 50)] + \
    [("good-word k=25 interleaved", ("gwi", 25)), ("combined (30% + k=25)", ("both", (0.3, 25)))]


def attacked_sets(Xte, yte, gw_attacker, seed):
    """Return {attack name: list of test texts} with only the SPAM messages modified."""
    yte = np.asarray(yte)
    spam_idx = np.where(yte == 1)[0]
    out = {}
    for ai, (name, spec) in enumerate(ATTACKS):
        Xa = list(Xte)
        r = random.Random(seed * 1000 + ai)          # deterministic per attack
        for i in spam_idx:
            if spec is None:
                continue
            kind, val = spec
            if kind == "obf":
                Xa[i] = obfuscate(Xa[i], val, r)
            elif kind == "gw":
                Xa[i] = good_word(Xa[i], gw_attacker, val, r)
            elif kind == "gwi":
                tok = Xa[i].split()
                for _ in range(val):
                    tok.insert(r.randrange(len(tok) + 1), r.choice(gw_attacker))
                Xa[i] = " ".join(tok)
            else:
                Xa[i] = good_word(obfuscate(Xa[i], val[0], r), gw_attacker, val[1], r)
        out[name] = Xa
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enron", required=True)
    ap.add_argument("--spamassassin", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0, help="quick smoke test on a random subsample")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(0); np.random.seed(0)
    log = open(os.path.join(a.out, "run_log.txt"), "w")

    def say(*x):
        s = " ".join(str(i) for i in x)
        print(s, flush=True); log.write(s + "\n"); log.flush()

    def save(df_, name):
        df_.to_csv(os.path.join(a.out, name), index=False)

    pd.set_option("display.width", 250)
    df = load_enron(a.enron)
    if a.sample:
        df = df.sample(a.sample, random_state=0).reset_index(drop=True)
    say(f"Enron-Spam after cleaning/dedup: {len(df)} (ham={int((df.y==0).sum())}, spam={int((df.y==1).sum())})")
    yr = pd.crosstab(df.date.dt.year, df.y).rename(columns={0: "ham", 1: "spam"}).reset_index()
    save(yr, "E0_year_by_class.csv"); say(yr.to_string(index=False))

    # ---------------- C selection (3-fold grid search on the seed-0 training split) ---- #
    tr0, _ = train_test_split(df, test_size=0.2, stratify=df.y, random_state=0)
    grid = []
    for name in ["LR (word)", "LR (word+char+norm)"]:
        for C in (1.0, 10.0, 100.0):
            cv = StratifiedKFold(3, shuffle=True, random_state=0)
            f1 = cross_val_score(build(name, C), list(tr0.text), tr0.y, cv=cv, scoring="f1").mean()
            grid.append({"model": name, "C": C, "cv_F1": f1})
    grid = pd.DataFrame(grid); save(grid, "grid_search.csv"); say(grid.to_string(index=False))
    C_word = float(grid[grid.model == "LR (word)"].sort_values("cv_F1").C.iloc[-1])
    C_hyb = float(grid[grid.model == "LR (word+char+norm)"].sort_values("cv_F1").C.iloc[-1])

    def C_of(n):
        return C_word if n == "LR (word)" else C_hyb

    # ---------------- E1 random stratified split (replication), several seeds ------- #
    e1 = []
    for seed in range(a.seeds):
        tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=seed)
        for name in MODELS:
            m, t = fit(name, tr.text, tr.y.values, seed, C_of(name))
            met, _ = metrics(te.y.values, scores(m, te.text), default_thr(m), hasattr(m, "predict_proba"))
            met.update(model=name, seed=seed, fit_s=t); e1.append(met)
        say(f"E1 seed {seed} done")
    e1 = pd.DataFrame(e1); save(e1, "E1_random_split_all_seeds.csv")
    agg = e1.drop(columns="seed").groupby("model", sort=False).agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    save(agg.reset_index(), "E1_random_split_summary.csv")
    say(agg[["Accuracy_mean", "Accuracy_std", "F1_mean", "FPR_mean", "AUC_mean"]].round(4).to_string())

    # ---------------- E2 class-wise chronological split + recency-calibrated threshold - #
    trc, tec = chrono_split(df)
    fit_p, cal_p = [], []
    for _, g in trc.groupby("y"):                # newest 10% of each class in the training period
        g = g.sort_values("date"); c = int(len(g) * 0.9)
        fit_p.append(g.iloc[:c]); cal_p.append(g.iloc[c:])
    fitc, calc = pd.concat(fit_p), pd.concat(cal_p)
    say(f"E2 chrono: fit {len(fitc)}, calibration {len(calc)}, test {len(tec)}; "
        f"ham fit<= {fitc[fitc.y==0].date.max().date()}, ham test>= {tec[tec.y==0].date.min().date()}; "
        f"spam fit<= {fitc[fitc.y==1].date.max().date()}, spam test>= {tec[tec.y==1].date.min().date()}")
    E2_MODELS = MODELS + ["HyRoS+GW", "Monotone top-3"]
    e2, fitted, thr_rc = [], {}, {}
    yte = tec.y.values
    for name in E2_MODELS:
        m, t = fit(name, fitc.text, fitc.y.values, 0, C_of(name))
        fitted[name] = m
        s_cal = scores(m, calc.text)
        thr_rc[name] = thr_at_fpr(s_cal[calc.y.values == 0], 0.01)
        s = scores(m, tec.text)
        proba = hasattr(m, "predict_proba")
        for mode, thr in (("default", default_thr(m)), ("recency-calibrated", thr_rc[name])):
            if thr is None:
                continue
            met, _ = metrics(yte, s, thr, proba and mode == "default")
            met.update(model=name, threshold=mode, thr_value=thr, fit_s=t); e2.append(met)
    e2 = pd.DataFrame(e2); save(e2, "E2_chronological.csv")
    say(e2[["model", "threshold", "Accuracy", "F1", "Recall", "FPR", "MCC", "AUC"]].round(4).to_string(index=False))

    # ---------------- E3 cross-corpus: Enron-Spam -> SpamAssassin ------------------- #
    if a.spamassassin:
        sa = load_spamassassin(a.spamassassin)
        if a.sample:
            sa = sa.sample(min(a.sample, len(sa)), random_state=0).reset_index(drop=True)
        # Enron-Spam took part of its spam from the SpamAssassin corpus (Metsis et al., 2006):
        # drop SpamAssassin messages whose first 25 words also occur in Enron-Spam.
        enron_keys = {k for k in map(overlap_key, df.text) if k}
        dup = np.array([(overlap_key(t) in enron_keys) for t in sa.text])
        say(f"SpamAssassin messages overlapping Enron-Spam (removed): {int(dup.sum())} "
            f"(ham={int((dup & (sa.y.values == 0)).sum())}, spam={int((dup & (sa.y.values == 1)).sum())})")
        sa = sa[~dup].reset_index(drop=True)
        say(f"SpamAssassin hard ham after removal: {int((sa.group == 'hard-ham-1').sum())}")
        say(f"SpamAssassin after parsing/dedup: {len(sa)} (ham={int((sa.y==0).sum())}, spam={int((sa.y==1).sum())})")
        e3, e3r = [], []
        hard = (sa.group == "hard-ham-1").values
        ham_idx = np.where(sa.y.values == 0)[0]
        for name in MODELS:
            m, _ = fit(name, df.text, df.y.values, 0, C_of(name))
            s = scores(m, sa.text)
            met, pred = metrics(sa.y.values, s, default_thr(m), hasattr(m, "predict_proba"))
            met["FPR_hard_ham"] = pred[hard].mean()
            met.update(model=name); e3.append(met)
            # few-shot threshold recalibration with n legitimate e-mails of the new mailbox
            for n in (25, 50, 100, 200, 500):
                for rep in range(10):
                    rng = np.random.default_rng(1000 * n + rep)
                    cal_i = rng.choice(ham_idx, n, replace=False)
                    mask = np.ones(len(sa), bool); mask[cal_i] = False
                    thr = thr_at_fpr(s[cal_i], 0.01)
                    yy, ss = sa.y.values[mask], s[mask]
                    pr = (ss >= thr).astype(int)
                    e3r.append({"model": name, "n_ham": n, "rep": rep, "Recall": pr[yy == 1].mean(),
                                "FPR": pr[yy == 0].mean(), "FPR_hard_ham": pr[hard[mask]].mean(),
                                "F1": f1_score(yy, pr)})
            say(f"E3 {name} done")
        e3 = pd.DataFrame(e3); save(e3, "E3_cross_corpus.csv")
        say(e3[["model", "Accuracy", "F1", "Recall", "FPR", "FPR_hard_ham", "AUC"]].round(4).to_string(index=False))
        e3r = pd.DataFrame(e3r); save(e3r, "E3_fewshot_recalibration_all.csv")
        e3s = e3r.groupby(["model", "n_ham"], sort=False).agg(["mean", "std"]).drop(columns="rep")
        e3s.columns = [f"{c}_{s}" for c, s in e3s.columns]
        save(e3s.reset_index(), "E3_fewshot_recalibration_summary.csv")
        say(e3s.round(4).to_string())

    # ---------------- E4 evasion attacks at the recency-calibrated threshold -------- #
    gw_att = ham_word_list(list(trc.text), trc.y.values)
    with open(os.path.join(a.out, "attacker_good_words.txt"), "w") as fh:
        fh.write("\n".join(gw_att))
    sets = attacked_sets(tec.text, yte, gw_att, 7)
    spam_idx = np.where(yte == 1)[0]
    rows, preds = [], {}
    for name in E2_MODELS:
        m = fitted[name]
        for att, X in sets.items():
            s = scores(m, X)
            pred = (s >= thr_rc[name]).astype(int)
            preds[(att, name)] = pred
            lo, hi = boot_ci(pred[spam_idx], B=500)
            rows.append({"attack": att, "model": name, "SpamRecall": pred[spam_idx].mean(),
                         "CI_low": lo, "CI_high": hi, "FPR": pred[yte == 0].mean(),
                         "F1": f1_score(yte, pred)})
    e4 = pd.DataFrame(rows); save(e4, "E4_attacks_chronological.csv")
    piv = e4.pivot(index="attack", columns="model", values="SpamRecall").loc[[n for n, _ in ATTACKS], E2_MODELS]
    save(piv.reset_index(), "E4_attacks_recall_pivot.csv")
    say(piv.round(4).to_string())

    # same attacks on the random split (replication setting, default threshold)
    tr, te = train_test_split(df, test_size=0.2, stratify=df.y, random_state=0)
    gw_r = ham_word_list(list(tr.text), tr.y.values)
    sets_r = attacked_sets(te.text, te.y.values, gw_r, 7)
    yr_ = te.y.values; si_r = np.where(yr_ == 1)[0]
    rr = []
    for name in ["LR (word)", "HyRoS"]:
        m, _ = fit(name, tr.text, tr.y.values, 0, C_of(name))
        for att, X in sets_r.items():
            pred = (scores(m, X) >= 0.5).astype(int)
            rr.append({"attack": att, "model": name, "SpamRecall": pred[si_r].mean(), "FPR": pred[yr_ == 0].mean()})
    rr = pd.DataFrame(rr); save(rr, "E4_attacks_random_split.csv")
    say(rr.pivot(index="attack", columns="model", values="SpamRecall").round(4).to_string())

    # ---------------- E6 significance: LR (word) vs HyRoS, both recency-calibrated ---- #
    sig = []
    for att, _ in ATTACKS:
        n01, n10, p = mcnemar(yte, preds[(att, "LR (word)")], preds[(att, "HyRoS")])
        sig.append({"test_set": f"chronological, {att}", "LR(word)_only_correct": n01,
                    "HyRoS_only_correct": n10, "p_value": p})
    sig = pd.DataFrame(sig); save(sig, "E6_mcnemar.csv"); say(sig.to_string(index=False))

    # ---------------- E5 efficiency --------------------------------------- #
    eff = []
    sample = list(tec.text.iloc[:2000])
    for name in E2_MODELS:
        m = fitted[name]
        scores(m, sample[:50])                                  # warm-up
        t0 = time.perf_counter(); scores(m, sample); dt = time.perf_counter() - t0
        if isinstance(m, MonotoneTopK):
            nfeat = len(m.vec.vocabulary_)
        elif isinstance(m.named_steps["v"], FeatureUnion):
            nfeat = sum(len(v.vocabulary_) for _, v in m.named_steps["v"].transformer_list)
        else:
            nfeat = len(m.named_steps["v"].vocabulary_)
        eff.append({"model": name, "fit_s": float(e2[e2.model == name].fit_s.iloc[0]),
                    "latency_ms_per_email": 1000 * dt / len(sample),
                    "model_size_MB": len(pickle.dumps(m)) / 1e6, "n_features": nfeat})
    eff = pd.DataFrame(eff); save(eff, "E5_efficiency.csv"); say(eff.round(3).to_string(index=False))
    say("done")


if __name__ == "__main__":
    main()
