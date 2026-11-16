#!/usr/bin/env python3
"""Builds the main tables (tables.json) and the supplementary tables (supp_tables.json) of the
revised manuscript from the result files in results/, results_rev/ and results_rev2/.
Every number is read from a CSV file. Supplementary tables are keyed ({S:key}); the manuscript
builder numbers them in order of first citation.

Usage: TABLES_OUT=<folder> python make_tables_rev2.py
"""
import re, json, os
import numpy as np, pandas as pd

R0, R1, R2, R3 = "results", "results_rev", "results_rev2", "results_rev3"
OUT = os.environ.get("TABLES_OUT", os.path.join("paper", "rev")); os.makedirs(OUT, exist_ok=True)
rd = lambda d, f: pd.read_csv(os.path.join(d, f))
SIX = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+norm)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]
HY = [f"HyRoS s{s}" for s in range(5)]
LABEL = {"NB (word)": "NB (word)", "SVM (word)": "SVM (word)", "LR (word)": "LR (word) [@hadi2026]",
         "LR (word+norm)": "LR+norm", "LR (word+char)": "A1: LR (word+char)", "LR (word+char+norm)": "A2: A1 + normalisation",
         "HyRoS": "HyRoS", "HyRoS+GW": "HyRoS+GW", "Monotone top-3": "Monotone top-3"}
SHORT = {"NB (word)": "NB", "SVM (word)": "SVM", "LR (word)": "LR", "LR (word+norm)": "LR+norm", "LR (word+char)": "A1",
         "LR (word+char+norm)": "A2", "HyRoS": "HyRoS", "HyRoS+GW": "HyRoS+GW", "Monotone top-3": "Mono."}
ATT = [("clean", "No attack"), ("obfuscation 10%", "Obfuscation 10%"), ("obfuscation 20%", "Obfuscation 20%"),
       ("obfuscation 30%", "Obfuscation 30%"), ("obfuscation 50%", "Obfuscation 50%"),
       ("good-word k=10", "Good words appended, m = 10"), ("good-word k=25", "Good words appended, m = 25"),
       ("good-word k=50", "Good words appended, m = 50"), ("good-word k=25 interleaved", "Good words interleaved, m = 25"),
       ("combined (30% + k=25)", "Obfuscation 30% + appended m = 25"),
       ("held-out edits 50%", "Held-out edits 50%"), ("held-out edits 100%", "Held-out edits 100%"),
       ("held-out edits 30% + good-word k=25", "Held-out edits 30% + appended m = 25")]
AL = dict(ATT)
f2 = lambda x: f"{x:.2f}"
f1 = lambda x: f"{x:.1f}"
mn = lambda s: __import__("re").sub(r"-(0\.0+)(?![0-9])", r"\1", s).replace("-", "−")


def ci(v, lo, hi, d=1, scale=1.0):
    fmt = (lambda x: f"{scale * x:.{d}f}")
    return mn(f"{fmt(v)}\n({fmt(lo)}–{fmt(hi)})")


def sgn(x):
    return ("+" if x >= 0.05 else ("−" if x <= -0.05 else "")) + f"{abs(x):.1f}"


T, S = {}, {}

# ------------------------------------------------------------------ Table 1: E1 + E2
e1 = rd(R0, "E1_random_split_summary.csv").set_index("model")
r7 = rd(R2, "R7_e2_thresholds.csv")
rows = []
for m in SIX + ["HyRoS+GW", "Monotone top-3"]:
    def cell(mode, col, lo, hi):
        if m == "HyRoS":
            g = r7[(r7.model.isin(HY)) & (r7.threshold == mode)]
            return f"{g[col].mean():.2f}\n(seeds {g[col].min():.2f}–{g[col].max():.2f})"
        g = r7[(r7.model == m) & (r7.threshold == mode)]
        if not len(g):
            return "–"
        x = g.iloc[0]
        return f"{x[col]:.2f}" if lo is None else f"{x[col]:.2f}\n({x[lo]:.2f}–{x[hi]:.2f})"
    rows.append([LABEL[m],
                 f"{100 * e1.loc[m, 'Accuracy_mean']:.2f} ± {100 * e1.loc[m, 'Accuracy_std']:.2f}" if m in e1.index else "–",
                 f"{100 * e1.loc[m, 'FPR_mean']:.2f} ± {100 * e1.loc[m, 'FPR_std']:.2f}" if m in e1.index else "–",
                 cell("default", "FPR", "FPR_lo", "FPR_hi"), cell("default", "Recall", None, None),
                 cell("calibrated", "FPR", "FPR_lo", "FPR_hi"), cell("calibrated", "Recall", None, None),
                 cell("calibrated", "FPR_without_hourahead", None, None)])
T["t1"] = {"title": "Random-split replication (E1) and class-wise chronological stress test (E2) on Enron-Spam, %",
           "header": ["Model", "E1 accuracy", "E1 FPR", "E2 FPR, default θ", "E2 recall, default θ",
                      "E2 FPR, calibrated θ", "E2 recall, calibrated θ", "E2 FPR, calibrated θ, without HourAhead"],
           "rows": rows, "widths": [0.19, 0.12, 0.11, 0.12, 0.1, 0.13, 0.1, 0.13], "boldRows": [],
           "note": "E1: mean ± standard deviation over five stratified 80/20 random splits. E2: oldest 80% of each class for training, "
                   "newest 20% for testing (3,118 ham, 2,910 spam). Default θ: spam probability ≥ 0.5 for NB and the LR models, signed "
                   "margin ≥ 0 for the SVM; the monotone scorer has no default. Calibrated θ: Eq. (4), 1% target on the newest 10% of "
                   "training ham. Parentheses: exact 95% intervals; for HyRoS, the mean over five training seeds and the range over them. "
                   "Last column: FPR on the 2,344 test ham that are not automated HourAhead notices. Recall intervals are in "
                   "Supplementary Table {S:e2detail}."}

# ------------------------------------------------------------------ Table 2: E2c
r1 = rd(R1, "R1_contemporaneous_all.csv"); x1 = rd(R2, "R9_e2c_exact_intervals.csv")
g = r1.groupby(["split", "model", "threshold"])[["FPR", "Recall"]].agg(["mean", "std"])
rows = []
for m in ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]:
    rnd = g.loc[("random", m, "default")]
    def ex(split, thr, kind):
        x = x1[(x1.split == split) & (x1.model == m) & (x1.threshold == thr)].iloc[0]
        if kind == "FPR":
            return f"{100 * x.FP / x.n_ham:.2f}\n({x.FPR_lo:.2f}–{x.FPR_hi:.2f})"
        return f"{100 * x.TP / x.n_spam:.1f}\n({x.Recall_lo:.1f}–{x.Recall_hi:.1f})"
    rows.append([LABEL[m] if m != "HyRoS" else "HyRoS",
                 f"{100 * rnd[('FPR', 'mean')]:.2f} ± {100 * rnd[('FPR', 'std')]:.2f}",
                 f"{100 * rnd[('Recall', 'mean')]:.1f} ± {100 * rnd[('Recall', 'std')]:.1f}",
                 ex("joint chronological", "default", "FPR"), ex("joint chronological", "default", "Recall"),
                 ex("class-wise chronological", "default", "FPR"), ex("class-wise chronological", "default", "Recall"),
                 ex("joint chronological", "calibrated", "FPR")])
T["t2"] = {"title": "Contemporaneous control on SpamAssassin mail received from July to September 2002 (E2c), %",
           "header": ["Model", "Random FPR", "Random recall", "Joint chron. FPR", "Joint chron. recall",
                      "Class-wise chron. FPR", "Class-wise chron. recall", "Joint chron. FPR, calibrated θ"],
           "rows": rows, "widths": [0.18, 0.11, 0.11, 0.12, 0.12, 0.12, 0.12, 0.12], "boldRows": [],
           "note": "Models trained and tested within the window (4,309 messages); default thresholds except in the last column. HyRoS: "
                   "seed 0 in the chronological columns; in the random columns the augmentation seed equals the split seed. Random: "
                   "mean ± standard deviation over five stratified 80/20 splits (630 ham and 232 spam per test set). Joint chronological: "
                   "all messages ordered by delivery time; test set of 686 ham and 176 spam received from 12 to 30 September 2002. "
                   "Class-wise chronological: each class ordered separately, as in E2 (631 ham, 232 spam). Parentheses: exact 95% "
                   "intervals. Results by folder and sender are in Supplementary Table {S:e2c}."}

# ------------------------------------------------------------------ Table 3: E3 strict cross-corpus
r8 = rd(R2, "R8_e3_default.csv").set_index("model")
rs = rd(R2, "R8_recalibration_summary.csv"); m8 = rd(R2, "R8_e3_matched_fpr1_joint.csv").set_index("model")
rows = []
for m in SIX:
    if m == "HyRoS":
        x = r8.loc[HY].mean()
    else:
        x = r8.loc[m]
    r = rs[(rs.model == m) & (rs.n_ham == 100)].iloc[0]; q = m8.loc[m]
    rows.append([LABEL[m], f"{x.FPR:.2f}", f"{x.Recall:.2f}", f"{x.Precision:.2f}", f"{x.AUC:.3f}",
                 ci(r.FPR_mean, r.FPR_lo, r.FPR_hi, 2, 100), ci(r.Recall_mean, r.Recall_lo, r.Recall_hi, 1, 100),
                 ci(r.Precision_mean, r.Precision_lo, r.Precision_hi, 1, 100), ci(q.Recall, q.CI_lo, q.CI_hi, 1)])
T["t3"] = {"title": "Cross-corpus transfer from Enron-Spam to the strict SpamAssassin test set (E3), %",
           "header": ["Model", "FPR, default θ", "Recall, default θ", "Precision, default θ", "AUC",
                      "FPR, n = 100", "Recall, n = 100", "Precision, n = 100", "Recall at 1% FPR"],
           "rows": rows, "widths": [0.17, 0.09, 0.09, 0.1, 0.07, 0.12, 0.12, 0.12, 0.12], "boldRows": [],
           "note": "Test set: 4,110 ham (250 hard ham) and 702 spam. Default θ as in Table 1. n = 100: few-shot recalibration with 100 "
                   "SpamAssassin ham messages; mean and 2.5th–97.5th percentiles over 200 shared draws (1,000 for HyRoS, 200 per seed); "
                   "NB flagged nothing in some draws, which are left out of its precision. Recall at 1% FPR: threshold placed on all "
                   "SpamAssassin ham (a diagnostic), with a joint-bootstrap 95% interval. HyRoS: mean over five training seeds."}

# ------------------------------------------------------------------ Table 4: matched 1.5% FPR
r6 = rd(R2, "R6_matched_fpr_joint.csv"); q4 = r6[np.isclose(r6.target_FPR, 0.015)].set_index(["attack", "model"])
rows = [[lab] + [ci(q4.loc[(a, m)].Recall, q4.loc[(a, m)].CI_lo, q4.loc[(a, m)].CI_hi, 1, 100) for m in SIX] for a, lab in ATT]
T["t4"] = {"title": "Spam recall under evasion attacks with every threshold set to flag 1.5% of the clean E2 test ham, % (joint-bootstrap 95% interval)",
           "header": ["Attack"] + [SHORT[m] for m in SIX], "rows": rows,
           "widths": [0.235, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.115],
           "note": "A1: LR (word+char); A2: A1 + normalisation; LR+norm: LR on normalised word features; HyRoS: A2 + obfuscation "
                   "augmentation, mean over five training seeds. m: number of good words added to each spam message. The obfuscation rows "
                   "use the edits that the augmentation and normaliser target; the held-out rows use edits they were not built for. Intervals resample test ham "
                   "and spam, re-place the thresholds and, for HyRoS, draw one of the five trained models. HyRoS+GW, the monotone scorer and "
                   "the 1% rate are in Supplementary Tables {S:att15} and {S:att10}."}

# ------------------------------------------------------------------ Table 5: component effects at 1.5%
ce = rd(R2, "R6_component_effects_joint.csv")
c5 = ce[np.isclose(ce.target_FPR, 0.015)].set_index(["comparison", "attack"])
P5 = ["A1 - LR: character n-grams", "A2 - A1: normalisation after character n-grams", "HyRoS - A2: obfuscation augmentation",
      "HyRoS - LR: all components", "LR+norm - LR: normalisation alone"]


def eff(c, a):
    x = c5.loc[(c, a)]
    return mn(f"{sgn(x.diff_pp)}{'*' if x.p_holm < 0.05 else ''}\n({x.CI_lo_pp:.1f}, {x.CI_hi_pp:.1f})")


rows = [[lab] + [eff(c, a) for c in P5] for a, lab in ATT]
T["t5"] = {"title": "Paired effect of each component on spam recall at a matched 1.5% false-positive rate, percentage points (joint-bootstrap 95% interval)",
           "header": ["Attack", "A1 − LR\n(character n-grams)", "A2 − A1\n(normalisation)", "HyRoS − A2\n(augmentation)",
                      "HyRoS − LR\n(all)", "LR+norm − LR\n(normalisation alone)"],
           "rows": rows, "widths": [0.26, 0.15, 0.15, 0.15, 0.13, 0.16],
           "note": "Positive values favour the first model of each pair. Intervals and p-values come from the joint bootstrap of Table 4 "
                   "(threshold variability and, for HyRoS, the variation between augmentation seeds included); the intervals are not adjusted "
                   "for multiplicity. * Holm-adjusted bootstrap p < 0.05 over the thirteen attack conditions of that comparison. Further "
                   "comparisons and the 1% rate are in Supplementary Table {S:comp}; sensitivity analyses in Supplementary Table {S:sens}."}
json.dump(T, open(os.path.join(OUT, "tables.json"), "w"), ensure_ascii=False, indent=1)

# ================================================================== supplementary
e5 = rd(R0, "E5_efficiency.csv").set_index("model")
S["e1"] = {"title": "Random-split replication (E1), mean ± standard deviation over five seeds",
           "header": ["Model", "Accuracy, %", "Precision, %", "Recall, %", "F1-score, %", "FPR, %", "MCC", "AUC"],
           "rows": [[LABEL[m].replace(" [@hadi2026]", "")] + [f"{100 * e1.loc[m, k + '_mean']:.2f} ± {100 * e1.loc[m, k + '_std']:.2f}" for k in ("Accuracy", "Precision", "Recall", "F1", "FPR")]
                    + [f"{e1.loc[m, k + '_mean']:.4f} ± {e1.loc[m, k + '_std']:.4f}" for k in ("MCC", "AUC")]
                    for m in ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]],
           "note": "Default thresholds (spam probability 0.5; SVM margin 0). HyRoS: training seed 0."}
d0 = rd(R1, "R0_dedup_rules.csv")
S["dedup"] = {"title": "Messages retained from the single-file Enron-Spam release under different de-duplication rules",
              "header": ["Rule", "Messages"], "rows": [[r.rule, f"{r.messages:,}"] for r in d0.itertuples()]
                      + [[r.rule, f"{int(r.kept):,}"] for r in rd(R3, "R12_dedup_replicated_rule.csv").itertuples()],
              "note": "The first row is the raw count; the next six rows are de-duplication rules applied in this study (the rule used here "
                      "keeps 30,137). The last two rows apply the rule stated by Hadi and Adil, who report 30,089 messages."}
fl = rd(R1, "R2_overlap_flags.csv")
rows = []
for col, lab in [("prefix25", "First 25 words open an Enron-Spam message"), ("exact", "Exact match of the full token sequence"),
                 ("jaccard50", "Jaccard similarity of 5-word shingles ≥ 0.5 (MinHash LSH, verified)"),
                 ("containment80", "Containment ≥ 0.8 in either direction (≥ 20 shingles)"), ("any", "Any of the above (removed)")]:
    v = fl[col].astype(bool)
    rows.append([lab, int((v & (fl.y == 0)).sum()), int((v & (fl.y == 1)).sum()), int((v & (fl.group == "spam-1")).sum()),
                 int((v & (fl.group == "spam-2")).sum()), int((v & ~fl.prefix25.astype(bool)).sum())])
nd = rd(R2, "R8_exact_near_duplicates.csv")
for lab, mask in [("Exact re-check of the strict set: Jaccard ≥ 0.5 or containment ≥ 0.8", (nd.maxJ_exact >= 0.5) | nd.containment80_exact),
                  ("Exact re-check of the strict set: Jaccard ≥ 0.3", nd.maxJ_exact >= 0.3)]:
    rows.append([lab, int((mask & (nd.y == 0)).sum()), int((mask & (nd.y == 1)).sum()), "–", "–", "–"])
ro = rd(R2, "R8_residual_overlap.csv")
for r in ro[ro.check.str.startswith("recall")].itertuples():
    rows.append([r.check.replace("recall at 1% FPR, ", "Recall at 1% FPR, ").replace(": all strict spam / without residual overlap", "") +
                 ": all strict spam → without Jaccard ≥ 0.3 or containment", "–", f"{r.ham:.2f} → {r.spam:.2f}", "–", "–", "–"])
_of = rd(R3, "R12_overlap_extended_flags.csv"); _ox = rd(R3, "R12_overlap_extended_recall.csv").set_index("spam_set")
rows.append(["Later check of the strict spam: same subject line and word-set Jaccard ≥ 0.3, or word-set Jaccard ≥ 0.5, not flagged above",
             "–", int((_of.any_suspect & ~_of.residual_shingle).sum()), "–", "–", "–"])
for m_, lab_ in (("LR (word)", "LR (word)"), ("LR (word+char)", "LR (word+char)"), ("HyRoS", "HyRoS")):
    rows.append([f"Recall at 1% FPR, {lab_}: all strict spam → without any suspected copy ({int(_ox.loc['without any suspected copy'].n_spam)} spam)",
                 "–", f"{_ox.loc['all strict spam', m_]:.2f} → {_ox.loc['without any suspected copy', m_]:.2f}", "–", "–", "–"])
tot = fl.groupby("group").size()
S["overlap"] = {"title": "SpamAssassin messages flagged as copies of Enron-Spam messages, and an exact re-check of the strict test set",
                "header": ["Check", "Ham", "Spam", "spam-1 folder", "spam-2 folder", "Not found by the prefix check"], "rows": rows,
                "note": f"De-duplicated SpamAssassin: {len(fl):,} messages ({int((fl.y == 0).sum()):,} ham, {int((fl.y == 1).sum()):,} spam; "
                        f"spam-1 {tot.get('spam-1', 0)}, spam-2 {tot.get('spam-2', 0)}). Strict set after removal: 4,812 messages. The exact "
                        "re-check compares every strict-set message with every Enron-Spam message without LSH."}
S["betarule"] = {"title": "False-positive rate of the few-shot threshold when calibration ham and later ham come from the same distribution",
                 "header": ["n", "Rank r of θ from the top", "Expected FPR, %", "2.5th percentile, %", "97.5th percentile, %", "P(FPR > 2%)"],
                 "rows": [[int(r.n_ham), int(r.rank_from_top), f"{r.expected_FPR_pct:.2f}", f"{r['q2.5_pct']:.2f}", f"{r['q97.5_pct']:.2f}",
                           f"{r.P_FPR_above_2pct:.3f}"] for _, r in rd(R2, "R10_recalibration_rule.csv").iterrows()],
                 "note": "Beta(r, n − r + 1) distribution of the FPR obtained when θ is the r-th largest of n calibration scores (continuous scores)."}
# E2 detail (recall intervals, seeds)
rows = []
for _, x in r7.iterrows():
    rows.append([SHORT.get(x.model, x.model.replace("HyRoS s", "HyRoS seed ")), x.threshold, f"{x.FP}/{x.n_ham}", f"{x.FPR:.2f} ({x.FPR_lo:.2f}–{x.FPR_hi:.2f})",
                 f"{x.TP}/{x.n_spam}", f"{x.Recall:.2f} ({x.Recall_lo:.2f}–{x.Recall_hi:.2f})", f"{x.FP_hourahead}", f"{x.FPR_without_hourahead:.2f}"])
S["e2detail"] = {"title": "Class-wise chronological stress test (E2): counts, exact 95% intervals and the automated HourAhead notices",
                 "header": ["Model", "Threshold", "False positives / ham", "FPR, %", "Detected / spam", "Recall, %", "FP that are HourAhead", "FPR without HourAhead, %"],
                 "rows": rows, "note": "The 774 HourAhead notices make up 24.8% of the test ham and do not occur in the training or calibration ham."}
bx = rd(R2, "R7_e2_fp_by_mailbox.csv")
rows = []
for (m, th), g in bx.groupby(["model", "threshold"], sort=False):
    if m in ("LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS s0", "LR (word+norm)"):
        rows.append([SHORT.get(m, "HyRoS seed 0"), th] + [f"{int(r.FP)}/{int(r.n_ham)}" for r in g.itertuples()])
S["mailbox"] = {"title": "False positives on the E2 test ham by Enron mailbox",
                "header": ["Model", "Threshold"] + list(bx.mailbox.unique()), "rows": rows,
                "note": "Mailboxes are the six Enron-Spam blocks (Enron1–Enron6) of the single-file release."}
mc = rd(R2, "R7_mcnemar_hourahead.csv")
rows = [[f"seed {r.seed}", r.ham_subset, r.n_paired, r.both_correct, r.LR_only, r.HyRoS_only, r.both_wrong,
         "< 0.001" if r.p < 0.001 else f"{r.p:.3f}"] for r in mc.itertuples()]
r5 = rd(R1, "R5_lr_vs_hyros_calibrated.csv")
for r in r5.itertuples():
    rows.append([f"seed 0, {AL[r.attack]}", "all test ham", r.n_paired, r.both_correct, r.LR_only, r.HyRoS_only, r.both_wrong,
                 ("< 0.001" if r.p < 0.001 else f"{r.p:.3f}") + f" (Holm {('< 0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}')})"])
S["mcnemar"] = {"title": "LR (word) and HyRoS at their recency-calibrated thresholds on the E2 test set: exact McNemar tests",
                "header": ["HyRoS model / attack", "Ham included", "Paired messages", "Both correct", "LR only", "HyRoS only", "Both wrong", "p"],
                "rows": rows, "note": "Paired observations: all test messages, each classified correctly or not; the test weighs a false "
                                       "positive and a missed spam equally. Holm adjustment over the ten attack conditions of the original "
                                       "attack set."}
S["e2c"] = {"title": "Contemporaneous control (E2c): joint chronological split by folder and sender, and with the messages that lack a Received header",
            "header": ["Window", "Model", "Threshold", "FPR, % (exact CI)", "FP: easy-ham-1 / hard-ham", "FP from the RSS sender", "FPR without RSS sender, %", "Recall, %"],
            "rows": [[("published" if x.window == "orig" else "with Date-header fallback"), SHORT.get(x.model, x.model), x.threshold,
                      f"{x.FPR:.2f} ({x.FPR_lo:.2f}–{x.FPR_hi:.2f})", f"{int(x['FP_easy-ham-1'])}/{int(x['n_easy-ham-1'])} ; {int(x['FP_hard-ham-1'])}/{int(x['n_hard-ham-1'])}",
                      f"{int(x.FP_rss)}/{int(x.n_rss)}", f"{x.FPR_without_rss:.2f}", f"{x.Recall:.1f}"] for _, x in rd(R2, "R9_e2c_joint_chronological.csv").iterrows()],
            "note": "The RSS sender (one gateway address relaying short news items) first appears in the test weeks. No easy-ham-2 or spam-2 message falls in the "
                    "calibration or test weeks, while they make up most of the training weeks. HyRoS: training seed 0."}
rows = []
for _, r in rs.iterrows():
    if r.model == "HyRoS - LR (paired draws)":
        rows.append(["HyRoS − LR (paired)", int(r.n_ham), mn(f"{100 * r.FPR_mean:+.2f}"), "–", mn(f"{100 * r.Recall_mean:+.1f} ({100 * r.Recall_lo:.1f} to {100 * r.Recall_hi:.1f})"),
                     f"ahead in {100 * r.share_HyRoS_ahead:.1f}% of draws", "–"])
    else:
        rows.append([SHORT[r.model], int(r.n_ham), f"{100 * r.FPR_mean:.2f} ({100 * r.FPR_lo:.2f}–{100 * r.FPR_hi:.2f})", f"{100 * r.FPR_share_above_2pct:.1f}",
                     f"{100 * r.Recall_mean:.1f} ({100 * r.Recall_lo:.1f}–{100 * r.Recall_hi:.1f})",
                     f"{100 * r.Precision_mean:.1f} ({100 * r.Precision_lo:.1f}–{100 * r.Precision_hi:.1f})", int(r.draws_flagging_nothing)])
S["recal"] = {"title": "Few-shot recalibration on the strict SpamAssassin set: mean and 2.5th–97.5th percentiles over the shared draws",
              "header": ["Model", "n", "FPR, %", "Draws with FPR > 2%, %", "Recall, %", "Precision, %", "Draws flagging nothing"], "rows": rows,
              "note": "200 draws per n, shared by all models (HyRoS: 200 draws for each of five seeds). The percentiles describe variability "
                      "within one external corpus. Precision is left out for draws that flag nothing."}
S["e3match"] = {"title": "Cross-corpus spam recall at a matched 1% FPR on all SpamAssassin ham, and paired differences (joint bootstrap)",
                "header": ["Model or difference", "Recall or difference, %", "95% interval", "Bootstrap p"],
                "rows": [[LABEL.get(k, k).replace(" [@hadi2026]", ""), f"{r.Recall:.2f}", mn(f"{r.CI_lo:.2f} to {r.CI_hi:.2f}"),
                          ("–" if pd.isna(r.get("p_boot", np.nan)) else f"{r.p_boot:.4f}")] for k, r in m8.iterrows()],
                "note": "Diagnostic analysis: the threshold uses all SpamAssassin ham. HyRoS: mean over five seeds; intervals include seed variability."}
o3 = rd(R0, "E3_cross_corpus.csv").set_index("model"); o3r = rd(R0, "E3_fewshot_recalibration_summary.csv")
S["prefix"] = {"title": "Cross-corpus results on the prefix-only SpamAssassin set (5,455 messages, 1,345 spam)",
               "header": ["Model", "FPR, default θ, %", "Recall, default θ, %", "AUC", "FPR, n = 100 (mean of 10 draws), %", "Recall, n = 100, %"],
               "rows": [[LABEL[m].replace(" [@hadi2026]", ""), f2(100 * o3.loc[m].FPR), f2(100 * o3.loc[m].Recall), f"{o3.loc[m].AUC:.3f}",
                         f2(100 * o3r[(o3r.model == m) & (o3r.n_ham == 100)].iloc[0].FPR_mean), f2(100 * o3r[(o3r.model == m) & (o3r.n_ham == 100)].iloc[0].Recall_mean)]
                        for m in ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS"]],
               "note": "Removing only the messages flagged by the 25-word prefix check leaves 643 near-duplicates of Enron-Spam spam; these "
                       "were easy spam for every model, so the strict set of Table 3 lowers recall and AUC but leaves the FPR unchanged. HyRoS: seed 0."}
ALL9 = SIX + ["HyRoS+GW", "Monotone top-3"]
for key, tgt in (("att15", 0.015), ("att10", 0.01)):
    q = r6[np.isclose(r6.target_FPR, tgt)].set_index(["attack", "model"])
    S[key] = {"title": f"Spam recall under attack at a matched {100 * tgt:g}% FPR on the clean E2 test ham, all models, % (joint-bootstrap 95% interval)",
              "header": ["Attack"] + [SHORT[m] for m in ALL9],
              "rows": [[lab] + [mn(f"{100 * q.loc[(a, m)].Recall:.1f} ({100 * q.loc[(a, m)].CI_lo:.1f}–{100 * q.loc[(a, m)].CI_hi:.1f})") for m in ALL9] for a, lab in ATT],
              "note": "HyRoS: mean over five seeds. HyRoS+GW: seed 0. Mono.: monotone top-3 scorer."}
S["comp"] = {"title": "Paired component effects at matched FPR (joint bootstrap): differences in spam recall, percentage points",
             "header": ["Target FPR", "Comparison", "Attack", "Difference", "95% interval", "Bootstrap p", "Holm p"],
             "rows": [[f"{100 * r.target_FPR:g}%", r.comparison, AL[r.attack], mn(f"{r.diff_pp:+.2f}"), mn(f"{r.CI_lo_pp:.2f} to {r.CI_hi_pp:.2f}"),
                       f"{r.p_boot:.4f}", f"{r.p_holm:.4f}"] for r in ce.itertuples()],
             "note": "Holm adjustment over the thirteen attack conditions within each comparison and target."}
sd = rd(R2, "R6_hyros_seeds.csv")
rows = []
for (t, a), g in sd.groupby(["target_FPR", "attack"], sort=False):
    rows.append([f"{100 * t:g}%", AL[a]] + [f"{100 * v:.1f}" for v in g.Recall_HyRoS] + [f"{100 * g.Recall_LR.iloc[0]:.1f}"])
S["seeds"] = {"title": "HyRoS spam recall at matched FPR for each of the five training seeds, %",
              "header": ["Target FPR", "Attack", "Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4", "LR (word)"], "rows": rows,
              "note": "LR, A1 and A2 are deterministic; only the augmentation of HyRoS depends on the seed."}
e4 = rd(R0, "E4_attacks_chronological.csv").set_index(["attack", "model"])
OLD8 = ["NB (word)", "SVM (word)", "LR (word)", "LR (word+char)", "LR (word+char+norm)", "HyRoS", "HyRoS+GW", "Monotone top-3"]
S["calatt"] = {"title": "Spam recall under attack at the recency-calibrated thresholds (operating points differ between models), %",
               "header": ["Attack"] + [SHORT[m] for m in OLD8],
               "rows": [[lab] + [f1(100 * e4.loc[(a, m)].SpamRecall) for m in OLD8] for a, lab in ATT[:10]] +
                       [["FPR on clean test ham"] + [f2(100 * e4.loc[("clean", m)].FPR) for m in OLD8]],
               "note": "HyRoS and HyRoS+GW: seed 0. Because the calibrated thresholds give different FPRs, Tables 4 and 5 should be used for comparisons."}
sens = rd(R3, "R11_component_sensitivity.csv")
_an = {"cluster bootstrap, HourAhead as one cluster, matched 1.5%": "cluster", "recall averaged over matched FPR 0.5-3%": "averaged"}
CMP = {"A1 - LR: character n-grams": "A1 − LR", "HyRoS - A2: obfuscation augmentation": "HyRoS − A2", "HyRoS - LR: all components": "HyRoS − LR",
       "LR+norm - LR: normalisation alone": "LR+norm − LR"}
rows = []
for a, lab in ATT:
    row = [lab]
    for cmp_ in CMP:
        for an in _an:
            x = sens[(sens.analysis == an) & (sens.comparison == cmp_) & (sens.attack == a)].iloc[0]
            row.append(mn(f"{sgn(x.diff_pp)}{'*' if x.p_holm < 0.05 else ''}\n({x.CI_lo_pp:.1f}, {x.CI_hi_pp:.1f})"))
    rows.append(row)
_hf = rd(R3, "R11_hourahead_flagged.csv"); _fd = rd(R3, "R11_e2_calibrated_fpr_difference.csv").set_index("ham_subset")
_mc = rd(R3, "R11_mcnemar_by_class.csv").set_index("part")
_hc = _hf[_hf.model.isin(["LR (word+char)", "LR (word+char+norm)"] + [f"HyRoS s{s}" for s in range(5)])]
_h15 = _hc[np.isclose(_hc.target_FPR, 0.015)].hourahead_flagged; _h10 = _hc[np.isclose(_hc.target_FPR, 0.01)].hourahead_flagged
_a, _n = _fd.loc["all test ham"], _fd.loc["without HourAhead"]
_mh, _ms = _mc.loc["test ham without HourAhead"], _mc.loc["test spam"]
S["sens"] = {"title": "Sensitivity of the component effects of Table 5 to the HourAhead notices, percentage points",
             "header": ["Attack"] + [f"{c}, {an}" for c in CMP.values() for an in _an.values()], "rows": rows,
             "note": re.sub(r"-(?=\d)", "−", f"Cluster: joint bootstrap at a matched 1.5% FPR in which the 774 HourAhead notices are resampled as one unit "
                        f"(2,000 replicates). Averaged: recall averaged over thresholds that flag 0.5%, 0.75%, …, 3% of all test ham "
                        f"(1,000 replicates). * Holm-adjusted p < 0.05 over the thirteen attacks. HourAhead notices among the flagged ham: "
                        f"{_h15.min()}–{_h15.max()} of 46 for the character-based models and none for the word-level models at 1.5%; "
                        f"{_h10.min()}–{_h10.max()} of 31 and none at 1%. E2 calibrated FPR, HyRoS − LR: {_a.diff_pp:+.2f} points "
                        f"({_a.CI_lo_pp:.2f} to {_a.CI_hi_pp:.2f}) on all test ham and {_n.diff_pp:+.2f} ({_n.CI_lo_pp:.2f} to {_n.CI_hi_pp:.2f}) "
                        f"without the notices. McNemar counts by class (LR only / HyRoS only, seed 0): ham without HourAhead "
                        f"{int(_mh.LR_only)} / {int(_mh.HyRoS_only)} (p = {_mh.p:.2f}); spam {int(_ms.LR_only)} / {int(_ms.HyRoS_only)} (p = {_ms.p:.2f}).")}
S["cost"] = {"title": "Cost of each model (E5)", "header": ["Model", "Training time (s)", "Latency (ms per message)", "Model size (MB)", "Features"],
             "rows": [[LABEL[m].replace(" [@hadi2026]", ""), f"{e5.loc[m].fit_s:.1f}", f"{e5.loc[m].latency_ms_per_email:.2f}",
                       f"{e5.loc[m].model_size_MB:.1f}", f"{int(e5.loc[m].n_features):,}"] for m in OLD8],
             "note": "CPU only; latency averaged over 2,000 legitimate test messages."}
S = json.loads(json.dumps(S, ensure_ascii=False).replace(" [@hadi2026]", ""))
json.dump(S, open(os.path.join(OUT, "supp_tables.json"), "w"), ensure_ascii=False, indent=1)
print("tables:", list(T), "supplementary:", list(S))
