# HyRoS: code and results

Code, result files and figures for the paper

> H. S. Chasib, "Beyond Near-Perfect Accuracy: Temporal Drift, Cross-Corpus Shift and Evasion
> Robustness of Lightweight Spam Filters" (under review).

The paper re-examines the lightweight TF-IDF spam filters of Hadi & Adil, *Wasit Journal of
Computer and Mathematics Science* 5(3):1–11, 2026 (doi:10.31185/wjcms.562), under temporal drift,
cross-corpus transfer and evasion attacks, and proposes the HyRoS filter.

## 1. Requirements

Python 3.10+ on any CPU (no GPU). Tested with scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy, matplotlib.

```bash
pip install -r requirements.txt
```

## 2. Data (both public, not redistributed here)

| Corpus | Use | Source |
|---|---|---|
| Enron-Spam (Metsis, Androutsopoulos & Paliouras, 2006), single CSV | training / testing | https://github.com/MWiechmann/enron_spam_data |
| SpamAssassin public corpus | external test set only | npm package `@stdlib/datasets-spam-assassin` |

```bash
mkdir -p data && cd data
curl -L -o enron_spam_data.zip https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip
unzip enron_spam_data.zip
npm pack @stdlib/datasets-spam-assassin
mkdir -p sa && tar -xzf stdlib-datasets-spam-assassin-*.tgz -C sa
cd ..
```

Part of the spam in Enron-Spam originally came from the SpamAssassin collection. The code removes
every SpamAssassin message whose first 25 words also open an Enron-Spam message (464 spam, 0 ham)
before the cross-corpus test.

## 3. Reproduce everything

```bash
bash run_all.sh            # about 1.5 h on one laptop CPU core
```

which runs

```bash
python hyros_experiments.py --enron data/enron_spam_data.csv --spamassassin data/sa/package/data --out results --seeds 5
python e7_matched_fpr.py   --enron data/enron_spam_data.csv --out results
python make_figures.py       # the four figures of the paper
```

All randomness is seeded; with the same library versions the files in `results/` are regenerated exactly.
A quick smoke test on a subsample: add `--sample 1500 --seeds 1 --out /tmp/smoke`.

## 4. Experiments and result files

| File | Experiment |
|---|---|
| `E0_year_by_class.csv` | messages per year and class (the class–era confound) |
| `grid_search.csv` | 3-fold grid search of the LR constant C |
| `E1_random_split_*.csv` | stratified 80/20 random split, 5 seeds (replication) |
| `E2_chronological.csv` | class-wise chronological split; default and recency-calibrated thresholds |
| `E3_cross_corpus.csv` | train on Enron-Spam, test on SpamAssassin (default threshold) |
| `E3_fewshot_recalibration_*.csv` | threshold re-estimated from n = 25…500 SpamAssassin ham, 10 draws each |
| `E4_attacks_chronological.csv` | evasion attacks, spam recall with 95% bootstrap CIs, calibrated thresholds |
| `E4_attacks_random_split.csv` | the same attacks on the random split |
| `E5_efficiency.csv` | training time, latency per e-mail, model size, number of features |
| `E6_mcnemar.csv` | exact McNemar tests, LR (word) vs HyRoS |
| `E7_matched_fpr*.csv` | sensitivity analysis: all models at the same 1.5% FPR on test ham |
| `attacker_good_words.txt` | the attacker's 200 ham-indicative words |

## 5. Models

| Name | Features | Extra |
|---|---|---|
| NB / SVM / LR (word) | word 1–2-gram TF-IDF | baselines of Hadi & Adil (2026) |
| A1: LR (word+char) | + char_wb 2–5-gram TF-IDF | ablation |
| A2: LR (word+char+norm) | + anti-obfuscation normalisation | ablation |
| **HyRoS** | as A2 | + obfuscation augmentation of training spam; recency-calibrated threshold |
| HyRoS+GW | as HyRoS | + good-word augmentation (negative result) |
| Monotone top-3 | binary word presence, non-negative weights | score cannot decrease under word insertion |

Normalisation: NFKC; Cyrillic/Greek look-alikes → Latin; letters separated by `. - _ * ~ +` or
spaces are joined; leetspeak is decoded only inside tokens that mix letters and symbols.

Recency-calibrated threshold: the newest 10% of each class in the training period is held out and
the threshold is set to the 99th percentile of the held-out ham scores (target FPR 1%).

## 6. Licence

Code: MIT. The corpora keep their original terms.
