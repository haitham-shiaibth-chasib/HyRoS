# HyRoS: code and results

Code, split indices and result files for the paper

> H. S. Chasib, "Beyond Near-Perfect Accuracy: Chronological, Cross-Corpus and Evasion Stress Tests of
> Lightweight Spam Filters" (revised version, under review).

The paper re-examines the lightweight TF-IDF spam filters of Hadi & Adil, *Wasit Journal of Computer and
Mathematics Science* 5(3):1–11, 2026 (doi:10.31185/wjcms.562), under a class-wise chronological stress test,
a contemporaneous control, transfer to SpamAssassin and evasion attacks at matched false-positive rates, and
isolates by ablation what normalisation, character n-grams and obfuscation augmentation contribute (HyRoS).

## Versions

| Release | Content |
|---|---|
| v2.0.0 | Revised version: contemporaneous control (E2c), near-duplicate audit, strict cross-corpus test and recalibration, matched-FPR attacks with joint bootstrap intervals, ablation in both orders, sensitivity analyses (R0–R13) |
| v1.0.0 | Code and results of the first submission (`hyros_experiments.py`, `e7_matched_fpr.py`, `results/`) |

The archive is also on Zenodo: https://doi.org/10.5281/zenodo.23845125 (all versions).

## Contents

| Path | What it is |
|---|---|
| `hyros_experiments.py` | Models, preprocessing, normalisation, augmentation, attacks, calibration; experiments E0–E6 |
| `e7_matched_fpr.py` | Matched-FPR recall for all eight models at fixed thresholds (first analysis of this kind) |
| `revision_experiments.py` | R0 de-duplication rules, R1 contemporaneous control (E2c), R2 near-duplicate audit, R3 strict cross-corpus test and recalibration, R4/R5 matched-FPR attacks and McNemar tables |
| `revision_experiments2.py` | R6 matched FPR with a joint bootstrap (threshold and training-seed variability), held-out edit family, LR+norm; R7 E2 intervals, mailboxes and HourAhead notices; R8 E3 joint bootstrap, paired recalibration draws, exact near-duplicate re-check; R9 E2c by folder and sender; R10 recalibration rule |
| `revision_experiments3.py` | R11 sensitivity of the component effects to the HourAhead notices (cluster bootstrap, recall averaged over matched FPR 0.5–3%), paired interval for the E2 calibrated FPR difference, McNemar by class; R12 de-duplication rule of the replicated study and a subject plus word-set check for further copies in the strict set; rebuilds and checks `splits/enron_eid_to_release_row.csv` |
| `grid_search_e2.py` | Grid search for C on the E2 training set only (R13) |
| `make_tables_rev2.py` | Builds every table of the manuscript and of the supplementary PDF from the CSV files (`tables/`) |
| `make_figures.py`, `make_figures_rev.py` | Fig. 1 and Fig. S1 (`make_figures.py`); Figs. 2 and 3 (`make_figures_rev.py`) |
| `run_all.sh` | Regenerates everything in order |
| `results/`, `results_rev/`, `results_rev2/`, `results_rev3/` | Result files; logs `run_log.txt`, `revision_log.txt`, `revision2_log.txt` |
| `results_rev/splits/` | Split indices and the strict external test set |
| `figures/` | The figures of the paper (600 dpi) |

## Data

Both corpora are public and are not redistributed here.

* Enron-Spam (Metsis, Androutsopoulos and Paliouras, 2006), single-file release `enron_spam_data.csv`
  (33,716 rows; SHA-256 begins `009c8635`) from `https://github.com/MWiechmann/enron_spam_data`, where it is
  distributed as `enron_spam_data.zip`. Unzip it and place the CSV at `data/enron_spam_data.csv`.
* SpamAssassin public corpus as distributed in the npm package `@stdlib/datasets-spam-assassin@0.2.3`
  (requires Node.js): `npm pack @stdlib/datasets-spam-assassin@0.2.3 && mkdir -p data/sa && tar -xzf stdlib-datasets-spam-assassin-0.2.3.tgz -C data/sa`,
  so that the JSON files are in `data/sa/package/data/<folder>/`.

## Running

```
python3 -m pip install -r requirements.txt
./run_all.sh
```

Versions used: Python 3.13, scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, matplotlib 3.11.2,
datasketch 2.0.0. Everything runs on a CPU; the full run takes about two to three hours on two cores.
`revision_experiments2.py` caches model scores in `results_rev2/scores/` (not included, to keep the archive small; regenerated
by the run, which takes about 30 minutes). `revision_experiments3.py` reads these cached scores and runs in about a minute.
`run_all.sh` writes into the result folders shipped with the archive; to compare a rerun with the shipped files, copy the
archive first and run it in the copy.

## Which file supports which table or figure

| Manuscript item | Result files |
|---|---|
| Table 1 (E1, E2) | `results/E1_random_split_summary.csv`, `results_rev2/R7_e2_thresholds.csv` |
| Table 2 (E2c) | `results_rev/R1_contemporaneous_all.csv`, `results_rev2/R9_e2c_exact_intervals.csv` |
| Table 3 (E3) | `results_rev2/R8_e3_default.csv`, `R8_recalibration_summary.csv`, `R8_e3_matched_fpr1_joint.csv` |
| Table 4 (matched-FPR attacks) | `results_rev2/R6_matched_fpr_joint.csv` (target_FPR = 0.015) |
| Table 5 (component effects) | `results_rev2/R6_component_effects_joint.csv` (target_FPR = 0.015); sensitivity in `results_rev3/R11_component_sensitivity.csv` |
| Fig. 1 | `results/E0_year_by_class.csv` |
| Fig. 2 | `results_rev2/R8_recalibration_summary.csv` (all draws in `R8_recalibration_draws.csv`) |
| Fig. 3 | `results_rev2/R6_matched_fpr_joint.csv` |
| Supplementary tables | built by `make_tables_rev2.py`, which names the result file used for each table |

## Preprocessing and the 30,137 messages

`load_enron()` concatenates subject and body, drops empty messages and removes duplicates after lower-casing and
whitespace normalisation, leaving 30,137 messages (15,590 ham, 14,547 spam). The study being replicated reports 30,089. Its own rule (lower-case, keep letters, digits, `$` and `%`, collapse
whitespace) gives 30,090, and 30,089 once messages left empty by the cleaning are dropped
(`results_rev3/R12_dedup_replicated_rule.csv`). `results_rev/R0_dedup_rules.csv` applies six further rules (after the
raw count) that retain between 29,779 and 30,494 messages.

## Split indices

* `splits/enron_eid_to_release_row.csv` (rebuilt and checked by `revision_experiments3.py`): `eid` is the position of a message after `load_enron()`; `release_row` is
  its 0-based row in `enron_spam_data.csv` (the column `Message ID` of the release equals `release_row`). The six
  blocks of release rows (0–5171, 5172–11028, 11029–16540, 16541–22540, 22541–27715, 27716–33715) are Enron1–Enron6.
* `splits/E1_seed{0..4}_test_ids.csv`: test messages of the five random splits of E1.
* `splits/E2_parts.csv`: part of every message in the class-wise chronological split of E2 (`fit`,
  `calibration`, `test`). `revision_experiments2.py` checks that the split it rebuilds is identical to this file
  (the split sorts day-level dates, so ties are ordered by pandas' default sort; with the pinned versions the order is
  reproducible, and `E2_parts.csv` is the reference).
* `splits/SA_strict_ids.csv`: SpamAssassin messages of the strict external test set (4,812).
* `splits/R1_window_ids.csv`: messages of the July–September 2002 window of E2c with their delivery time.
* `results_rev/R2_overlap_flags.csv`, `results_rev2/R8_exact_near_duplicates.csv`: overlap checks per message.

## Calibration

`thr_at_fpr()` returns the smallest calibration-ham score at or above the empirical (1 − α) quantile
(`numpy.quantile(..., method="higher")`) plus 1e-12; a message is flagged when its score exceeds that quantile.
With α = 0.01 this is the largest score for n ≤ 100, the second largest for n = 200 and the fifth for n = 500.

## Random seeds

| Step | Seed |
|---|---|
| E1 random splits | `random_state` = 0–4 |
| Grid search for C | seed-0 training split, `StratifiedKFold(3, shuffle=True, random_state=0)` |
| HyRoS augmentation | `random.Random(seed)`; in E1 and the random splits of E2c the seed equals the split seed (0–4); seed 0 elsewhere in E2–E5 and R1–R5; seeds 0–4 in R6–R8 and R11–R12 |
| Attacks | `random.Random(7 * 1000 + attack index)`; held-out edits `random.Random(99/101/103)` |
| Few-shot recalibration | `numpy.random.default_rng(1000 * n + draw)` (10 draws, E3); `default_rng(10000 * n + draw)` (200 draws, R3, R8) |
| E2c random splits | `random_state` = 0–4 for the 80/20 split and the stratified 10% calibration split |
| Bootstrap | E4: seed 0, 500 resamples; R3–R5: one `default_rng(2026)` used in order; R6 and R8: one `default_rng(2027)` used in order (2,000 replicates); R11: `default_rng(2027)` per analysis and `default_rng(2028)`; R12: `default_rng(2029)` |
| MinHash | `seed=1` |

The linear SVM (liblinear) is not given a seed; its results were identical across reruns.

## Licence

MIT (see `LICENSE`).
