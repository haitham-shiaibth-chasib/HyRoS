#!/usr/bin/env bash
# Regenerates every result, table, figure and number of the paper.
set -euo pipefail
cd "$(dirname "$0")"
python hyros_experiments.py --enron data/enron_spam_data.csv --spamassassin data/sa/package/data --out results --seeds 5
python e7_matched_fpr.py --enron data/enron_spam_data.csv --out results
python make_figures.py --results results --out figures
echo "Done. Results in results/, figures in figures/."
