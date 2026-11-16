#!/usr/bin/env bash
# Regenerates every result file, table and figure of the revised manuscript.
# Expects the two public corpora in data/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
ENRON=data/enron_spam_data.csv
SA=data/sa/package/data
python3 hyros_experiments.py --enron "$ENRON" --spamassassin "$SA" --out results --seeds 5
python3 e7_matched_fpr.py --enron "$ENRON" --out results
python3 revision_experiments.py --enron "$ENRON" --spamassassin "$SA" --out results_rev
python3 revision_experiments2.py --enron "$ENRON" --spamassassin "$SA" --out results_rev2 --prev results_rev
python3 revision_experiments3.py --enron "$ENRON" --spamassassin "$SA" --rev results_rev --rev2 results_rev2 --out results_rev3
python3 grid_search_e2.py --enron "$ENRON" --parts results_rev/splits/E2_parts.csv --out results_rev3
TABLES_OUT=tables python3 make_tables_rev2.py
FIG_DPI=600 python3 make_figures.py --results results --out figures_original
FIG_DPI=600 python3 make_figures_rev.py --results results_rev2 --out figures
cp figures_original/years.png figures_original/pipeline.png figures/
echo "Done. Result files in results*/, tables in tables/, figures in figures/."
