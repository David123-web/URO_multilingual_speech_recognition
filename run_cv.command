#!/bin/bash
# Double-click to run cross-validation folds 1-3 (~95 min each, ~5 h total). Keeps the Mac awake.
# Fold 0 is the original run (models/ar_w2v2, results/finetune). Finished folds are skipped on rerun.
cd "$(dirname "$0")"
mkdir -p results/finetune
exec > >(tee -a results/finetune/cv_log.txt) 2>&1
echo "===== cross-validation started $(date) ====="
source .venv/bin/activate || { echo "venv not found"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))

for k in ${FOLDS:-1 2 3}; do
  if [ -f results/finetune/fold$k/predictions_test.csv ]; then echo "fold $k already done, skipping"; continue; fi
  echo "----- fold $k: $(date) -----"
  [ -f data/finetune_manifest_fold$k.csv ] || python src/cv_fold.py prepare --fold $k || exit 1
  caffeinate -i -s python src/cv_fold.py train --fold $k --ram-gb $RAM --max-hours ${MAX_HOURS:-3} \
    || { echo "FOLD $k TRAINING FAILED - tell Claude"; exit 1; }
  python src/cv_fold.py predict --fold $k || { echo "FOLD $k PREDICT FAILED - tell Claude"; exit 1; }
done
echo "===== CV ALL DONE $(date) - tell Claude ====="
