#!/bin/bash
# Double-click to fine-tune the accent-robust model (~3.5 h). Keeps the Mac awake.
cd "$(dirname "$0")"
mkdir -p results/finetune
exec > >(tee -a results/finetune/run_log.txt) 2>&1
echo "===== fine-tuning run started $(date) ====="
source .venv/bin/activate || { echo "venv not found"; exit 1; }
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
RAM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
echo "Mac memory: ${RAM} GB  |  chip: $(sysctl -n machdep.cpu.brand_string)"

[ -f data/finetune_manifest.csv ] || python src/prepare_finetune.py || exit 1

echo "----- 1. smoke test (2 training steps end-to-end) -----"
python src/finetune.py smoke --ram-gb $RAM || { echo "SMOKE TEST FAILED - tell Claude"; exit 1; }

echo "----- 2. training (time limit ${MAX_HOURS:-3} h; Mac kept awake) -----"
caffeinate -i -s python src/finetune.py train --ram-gb $RAM --max-hours ${MAX_HOURS:-3} || { echo "TRAINING FAILED - tell Claude"; exit 1; }

echo "----- 3. predictions on held-out speakers -----"
python src/finetune.py predict --split test && python src/finetune.py predict --split val
echo "===== ALL DONE $(date) - tell Claude ====="
