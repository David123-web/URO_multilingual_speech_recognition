#!/bin/bash
# Double-click to run. Output is also saved to results/run_log.txt
cd "$(dirname "$0")"
exec > >(tee -a results/run_log.txt) 2>&1
echo "===== run started $(date) ====="
source .venv/bin/activate || { echo "venv not found"; exit 1; }
python -c "import torch, transformers, librosa, panphon, scipy, matplotlib; print('packages OK')" || exit 1

echo "----- 1. self-tests -----"
python tests/test_pipeline.py || exit 1
python src/inventory_analysis.py --out results > /dev/null && echo "inventory analysis OK"

echo "----- 2. model smoke test -----"
python src/smoke_test.py results || exit 1

echo "----- 3. look for L2-ARCTIC -----"
ROOT=$(find data -maxdepth 3 -type d -name "ABA" 2>/dev/null | head -1 | xargs -I{} dirname {})
if [ -z "$ROOT" ]; then
  ZIP=$(ls data/l2arctic*.zip ~/Downloads/l2arctic*.zip 2>/dev/null | head -1)
  if [ -n "$ZIP" ]; then
    echo "unzipping $ZIP"; unzip -q -o "$ZIP" -d data/l2arctic_release_v5.0
    for z in data/l2arctic_release_v5.0/*.zip data/l2arctic_release_v5.0/*/*.zip; do
      [ -f "$z" ] && unzip -q -o "$z" -d "$(dirname "$z")"; done
    ROOT=$(find data -maxdepth 3 -type d -name "ABA" | head -1 | xargs -I{} dirname {})
  fi
fi
if [ -z "$ROOT" ]; then
  echo "L2-ARCTIC not found (put the zip in data/ or ~/Downloads). Stopping after tests."
  echo "===== done $(date) ====="; exit 0
fi
echo "corpus at $ROOT"

echo "----- 4. inference (quick pass: 150 utterances/speaker) -----"
python src/run_pfer_experiment.py infer --l2arctic "$ROOT" --out results/predictions.csv --per-speaker ${PER_SPEAKER:-150}
echo "----- 5. analysis -----"
python src/run_pfer_experiment.py analyze --predictions results/predictions.csv --out results
python src/run_pfer_experiment.py analyze --predictions results/predictions.csv --out results --reference perceived
echo "===== done $(date) ====="
