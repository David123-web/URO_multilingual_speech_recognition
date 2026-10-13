"""Speaker cross-validation for the accent-robust model, without editing
prepare_finetune.py / finetune.py: this wrapper imports them and swaps the
test speakers and output paths.

Fold k (1-3) takes, in each L1, the speaker k places after the fold-0 test
speaker (alphabetically, wrapping), so folds 0-3 together test all 24 speakers
once.  Fold 0 is the original run (data/finetune_manifest.csv, models/ar_w2v2,
results/finetune).  Sentences stay split by the same hash as fold 0.

  python src/cv_fold.py prepare --fold 1        # data/finetune_manifest_fold1.csv
  python src/cv_fold.py train   --fold 1 --ram-gb 24 --max-hours 3
  python src/cv_fold.py predict --fold 1        # results/finetune/fold1/predictions_test.csv
  python src/cv_fold.py smoke   --fold 1        # 2 steps, outputs to *_smoke and deleted after
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import finetune as ft  # noqa: E402
import prepare_finetune as pf  # noqa: E402
from common import SPEAKER_L1  # noqa: E402

FOLD0 = list(pf.TEST_SPEAKERS)


def fold_test_speakers(fold: int) -> list[str]:
    out = []
    for spk0 in FOLD0:
        group = sorted(s for s, l in SPEAKER_L1.items() if l == SPEAKER_L1[spk0])
        out.append(group[(group.index(spk0) + fold) % len(group)])
    return out


def manifest(fold: int) -> str:
    return f"data/finetune_manifest_fold{fold}.csv"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "smoke", "train", "predict"])
    ap.add_argument("--fold", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--l2arctic", default="data/l2arctic_release_v5.0")
    ap.add_argument("--ram-gb", type=float, default=16)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--l1-weight", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    a.manifest = manifest(a.fold)
    k = a.fold

    if a.cmd == "prepare":
        pf.TEST_SPEAKERS = fold_test_speakers(k)
        df = pf.build(Path(a.l2arctic))
        df.to_csv(a.manifest, index=False)
        print("fold", k, "test speakers:", pf.TEST_SPEAKERS)
        print(df.groupby(["split", "L1"]).size().unstack(fill_value=0))
        leak = set(df[df.split == "train"].utt_id) & set(df[df.split == "test"].utt_id)
        print("sentence overlap train/test:", len(leak))
        sys.exit(0)

    smoke = a.cmd == "smoke"
    ft.OUT = Path(f"results/finetune_smoke/fold{k}" if smoke else f"results/finetune/fold{k}")
    ft.CKPT = Path(f"models/ar_w2v2_fold{k}")
    a.model = str(ft.CKPT)
    if smoke:
        ft.train(a, smoke=True)          # smoke saves to models/ar_w2v2_smoke
        a.model, a.split = "models/ar_w2v2_smoke", "smoke"
        ft.predict(a)
        ft.log("SMOKE TEST OK")
        shutil.rmtree("results/finetune_smoke", ignore_errors=True)
        shutil.rmtree("models/ar_w2v2_smoke", ignore_errors=True)
    elif a.cmd == "train":
        ft.train(a)
    else:
        ft.predict(a)
