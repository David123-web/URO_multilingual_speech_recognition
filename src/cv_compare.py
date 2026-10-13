"""Pool the 4 speaker-CV folds and compare M0, M0+remap and AR-w2v2 on all 24
held-out speakers, reusing the scoring functions of compare_models.py.

Per fold, the symbol remap is learned on that fold's TRAINING split only.
Speaker means are the unit for between-L1 and between-model tests (4 per L1).

Outputs results/finetune/cv/: summary.csv, by_L1.csv, by_speaker.csv, h1.csv, tests.json
Usage: python src/cv_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import compare_models as cm  # noqa: E402

OUT = Path("results/finetune/cv")
FOLDS = {0: ("data/finetune_manifest.csv", "results/finetune/predictions_test.csv")} | {
    k: (f"data/finetune_manifest_fold{k}.csv", f"results/finetune/fold{k}/predictions_test.csv") for k in (1, 2, 3)}
NAMES = ["M0 zero-shot", "M0 + symbol remap", "AR-w2v2 (fine-tuned)"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pfer_fn = cm.get_pfer_fn()
    inv = pd.read_csv(cm.R / "inventory_phone_by_L1.csv", index_col=0)
    triples = pd.read_csv("data/annotation_triples.csv")
    m0 = pd.read_csv(cm.R / "predictions_v2.csv")

    data = {n: [] for n in NAMES}
    for k, (man_f, ft_f) in FOLDS.items():
        man = pd.read_csv(man_f)
        test_keys = set(man[man.split == "test"].utt_key)
        remap = cm.learn_remap(m0, set(man[man.split == "train"].utt_key))
        t0 = m0[m0.utt_key.isin(test_keys)].copy()
        t0["hyp"] = t0.prediction_raw.map(cm.segs)
        t0r = t0.copy()
        t0r["hyp"] = t0r.hyp.map(lambda s: [remap.get(cm.loose(x), x) for x in s])
        t1 = pd.read_csv(ft_f)
        t1["hyp"] = t1.prediction_raw.map(cm.segs)
        assert set(t1.utt_key) == test_keys, f"fold {k}: prediction/test mismatch"
        for n, df in zip(NAMES, (t0, t0r, t1)):
            data[n].append(df.assign(fold=k))
        print(f"fold {k}: {len(test_keys)} test utts, speakers {sorted(t1.speaker.unique())}")

    summary, per_l1, h1s, spk = [], [], [], []
    for n in NAMES:
        df = pd.concat(data[n], ignore_index=True)
        sc = cm.score(df, n, pfer_fn).assign(speaker=df.speaker.values)
        s = sc.groupby(["L1", "speaker"])[["PER", "PFER"]].mean().reset_index().assign(model=n)
        spk.append(s)
        l1m = s.groupby("L1").PER.mean()
        H, p = kruskal(*[g.PER for _, g in s.groupby("L1")])
        rec, _ = cm.accent_recovery(df, triples)
        row = {"model": n, "utterances": len(sc), "speakers": len(s),
               "PER": s.PER.mean(), "PER_speaker_sd": s.PER.std(), "PFER": s.PFER.mean(),
               "accent_gap_PER_max_minus_min": l1m.max() - l1m.min(),
               "kruskal_L1_H": H, "kruskal_L1_p": p, **rec}
        if n.startswith("AR"):
            row["L1_id_accuracy"] = float((df.L1 == df.L1_pred).mean())
            row["L1_id_speaker_majority_correct"] = int(
                sum(g.L1_pred.mode().iloc[0] == g.L1.iloc[0] for _, g in df.groupby("speaker")))
        summary.append(row)
        for l1, g in s.groupby("L1"):
            per_l1.append({"model": n, "L1": l1, "PER": g.PER.mean(), "PER_sd": g.PER.std(),
                           "PER_min": g.PER.min(), "PER_max": g.PER.max(), "PFER": g.PFER.mean()})
        h1s.append(cm.h1(cm.phone_err(sc), inv) | {"model": n})

    S, P, H1 = pd.DataFrame(summary), pd.DataFrame(per_l1), pd.DataFrame(h1s)
    SP = pd.concat(spk, ignore_index=True)
    # paired speaker-level tests between models (24 pairs)
    w = SP.pivot_table(index="speaker", columns="model", values="PER")
    tests = {}
    for a, b in [(NAMES[0], NAMES[2]), (NAMES[1], NAMES[2]), (NAMES[0], NAMES[1])]:
        d = w[a] - w[b]
        tests[f"{a} vs {b}"] = {"mean_PER_reduction": float(d.mean()), "speakers_improved": int((d > 0).sum()),
                                "wilcoxon_p": float(wilcoxon(w[a], w[b]).pvalue)}
    S.to_csv(OUT / "summary.csv", index=False)
    P.to_csv(OUT / "by_L1.csv", index=False)
    SP.to_csv(OUT / "by_speaker.csv", index=False)
    H1.to_csv(OUT / "h1.csv", index=False)
    (OUT / "tests.json").write_text(json.dumps(tests, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(S.round(4).T.to_string())
    print(P.pivot(index="L1", columns="model", values="PER").round(4))
    print(P.pivot(index="L1", columns="model", values="PER_sd").round(4).add_suffix(" sd"))
    print(H1.round(4).to_string(index=False))
    print(json.dumps(tests, indent=2))


if __name__ == "__main__":
    main()
