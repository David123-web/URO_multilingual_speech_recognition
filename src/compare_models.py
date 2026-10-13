"""Compare the zero-shot model, a symbol-remap baseline and the fine-tuned
accent-robust model (AR-w2v2) on the held-out test set
(unseen speakers x unseen sentences), all scored against CANONICAL phones.

Models
  M0        zero-shot ctaguchi model (results/predictions_v2.csv, test rows)
  M0+remap  M0 output with each output symbol replaced by the English phone it is
            most often aligned to on the TRAINING split (fixes pure notation
            mismatch such as [r]->/ɹ/, [i]->/ɪ/ without any acoustic learning)
  AR-w2v2   fine-tuned model (results/finetune/predictions_test.csv)

Metrics
  PER, PFER (vs canonical) overall and per L1, with utterance-bootstrap CIs
  accent gap: spread of PER across L1 groups
  H1 within-phone excess for consonants missing from the L1 (raw and L1-adjusted)
  accent recovery: on the hand-annotated test utterances, share of phones the
      speaker MISpronounced for which the model still outputs the intended phone
  L1 identification accuracy (AR-w2v2 only)

Usage: python src/compare_models.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import L1_ORDER, loose, segment  # noqa: E402
from run_pfer_experiment import align, get_pfer_fn  # noqa: E402

R = Path("results")
OUT = R / "finetune"
VOWELS = set("aeiouæɑɒɔəɚɛɜɝɪʊʌ")


def segs(s):
    return [x for x in segment(str(s) if isinstance(s, str) else "") if x.strip()]


def learn_remap(m0: pd.DataFrame, train_keys: set) -> dict:
    cnt = collections.defaultdict(collections.Counter)
    for ref, hyp in zip(m0[m0.utt_key.isin(train_keys)].reference, m0[m0.utt_key.isin(train_keys)].prediction_raw):
        path, _ = align(ref.split(), segs(hyp))
        for r, h in path:
            if r is not None and h is not None:
                cnt[loose(h)][r] += 1
    return {h: c.most_common(1)[0][0] for h, c in cnt.items()}


def score(df: pd.DataFrame, name: str, pfer_fn) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        ref = r.reference.split()
        hyp = r.hyp
        path, ed = align(ref, hyp)
        row = {"model": name, "utt_key": r.utt_key, "L1": r.L1, "PER": ed / len(ref), "path": path}
        if pfer_fn:
            row["PFER"] = pfer_fn("".join(hyp), "".join(ref)) / len(ref)
        rows.append(row)
    return pd.DataFrame(rows)


def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def phone_err(sc: pd.DataFrame) -> pd.DataFrame:
    rec = [(l1, r, h is not None and loose(h) == loose(r))
           for l1, path in zip(sc.L1, sc.path) for r, h in path if r is not None]
    t = pd.DataFrame(rec, columns=["L1", "phone", "ok"])
    g = t.groupby(["L1", "phone"]).agg(n=("ok", "size"), acc=("ok", "mean")).reset_index()
    g["error_rate"] = 1 - g.acc
    return g[g.n >= 20]


def h1(pe: pd.DataFrame, inv: pd.DataFrame, consonants_only=True, n_perm=2000, seed=0):
    pe = pe.copy()
    pe["missing"] = [inv.loc[p, l] in ("absent", "allophone", "parts") if p in inv.index else False
                     for p, l in zip(pe.phone, pe.L1)]
    if consonants_only:
        pe = pe[~pe.phone.str[0].isin(VOWELS)]

    def adj(d):
        return d.error_rate - d.groupby("phone").error_rate.transform("mean") \
            - d.groupby("L1").error_rate.transform("mean") + d.error_rate.mean()

    def contrast(d, vals):
        diffs = []
        for ph, g in d.groupby("phone"):
            m = g.missing.to_numpy()
            if 0 < m.sum() < len(m):
                v = vals.loc[g.index].to_numpy()
                diffs.append(v[m].mean() - v[~m].mean())
        return float(np.mean(diffs)) if diffs else np.nan, len(diffs)

    raw, k = contrast(pe, pe.error_rate)
    a = adj(pe)
    obs, _ = contrast(pe, a)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        s = pe.copy()
        s["missing"] = s.groupby("phone").missing.transform(lambda x: rng.permutation(x.to_numpy()))
        null.append(contrast(s, adj(s))[0])
    p = (np.sum(np.abs(np.array(null)) >= abs(obs)) + 1) / (n_perm + 1)
    return {"raw_excess": raw, "L1_adjusted_excess": obs, "phones": k, "perm_p": p}


def accent_recovery(df: pd.DataFrame, triples: pd.DataFrame):
    """On annotated utterances: for each intended phone, was it mispronounced
    (substitution/deletion) and did the model output the intended phone?"""
    tri = dict(zip(triples.utt_key, triples.triples))
    out = []
    for r in df.itertuples():
        if r.utt_key not in tri:
            continue
        t = [x for x in json.loads(tri[r.utt_key]) if x[0] is not None]   # drop insertions
        canon = [x[0] for x in t]
        path, _ = align(canon, r.hyp)
        k = 0
        for c, h in path:
            if c is None:
                continue
            typ = t[k][2]
            perceived = t[k][1]
            k += 1
            hit = h is not None and loose(h) == loose(c)
            said = h is not None and perceived is not None and loose(h) == loose(perceived)
            out.append((r.L1, typ in ("s", "d"), hit, said))
    o = pd.DataFrame(out, columns=["L1", "mispronounced", "outputs_intended", "outputs_what_was_said"])
    mis, ok = o[o.mispronounced], o[~o.mispronounced]
    return {"n_mispronounced_phones": int(len(mis)),
            "recovery_on_mispronounced": float(mis.outputs_intended.mean()),
            "outputs_what_was_said_on_mispronounced": float(mis.outputs_what_was_said.mean()),
            "accuracy_on_correctly_pronounced": float(ok.outputs_intended.mean())}, o


def main():
    man = pd.read_csv("data/finetune_manifest.csv")
    test_keys = set(man[man.split == "test"].utt_key)
    train_keys = set(man[man.split == "train"].utt_key)
    inv = pd.read_csv(R / "inventory_phone_by_L1.csv", index_col=0)
    triples = pd.read_csv("data/annotation_triples.csv")
    pfer_fn = get_pfer_fn()

    m0 = pd.read_csv(R / "predictions_v2.csv")
    remap = learn_remap(m0, train_keys)
    (OUT / "remap_table.json").write_text(json.dumps(remap, ensure_ascii=False, indent=0))
    t0 = m0[m0.utt_key.isin(test_keys)].copy()
    t0["hyp"] = t0.prediction_raw.map(segs)
    t0r = t0.copy()
    t0r["hyp"] = t0r.hyp.map(lambda s: [remap.get(loose(x), x) for x in s])
    models = {"M0 zero-shot": t0, "M0 + symbol remap": t0r}
    ft = OUT / "predictions_test.csv"
    if ft.exists():
        t1 = pd.read_csv(ft)
        t1["hyp"] = t1.prediction_raw.map(segs)
        models["AR-w2v2 (fine-tuned)"] = t1

    summary, per_l1, h1s, recs = [], [], [], []
    scored = {}
    for name, df in models.items():
        sc = score(df, name, pfer_fn)
        scored[name] = sc
        row = {"model": name, "utterances": len(sc), "PER": sc.PER.mean()}
        row["PER_ci"] = boot(sc.PER).round(4).tolist()
        if "PFER" in sc:
            row["PFER"] = sc.PFER.mean()
        l1m = sc.groupby("L1").PER.mean()
        row["accent_gap_PER_max_minus_min"] = l1m.max() - l1m.min()
        rec, _ = accent_recovery(df, triples)
        row.update(rec)
        summary.append(row)
        for l1, g in sc.groupby("L1"):
            lo, hi = boot(g.PER)
            per_l1.append({"model": name, "L1": l1, "PER": g.PER.mean(), "PER_lo": lo, "PER_hi": hi,
                           "PFER": g.PFER.mean() if "PFER" in g else np.nan})
        hh = h1(phone_err(sc), inv)
        hh["model"] = name
        h1s.append(hh)
        if name.startswith("AR"):
            acc = float((df.L1 == df.L1_pred).mean())
            cm = pd.crosstab(df.L1, df.L1_pred).reindex(index=L1_ORDER, columns=L1_ORDER, fill_value=0)
            cm.to_csv(OUT / "l1_confusion_test.csv")
            summary[-1]["L1_id_accuracy"] = acc
            summary[-1]["L1_id_speaker_majority_correct"] = int(
                sum(g.L1_pred.mode().iloc[0] == g.L1.iloc[0] for _, g in df.groupby("speaker")))
    S = pd.DataFrame(summary)
    S.to_csv(OUT / "comparison_summary.csv", index=False)
    P = pd.DataFrame(per_l1)
    P.to_csv(OUT / "comparison_by_L1.csv", index=False)
    H = pd.DataFrame(h1s)
    H.to_csv(OUT / "comparison_h1.csv", index=False)
    tags = {"M0 zero-shot": "M0", "M0 + symbol remap": "M0_remap", "AR-w2v2 (fine-tuned)": "AR_w2v2"}
    for name, sc in scored.items():
        phone_err(sc).assign(model=name).to_csv(OUT / f"phone_error_{tags[name]}.csv", index=False)
    plot(P, OUT / "figures" / "per_by_L1_models.png")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(S.round(4).to_string(index=False))
    print(P.pivot(index="L1", columns="model", values="PER").round(4))
    print(H.round(4).to_string(index=False))


def plot(P: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    models = list(dict.fromkeys(P.model))
    colors = ["#b9c4d0", "#7f93a8", "#2f6db5"][:len(models)]
    x = np.arange(len(L1_ORDER))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for k, (mname, c) in enumerate(zip(models, colors)):
        d = P[P.model == mname].set_index("L1").reindex(L1_ORDER)
        y = d.PER * 100
        ax.bar(x + (k - (len(models) - 1) / 2) * w, y, w * 0.92, color=c, label=mname)
        ax.errorbar(x + (k - (len(models) - 1) / 2) * w, y,
                    yerr=[y - d.PER_lo * 100, d.PER_hi * 100 - y], fmt="none", ecolor="#333", lw=0.8, capsize=2)
    ax.set_xticks(x, L1_ORDER)
    ax.set_ylabel("PER vs intended phones (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Held-out speakers & sentences: phone error by L1", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
