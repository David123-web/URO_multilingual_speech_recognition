"""Part B: how well does a universal IPA recogniser transcribe L2 English,
and do its errors line up with what is missing from the speaker's L1?

Two sub-commands:

  infer    Run ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns over
           L2-ARCTIC v5.0 and write one row per utterance.
           python src/run_pfer_experiment.py infer --l2arctic /path/to/l2arctic_release_v5.0 \
                  --out results/predictions.csv [--per-speaker 150] [--device mps]

  analyze  Score predictions (PER, length-normalised PFER), aggregate per L1 and
           speaker, align phones, and test the L1-inventory hypothesis.
           python src/run_pfer_experiment.py analyze --predictions results/predictions.csv --out results
           (also accepts the 2024 notebook's outputs.csv; see load_predictions)

Requirements: infer -> torch, transformers, librosa ;  analyze -> pandas, numpy,
scipy, matplotlib, and panphon (for PFER; PER still works without it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (L1_ORDER, SPEAKER_L1, canonical_ipa_from_textgrid, loose,  # noqa: E402
                    perceived_ipa_from_annotation, segment)

MODEL_ID = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"


# =============================================================================
# 1. Inference
# =============================================================================
def iter_utterances(root: Path, per_speaker: int | None = None):
    """Yield (speaker, utt_id, wav_path, textgrid_path, annotation_path|None).

    Files are paired by utterance id, not by directory order: the 2024 code
    zipped three os.listdir() results together, which silently misaligns audio
    and transcript whenever one folder is missing a file.
    """
    for spk_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in SPEAKER_L1):
        base = spk_dir
        if not (base / "wav").exists():           # zip-extracted layout SPK/SPK/wav
            inner = [d for d in spk_dir.iterdir() if d.is_dir()]
            base = inner[0] if inner else spk_dir
        tgs = {p.stem: p for p in (base / "textgrid").glob("*.TextGrid")}
        anns = {p.stem: p for p in (base / "annotation").glob("*.TextGrid")} \
            if (base / "annotation").exists() else {}
        wavs = sorted((base / "wav").glob("*.wav"))
        n = 0
        for wav in wavs:
            if wav.stem not in tgs:
                continue
            yield spk_dir.name, wav.stem, wav, tgs[wav.stem], anns.get(wav.stem)
            n += 1
            if per_speaker and n >= per_speaker:
                break


def cmd_infer(args):
    import librosa
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "model_vocab.json").write_text(
        json.dumps(sorted(processor.tokenizer.get_vocab()), ensure_ascii=False))

    done = set()
    if out.exists() and not args.overwrite:          # resumable
        done = set(pd.read_csv(out).utt_key)
    rows, n = [], 0
    for spk, utt, wav, tg, ann in iter_utterances(Path(args.l2arctic), args.per_speaker):
        key = f"{spk}/{utt}"
        if key in done:
            continue
        audio, _ = librosa.load(wav, sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            ids = model(inputs).logits.argmax(-1)
        pred = processor.batch_decode(ids)[0]
        rows.append({
            "utt_key": key, "speaker": spk, "L1": SPEAKER_L1[spk], "utt_id": utt,
            "reference": " ".join(canonical_ipa_from_textgrid(tg)),
            "perceived": " ".join(perceived_ipa_from_annotation(ann)) if ann else "",
            "prediction_raw": pred,
            "duration_s": round(len(audio) / 16000, 3),
        })
        n += 1
        if n % 200 == 0:
            _flush(rows, out)
            rows = []
            print(f"{n} utterances done", flush=True)
    _flush(rows, out)


def _flush(rows, out: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False, encoding="utf-8")


# =============================================================================
# 2. Scoring
# =============================================================================
def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "prediction_raw" in df:                                    # new format
        df["ref_segs"] = df.reference.fillna("").str.split()
        df["perc_segs"] = df.perceived.fillna("").map(lambda s: s.split() if s else [])
        df["hyp_segs"] = df.prediction_raw.fillna("").map(segment).map(lambda s: [x for x in s if x.strip()])
        df["legacy"] = False
    else:                                                         # 2024 outputs.csv
        df = df.rename(columns={"ethnicity": "L1", "predictions": "prediction_raw",
                                "transcripts": "reference"})
        df["speaker"] = pd.NA                                     # not saved in 2024
        df["ref_segs"] = df.reference.fillna("").map(lambda x: segment(x, merge_untied_affricates=True))
        df["perc_segs"] = [[] for _ in range(len(df))]
        df["hyp_segs"] = df.prediction_raw.fillna("").map(segment).map(lambda s: [x for x in s if x.strip()])
        df["legacy"] = True
        if "pfer" in df:
            df = df.rename(columns={"pfer": "pfer_raw_2024"})
    return df[df.ref_segs.map(len) > 0].reset_index(drop=True)


def align(ref: list[str], hyp: list[str]):
    """Levenshtein alignment; segments match if equal after stripping
    diacritics (length, aspiration, ...).  Returns list of (ref|None, hyp|None)."""
    R, H = [loose(x) for x in ref], [loose(x) for x in hyp]
    n, m = len(R), len(H)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                          D[i - 1, j - 1] + (R[i - 1] != H[j - 1]))
    i, j, path = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (R[i - 1] != H[j - 1]):
            path.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            path.append((ref[i - 1], None)); i -= 1
        else:
            path.append((None, hyp[j - 1])); j -= 1
    return path[::-1], int(D[n, m])


def get_pfer_fn():
    try:
        import panphon.distance
        dist = panphon.distance.Distance()
        return lambda h, r: dist.feature_edit_distance(h, r)
    except Exception:  # noqa: BLE001
        print("WARNING: panphon not installed -> PFER skipped (pip install panphon)")
        return None


def score(df: pd.DataFrame, reference_col="ref_segs") -> pd.DataFrame:
    pfer_fn = get_pfer_fn()
    per, pfer, alignments = [], [], []
    for ref, hyp in zip(df[reference_col], df.hyp_segs):
        path, ed = align(ref, hyp)
        per.append(ed / len(ref))
        alignments.append(path)
        if pfer_fn:
            # panphon: substitution = fraction of features differing,
            # insertion/deletion = 1; divide by reference length -> rate.
            pfer.append(pfer_fn("".join(hyp), "".join(ref)) / len(ref))
    out = df.copy()
    out["PER"] = per
    if pfer_fn:
        out["PFER"] = pfer
    out["alignment"] = alignments
    return out


# =============================================================================
# 3. Aggregation and hypothesis test
# =============================================================================
def bootstrap_ci(values: np.ndarray, groups: np.ndarray | None, n=2000, seed=0):
    """95% CI of the mean.  Resamples speakers (clusters) when available,
    because utterances from one speaker are not independent."""
    rng = np.random.default_rng(seed)
    if groups is None or pd.isna(groups).all():
        idx = rng.integers(0, len(values), (n, len(values)))
        means = values[idx].mean(1)
    else:
        ug = np.unique(groups)
        by = {g: values[groups == g] for g in ug}
        means = np.array([np.concatenate([by[g] for g in rng.choice(ug, len(ug))]).mean()
                          for _ in range(n)])
    return np.percentile(means, [2.5, 97.5])


def per_l1_table(df: pd.DataFrame, metrics) -> pd.DataFrame:
    rows = []
    for l1 in L1_ORDER:
        g = df[df.L1 == l1]
        if g.empty:
            continue
        r = {"L1": l1, "utterances": len(g),
             "speakers": g.speaker.nunique() if g.speaker.notna().any() else "n/a"}
        for m in metrics:
            lo, hi = bootstrap_ci(g[m].to_numpy(), g.speaker.to_numpy())
            r[f"{m}_mean"], r[f"{m}_ci_low"], r[f"{m}_ci_high"] = g[m].mean(), lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def phone_error_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (L1, English target phone): how often the recogniser's output at
    that position is NOT the target (substitution or deletion)."""
    rec = []
    for l1, path in zip(df.L1, df.alignment):
        for r, h in path:
            if r is None:
                continue
            rec.append((l1, r, h is not None and loose(h) == loose(r),
                        loose(h) if h is not None else "∅"))
    t = pd.DataFrame(rec, columns=["L1", "phone", "correct", "output"])
    agg = t.groupby(["L1", "phone"]).agg(n=("correct", "size"), acc=("correct", "mean")).reset_index()
    agg["error_rate"] = 1 - agg.acc
    top = (t[~t.correct].groupby(["L1", "phone"]).output
           .agg(lambda s: ", ".join(f"{k} ({v})" for k, v in s.value_counts().head(3).items()))
           .rename("most_common_outputs").reset_index())
    return agg.merge(top, on=["L1", "phone"], how="left")


def inventory_hypothesis(phone_err: pd.DataFrame, inventory_csv: Path, model_vocab: set[str],
                         n_perm=5000, seed=0):
    """H1: for the same English phone, the error rate is higher for speakers
    whose L1 lacks that phone (absent / allophone-only / parts-only).

    Within-phone contrasts control for phones that are simply hard for the
    model.  Phones the model cannot emit (outside its output vocabulary) are
    excluded, because their error rate is ~100% for every group.
    Significance: permutation test shuffling L1 status within each phone.
    """
    inv = pd.read_csv(inventory_csv, index_col=0)
    pe = phone_err[phone_err.n >= 30].copy()
    pe["status"] = [inv.loc[p, l1] if p in inv.index and l1 in inv.columns else np.nan
                    for p, l1 in zip(pe.phone, pe.L1)]
    pe["missing_in_L1"] = pe.status.isin(["absent", "allophone", "parts"])
    # a phone is emittable if every character of it is in the model's output
    # vocabulary (diphthongs are emitted as two vowel tokens)
    vocab_chars = {c for v in model_vocab for c in v}
    pe["model_can_emit"] = pe.phone.map(lambda p: all(c in vocab_chars for c in loose(p)))
    use = pe[pe.model_can_emit].dropna(subset=["status"])

    # per phone: error rates and missing-in-L1 flags, only phones with both kinds
    blocks = []
    for _, g in use.groupby("phone"):
        m = g.missing_in_L1.to_numpy()
        if 0 < m.sum() < len(m):
            blocks.append((g.error_rate.to_numpy(), m))
    k = len(blocks)

    def contrast(flags):
        return np.mean([e[f].mean() - e[~f].mean() for (e, _), f in zip(blocks, flags)])

    obs = contrast([m for _, m in blocks]) if k else np.nan
    rng = np.random.default_rng(seed)
    null = np.array([contrast([rng.permutation(m) for _, m in blocks]) for _ in range(n_perm)]) \
        if k else np.array([])
    p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1) if k else np.nan
    return {"mean_within_phone_difference_in_error_rate": obs, "phones_with_contrast": k,
            "permutation_p_value_two_sided": p,
            "phones_excluded_model_cannot_emit": sorted(pe[~pe.model_can_emit].phone.unique())}, pe


def cmd_analyze(args):
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = load_predictions(Path(args.predictions))
    if args.reference == "perceived":
        df = df[df.perc_segs.map(len) > 0]
        col = "perc_segs"
    else:
        col = "ref_segs"
    scored = score(df, col)
    metrics = [m for m in ("PER", "PFER") if m in scored]

    vocab_file = Path(args.predictions).parent / "model_vocab.json"
    if vocab_file.exists():
        vocab = {loose(v) for v in json.loads(vocab_file.read_text())}
        vocab_src = "model tokenizer"
    else:  # proxy: everything the model ever produced on this corpus
        vocab = {loose(s) for segs in scored.hyp_segs for s in segs}
        vocab_src = "segments observed in predictions (proxy)"

    tag = "" if args.reference == "canonical" else "_perceived"
    l1 = per_l1_table(scored, metrics)
    l1.to_csv(out / f"pfer_by_L1{tag}.csv", index=False)
    if scored.speaker.notna().any():
        (scored.groupby(["L1", "speaker"])[metrics].mean().reset_index()
         .to_csv(out / f"pfer_by_speaker{tag}.csv", index=False))
    pe = phone_error_table(scored)
    pe.to_csv(out / f"phone_error_by_L1{tag}.csv", index=False)
    hyp, pe2 = inventory_hypothesis(pe, out / "inventory_phone_by_L1.csv", vocab)
    hyp["vocab_source"] = vocab_src
    (out / f"hypothesis_test{tag}.json").write_text(json.dumps(hyp, indent=2, ensure_ascii=False, default=float))

    # Kruskal-Wallis across L1 groups on speaker means (or utterances if legacy)
    from scipy.stats import kruskal
    m0 = metrics[-1]
    unit = scored.groupby(["L1", "speaker"])[m0].mean().reset_index() \
        if scored.speaker.notna().any() else scored
    groups = [unit[unit.L1 == g][m0].to_numpy() for g in L1_ORDER if (unit.L1 == g).any()]
    H, pk = kruskal(*groups)
    (out / f"kruskal{tag}.json").write_text(json.dumps(
        {"metric": m0, "unit": "speaker means" if unit is not scored else "utterances",
         "H": H, "p": pk}, indent=2))

    plot_l1(l1, metrics[-1], out / "figures" / f"error_by_L1{tag}.png")
    plot_phone_heatmap(pe2, out / "figures" / f"phone_error_heatmap{tag}.png")
    # examples for the report
    scored.assign(hyp=scored.hyp_segs.map(" ".join), ref=scored[col].map(" ".join)) \
        .groupby("L1").head(3)[["L1", "ref", "hyp"] + metrics] \
        .to_csv(out / f"examples{tag}.csv", index=False)
    print(l1.round(4).to_string(index=False))
    print(json.dumps(hyp, indent=2, ensure_ascii=False, default=float))


def plot_l1(l1: pd.DataFrame, m: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = l1[f"{m}_mean"] * 100
    err = np.vstack([y - l1[f"{m}_ci_low"] * 100, l1[f"{m}_ci_high"] * 100 - y])
    ax.bar(l1.L1, y, color="#3b6ea5", width=0.6)
    ax.errorbar(l1.L1, y, yerr=err, fmt="none", ecolor="#222", capsize=4, lw=1)
    ax.set_ylabel(f"{m} (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Mean {m} by speaker L1 (95% bootstrap CI)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_phone_heatmap(pe: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = pe.pivot(index="L1", columns="phone", values="error_rate").reindex(L1_ORDER)
    miss = pe.pivot(index="L1", columns="phone", values="missing_in_L1").reindex(L1_ORDER)
    order = piv.mean().sort_values(ascending=False).index
    piv, miss = piv[order], miss[order]
    fig, ax = plt.subplots(figsize=(14, 3.8))
    im = ax.imshow(piv.values * 100, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = miss.iloc[i, j]
            if isinstance(v, (bool, np.bool_)) and v:
                ax.text(j, i, "•", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(order)), order)
    ax.set_yticks(range(len(piv.index)), piv.index)
    fig.colorbar(im, ax=ax, label="error rate (%)", fraction=0.025)
    ax.set_title("Per-phone recogniser error by L1  (• = phone not phonemic in that L1)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer")
    a.add_argument("--l2arctic", required=True)
    a.add_argument("--out", default="results/predictions.csv")
    a.add_argument("--per-speaker", type=int, default=None)
    a.add_argument("--device", default=None)
    a.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("analyze")
    b.add_argument("--predictions", required=True)
    b.add_argument("--out", default="results")
    b.add_argument("--reference", choices=["canonical", "perceived"], default="canonical",
                   help="canonical = forced-aligned dictionary phones (textgrid); "
                        "perceived = what annotators heard (3,599-utterance subset)")
    args = ap.parse_args()
    {"infer": cmd_infer, "analyze": cmd_analyze}[args.cmd](args)
