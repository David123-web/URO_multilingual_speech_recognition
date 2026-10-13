"""Part A: which English target phones are missing from each speaker group's L1?

Replaces the 2024 Jaccard comparison (notebook cells 10-19), which
  * tokenised whole unspaced utterances with lingpy, producing pseudo-segments
    such as 'ʌoʊ' and 'iʌʌ' (142 of 179 'phones' for the Arabic group),
  * matched PHOIBLE languages by substring of LanguageName (pulling in every
    Arabic variety, plus any language whose name merely contains the word),
  * compared 'g' (U+0067) with PHOIBLE's 'ɡ' (U+0261) and 'tʃ' with 't̠ʃ',
    so some shared segments could never match.

Usage:  python src/inventory_analysis.py --phoible phoible.csv --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import (ARABIC_ALL_ISO, ENGLISH_TARGETS_STRESS, L1_ISO, L1_ORDER,
                    loose, normalize_ipa)

STATUS_ORDER = ["exact", "close", "allophone", "parts", "absent"]


def language_inventory(df: pd.DataFrame, isos: list[str], rule: str = "majority"):
    """Return (phonemes, allophones) sets for a language.

    rule='majority': keep a segment if it occurs in >= half of the language's
    PHOIBLE inventories (robust to one idiosyncratic source);
    rule='union': keep it if any inventory lists it.
    """
    sub = df[df.ISO6393.isin(isos)]
    n_inv = sub.InventoryID.nunique()
    counts = sub.assign(seg=sub.Phoneme.map(normalize_ipa)) \
                .groupby("seg").InventoryID.nunique()
    thresh = 1 if rule == "union" else n_inv / 2
    phon = set(counts[counts >= thresh].index)
    # same majority rule on diacritic-stripped forms (t vs t̪ across sources)
    lcounts = sub.assign(seg=sub.Phoneme.map(loose)).groupby("seg").InventoryID.nunique()
    phon_loose = set(lcounts[lcounts >= thresh].index)
    allo = set()
    for a in sub.Allophones.dropna():
        allo |= {normalize_ipa(x) for x in str(a).split()}
    return phon, phon_loose, allo, n_inv


VOWELS = set("aeiouæɑɒɔəɚɛɜɝɪʊʌɐɘɵɨʉɯɤøœy")


def is_vowel(seg: str) -> bool:
    return seg[0] in VOWELS


def classify(target: str, phon: set[str], allo: set[str], phon_loose: set[str] | None = None) -> str:
    if phon_loose is None:
        phon_loose = {loose(p) for p in phon}
    if target in phon:
        return "exact"
    if loose(target) in phon_loose:
        return "close"          # same base segment, differing diacritic (t vs t̪, i vs iː)
    if target in allo or loose(target) in {loose(a) for a in allo}:
        return "allophone"      # exists in L1 only as a variant of another phoneme
    if len(target) == 2 and all(loose(c) in phon_loose for c in target):
        return "parts"          # diphthong/affricate absent as a unit, parts present
    return "absent"


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def run(phoible: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(phoible, low_memory=False)
    targets = ENGLISH_TARGETS_STRESS

    # --- sanity check: English inventories should cover the English targets
    rows = []
    for inv_id, g in df[df.ISO6393 == "eng"].groupby("InventoryID"):
        phon = {normalize_ipa(p) for p in g.Phoneme}
        st = [classify(t, phon, set()) for t in targets]
        miss = [t for t, s in zip(targets, st) if s == "absent"]
        rows.append({"InventoryID": inv_id, "source": g.Source.iloc[0],
                     "dialect": g.SpecificDialect.iloc[0],
                     "consonants_missing": [t for t in miss if not is_vowel(t)],
                     "vowels_missing": [t for t in miss if is_vowel(t)]})
    sanity = pd.DataFrame(rows)
    sanity.to_csv(out / "sanity_english_inventories.csv", index=False)

    # --- main analysis
    status = {}
    summary = []
    groups = [(l1, L1_ISO[l1], "majority") for l1 in L1_ORDER]
    groups.append(("Arabic (all PHOIBLE varieties)", ARABIC_ALL_ISO, "union"))
    for name, isos, rule in groups:
        phon, phon_loose, allo, n_inv = language_inventory(df, isos, rule)
        st = {t: classify(t, phon, allo, phon_loose) for t in targets}
        status[name] = st
        missing = [t for t in targets if st[t] in ("absent", "parts", "allophone")]
        cons = [t for t in targets if not is_vowel(t)]
        summary.append({
            "consonants_not_phonemic_in_L1": [t for t in cons if st[t] in ("absent", "allophone", "parts")],
            "n_consonants_not_phonemic": sum(st[t] in ("absent", "allophone", "parts") for t in cons),
            "n_consonant_targets": len(cons),
            "L1": name, "PHOIBLE_ISO": ",".join(isos), "n_inventories": n_inv, "rule": rule,
            "L1_inventory_size": len(phon),
            "n_targets": len(targets),
            "covered_exact_or_close": sum(st[t] in ("exact", "close") for t in targets),
            "allophone_only": [t for t in targets if st[t] == "allophone"],
            "diphthong_or_affricate_parts_only": [t for t in targets if st[t] == "parts"],
            "absent": [t for t in targets if st[t] == "absent"],
            "n_new_phones_for_learner": len(missing),
            "jaccard_loose_corrected": round(jaccard({loose(t) for t in targets},
                                                     phon_loose), 3),
        })
    summ = pd.DataFrame(summary)
    summ["coverage_pct"] = (100 * summ.covered_exact_or_close / summ.n_targets).round(1)
    summ.to_csv(out / "inventory_summary.csv", index=False)
    mat = pd.DataFrame(status)
    mat.index.name = "english_phone"
    mat.to_csv(out / "inventory_phone_by_L1.csv")
    plot(mat[L1_ORDER], out / "figures" / "inventory_heatmap.png")
    return sanity, summ, mat


def plot(mat: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {"exact": 0, "close": 1, "allophone": 2, "parts": 3, "absent": 4}
    labels = ["in L1 (exact)", "in L1 (diacritic differs)", "L1 allophone only",
              "only its parts in L1", "not in L1"]
    colors = ["#2a7f62", "#8cc9a8", "#f2d08a", "#f0a35e", "#c8453c"]
    # sort phones: most often missing first
    score = mat.apply(lambda r: sum(order[x] for x in r), axis=1)
    mat = mat.loc[score.sort_values(ascending=False).index]
    data = mat.apply(lambda c: c.map(order)).T.values
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.imshow(data, cmap=ListedColormap(colors), vmin=-0.5, vmax=4.5, aspect="auto")
    ax.set_xticks(range(len(mat.index)), mat.index, fontsize=11)
    ax.set_yticks(range(len(mat.columns)), mat.columns)
    ax.set_xlabel("English target phone (from L2-ARCTIC transcripts)")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - .5 for x in range(1, len(mat.index))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(mat.columns))], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    ax.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False, fontsize=9)
    ax.set_title("Is each English phone part of the speaker's first-language inventory? (PHOIBLE)",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phoible", type=Path, default=Path("phoible.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    a = ap.parse_args()
    sanity, summ, mat = run(a.phoible, a.out)
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(sanity.to_string(index=False))
    print(summ.drop(columns=["rule"]).to_string(index=False))
