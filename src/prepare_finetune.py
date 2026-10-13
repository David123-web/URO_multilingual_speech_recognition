"""Build the fine-tuning manifest for the accent-robust model.

Targets are the CANONICAL (intended) English phones from the forced-aligned
TextGrids, so the model learns to output /θɪŋk/ even when a speaker says [sɪŋk].

Split (both speaker- AND sentence-disjoint, because every L2-ARCTIC speaker reads
the same ~1,130 ARCTIC prompts and a speaker-only split would let the model
memorise sentences):
  * test speakers: one per L1 (ZHAA, TXHC, TNI, YKWK, NJS, TLV)
  * test sentences: ~20% of prompt ids, chosen by a fixed hash
  train = train speakers x train sentences
  val   = train speakers x test sentences (300 utts, sentence generalisation)
  test  = test speakers  x test sentences (unseen voices AND unseen sentences)

Usage: python src/prepare_finetune.py --l2arctic data/l2arctic_release_v5.0 --out data/finetune_manifest.csv
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import (SPEAKER_L1, canonical_ipa_from_textgrid,  # noqa: E402
                    perceived_ipa_from_annotation)

TEST_SPEAKERS = ["ZHAA", "TXHC", "TNI", "YKWK", "NJS", "TLV"]  # one per L1


def is_test_sentence(utt_id: str) -> bool:
    return int(hashlib.md5(utt_id.encode()).hexdigest(), 16) % 5 == 0


def to_label(phones: list[str]) -> str:
    """Label string in the model's character conventions: no spaces,
    affricates written with a tie bar (t͡ʃ, d͡ʒ) as in the model vocabulary."""
    tie = {"tʃ": "t͡ʃ", "dʒ": "d͡ʒ"}
    return "".join(tie.get(p, p) for p in phones)


def build(root: Path) -> pd.DataFrame:
    rows = []
    for spk in sorted(SPEAKER_L1):
        base = root / spk
        if not (base / "wav").exists():
            continue
        tgs = {p.stem: p for p in (base / "textgrid").glob("*.TextGrid")}
        anns = {p.stem: p for p in (base / "annotation").glob("*.TextGrid")}
        for wav in sorted((base / "wav").glob("*.wav")):
            if wav.stem not in tgs:
                continue
            canon = canonical_ipa_from_textgrid(tgs[wav.stem])
            if not canon:
                continue
            ann = anns.get(wav.stem)
            rows.append({
                "utt_key": f"{spk}/{wav.stem}", "speaker": spk, "L1": SPEAKER_L1[spk],
                "utt_id": wav.stem, "wav": f"{spk}/wav/{wav.name}",   # relative to the L2-ARCTIC root
                "test_sentence": is_test_sentence(wav.stem),
                "canonical": " ".join(canon), "label": to_label(canon),
                "perceived": " ".join(perceived_ipa_from_annotation(ann)) if ann else "",
            })
    df = pd.DataFrame(rows)
    test_spk = df.speaker.isin(TEST_SPEAKERS)
    df["split"] = "unused"                                  # test spk x train sentences
    df.loc[~test_spk & ~df.test_sentence, "split"] = "train"
    df.loc[test_spk & df.test_sentence, "split"] = "test"
    # validation: 50 utterances per L1 from train speakers x test sentences
    cand = df[~test_spk & df.test_sentence]
    val_idx = cand.groupby("L1", group_keys=False).apply(
        lambda g: g.sample(min(50, len(g)), random_state=0)).index
    df.loc[val_idx, "split"] = "val"
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2arctic", default="data/l2arctic_release_v5.0")
    ap.add_argument("--out", default="data/finetune_manifest.csv")
    a = ap.parse_args()
    df = build(Path(a.l2arctic))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "L1"]).size().unstack(fill_value=0))
    print("test sentences:", df[df.test_sentence].utt_id.nunique(),
          "train sentences:", df[~df.test_sentence].utt_id.nunique())
    leak = set(df[df.split == "train"].utt_id) & set(df[df.split == "test"].utt_id)
    print("sentence overlap train/test:", len(leak))
