"""Figure: per-speaker PER (all 24 held-out speakers, 4-fold speaker CV), zero-shot vs fine-tuned.
Usage: python src/plot_cv.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import L1_ORDER

sp = pd.read_csv("results/finetune/cv/by_speaker.csv")
fig, ax = plt.subplots(figsize=(8.5, 3.9))
style = {"M0 zero-shot": ("#9aa7b4", -0.18, "o"), "AR-w2v2 (fine-tuned)": ("#2f6db5", 0.18, "D")}
for m, (c, dx, mk) in style.items():
    d = sp[sp.model == m]
    for i, l1 in enumerate(L1_ORDER):
        y = d[d.L1 == l1].PER.to_numpy() * 100
        ax.scatter(np.full(len(y), i + dx), y, s=28, color=c, marker=mk, zorder=3,
                   label=m if i == 0 else None, edgecolor="white", linewidth=0.6)
        ax.hlines(y.mean(), i + dx - 0.13, i + dx + 0.13, color=c, lw=2.2, zorder=2)
ax.set_xticks(range(len(L1_ORDER)), L1_ORDER)
ax.set_ylim(0, 85)
ax.set_ylabel("PER vs intended phones (%)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
ax.set_title("All 24 speakers held out once (4-fold speaker CV, unseen sentences); bar = L1 mean",
             loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig("results/finetune/cv/figures/per_by_speaker_cv.png", dpi=200)
