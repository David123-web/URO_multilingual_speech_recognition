"""Accent-robust phone recogniser: fine-tune the zero-shot IPA model on L2 English.

Model ("AR-w2v2", accent-robust wav2vec2):
  * backbone + CTC head initialised from ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns
    (its 314-symbol vocabulary already contains every English IPA character,
    so no layer is re-initialised)
  * CTC target = CANONICAL English phones (what the speaker meant), so accented
    realisations are mapped to the intended phone:  [sɪŋk] -> /θɪŋk/
  * auxiliary head: mean-pooled last hidden layer -> 6-way L1 classifier
    (the model also says WHICH accent it heard).  loss = CTC + 0.2 * CE(L1)
  * CNN feature encoder frozen; SpecAugment as in the base config.

Sub-commands (run from the project folder, inside .venv):
  python src/finetune.py smoke   --ram-gb 16          # 2 steps end-to-end, ~2 min
  python src/finetune.py train   --ram-gb 16 --max-hours 3
  python src/finetune.py predict --split test         # writes results/finetune/predictions_test.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # CTC loss falls back to CPU on MPS

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from common import L1_ORDER, loose, segment  # noqa: E402

BASE = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
L1_IDX = {l: i for i, l in enumerate(L1_ORDER)}
OUT = Path("results/finetune")
CKPT = Path("models/ar_w2v2")


def device():
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def log(msg, **kv):
    rec = {"t": time.strftime("%H:%M:%S"), "msg": msg, **kv}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train_log.jsonl", "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
class ARModel(torch.nn.Module):
    def __init__(self, w2v: Wav2Vec2ForCTC):
        super().__init__()
        self.w2v = w2v
        self.l1_head = torch.nn.Linear(w2v.config.hidden_size, len(L1_ORDER))

    def forward(self, input_values, attention_mask, labels=None):
        out = self.w2v(input_values, attention_mask=attention_mask, labels=labels,
                       output_hidden_states=True)
        h = out.hidden_states[-1]
        fm = self.w2v._get_feature_vector_attention_mask(h.shape[1], attention_mask).to(h.dtype)
        pooled = (h * fm.unsqueeze(-1)).sum(1) / fm.sum(1, keepdim=True).clamp(min=1)
        return out, self.l1_head(pooled)


def load_model(path: str | Path = BASE, dev="cpu"):
    proc = Wav2Vec2Processor.from_pretrained(str(path))
    w2v = Wav2Vec2ForCTC.from_pretrained(
        str(path), ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=proc.tokenizer.pad_token_id)
    m = ARModel(w2v)
    head = Path(path) / "l1_head.pt"
    if head.exists():
        m.l1_head.load_state_dict(torch.load(head, map_location="cpu"))
    return m.to(dev), proc


def save_model(m: ARModel, proc, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    m.w2v.save_pretrained(path)
    proc.save_pretrained(path)
    torch.save(m.l1_head.state_dict(), path / "l1_head.pt")


# ----------------------------------------------------------------------------
def load_manifest(args):
    df = pd.read_csv(args.manifest)
    root = Path(args.l2arctic)
    df["path"] = df.wav.map(lambda w: str(root / w))
    if "dur" not in df:                       # cache durations in the manifest
        df["dur"] = [sf.info(p).duration for p in df.path]
        df.drop(columns="path").to_csv(args.manifest, index=False)
    return df


def audio(path):
    a, _ = librosa.load(path, sr=16000)
    return a


def make_batches(df: pd.DataFrame, max_seconds: float, shuffle=True, seed=0):
    """Length-bucketed batches whose total (padded) audio <= max_seconds."""
    rng = random.Random(seed)
    idx = df.sort_values("dur").index.tolist()
    batches, cur, cur_max = [], [], 0.0
    for i in idx:
        d = df.at[i, "dur"]
        if cur and max(cur_max, d) * (len(cur) + 1) > max_seconds:
            batches.append(cur)
            cur, cur_max = [], 0.0
        cur.append(i)
        cur_max = max(cur_max, d)
    if cur:
        batches.append(cur)
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(df, ids, proc, dev, with_labels=True):
    wavs = [audio(df.at[i, "path"]) for i in ids]
    enc = proc.feature_extractor(wavs, sampling_rate=16000, padding=True, return_tensors="pt",
                                 return_attention_mask=True)
    batch = {"input_values": enc.input_values.to(dev),
             "attention_mask": enc.attention_mask.to(dev)}
    if with_labels:
        lab = proc.tokenizer([df.at[i, "label"] for i in ids], padding=True, return_tensors="pt")
        ids_t = lab.input_ids.masked_fill(lab.attention_mask.eq(0), -100)
        batch["labels"] = ids_t.to(dev)
    l1 = torch.tensor([L1_IDX[df.at[i, "L1"]] for i in ids], device=dev)
    return batch, l1


def check_labels(df, proc):
    unk = proc.tokenizer.unk_token_id
    bad = [l for l in df.label.head(2000) if unk in proc.tokenizer(l).input_ids]
    if bad:
        raise SystemExit(f"{len(bad)} labels contain unknown symbols, e.g. {bad[:3]}")
    log("labels OK: no unknown symbols")


@torch.no_grad()
def decode(m, proc, df, ids_list, dev):
    m.eval()
    preds, l1s, probs = [], [], []
    for ids in ids_list:
        batch, _ = collate(df, ids, proc, dev, with_labels=False)
        out, l1_logits = m(**batch)
        pred_ids = out.logits.argmax(-1)
        preds += proc.batch_decode(pred_ids)
        p = torch.softmax(l1_logits.float(), -1).cpu().numpy()
        probs += [json.dumps({L1_ORDER[k]: round(float(v), 4) for k, v in enumerate(r)}) for r in p]
        l1s += [L1_ORDER[k] for k in p.argmax(1)]
    m.train()
    return preds, l1s, probs


def per(refs, hyps):
    from run_pfer_experiment import align
    e = n = 0
    for r, h in zip(refs, hyps):
        rs = r.split()
        _, d = align(rs, [x for x in segment(h) if x.strip()])
        e += d
        n += len(rs)
    return e / max(n, 1)


def evaluate(m, proc, df, dev, max_seconds):
    batches = make_batches(df, max_seconds, shuffle=False)
    preds, l1s, _ = decode(m, proc, df, batches, dev)
    order = [i for b in batches for i in b]
    refs = [df.at[i, "canonical"] for i in order]
    acc = float(np.mean([l1s[k] == df.at[i, "L1"] for k, i in enumerate(order)]))
    return per(refs, preds), acc, preds[:2]


# ----------------------------------------------------------------------------
def config_for_ram(ram_gb: float):
    if ram_gb >= 32:
        return dict(batch_seconds=48.0, grad_ckpt=False, freeze_layers=0)
    if ram_gb >= 16:
        return dict(batch_seconds=24.0, grad_ckpt=True, freeze_layers=0)
    return dict(batch_seconds=12.0, grad_ckpt=True, freeze_layers=12)


def train(args, smoke=False):
    dev = device()
    cfg = config_for_ram(args.ram_gb)
    if smoke:
        cfg["batch_seconds"] = min(cfg["batch_seconds"], 12.0)
    log("config", device=dev, ram_gb=args.ram_gb, **cfg, lr=args.lr, max_hours=args.max_hours, smoke=smoke)
    torch.manual_seed(0)
    random.seed(0)
    df = load_manifest(args)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    if smoke:
        tr, va = tr.sample(8, random_state=0), va.sample(4, random_state=0)
    m, proc = load_model(BASE, dev)
    check_labels(df, proc)
    m.w2v.freeze_feature_encoder()
    if cfg["grad_ckpt"]:
        m.w2v.gradient_checkpointing_enable()
    if cfg["freeze_layers"]:
        for layer in m.w2v.wav2vec2.encoder.layers[:cfg["freeze_layers"]]:
            for p in layer.parameters():
                p.requires_grad = False
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    m.train()

    start = time.time()
    budget = args.max_hours * 3600
    plan = {"total": 10_000, "warmup": 300 if not smoke else 1}

    def lr_lambda(step):
        if step < plan["warmup"]:
            return (step + 1) / plan["warmup"]
        return max(0.02, 1 - (step - plan["warmup"]) / max(1, plan["total"] - plan["warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    base_per, base_acc, _ = evaluate(m, proc, va, dev, cfg["batch_seconds"])
    log("val before fine-tuning", val_PER=round(base_per, 4), val_L1_acc=round(base_acc, 4))
    best = math.inf
    step, epoch, done = 0, 0, False
    run_loss = run_ctc = run_ce = 0.0
    while not done:
        for ids in make_batches(tr, cfg["batch_seconds"], seed=epoch):
            batch, l1 = collate(tr, ids, proc, dev)
            out, l1_logits = m(**batch)
            ce = torch.nn.functional.cross_entropy(l1_logits, l1)
            loss = out.loss + args.l1_weight * ce
            if not torch.isfinite(loss):
                log("non-finite loss, batch skipped", step=step)
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run_loss += loss.item(); run_ctc += out.loss.item(); run_ce += ce.item()
            elapsed = time.time() - start
            if step == 20 and not smoke:     # fit the LR schedule to the time budget
                sec = elapsed / 20
                plan["total"] = int(min(args.max_epochs * len(make_batches(tr, cfg["batch_seconds"])),
                                        (budget - 20 * 60) / (sec * 1.08)))
                log("schedule fitted", sec_per_step=round(sec, 2), planned_steps=plan["total"],
                    steps_per_epoch=len(make_batches(tr, cfg["batch_seconds"])))
            if step % 50 == 0 or smoke:
                k = 1 if smoke else 50
                log("train", step=step, epoch=epoch, loss=round(run_loss / k, 4),
                    ctc=round(run_ctc / k, 4), l1_ce=round(run_ce / k, 4),
                    lr=f"{sched.get_last_lr()[0]:.2e}", elapsed_min=round(elapsed / 60, 1))
                run_loss = run_ctc = run_ce = 0.0
            last = step >= plan["total"] or elapsed > budget - 10 * 60 or (smoke and step >= 2)
            if step % args.eval_every == 0 or last:
                vper, vacc, ex = evaluate(m, proc, va, dev, cfg["batch_seconds"])
                log("val", step=step, val_PER=round(vper, 4), val_L1_acc=round(vacc, 4), example=ex[0])
                if vper < best:
                    best = vper
                    save_model(m, proc, CKPT if not smoke else Path("models/ar_w2v2_smoke"))
                    log("saved best", val_PER=round(vper, 4))
            if last:
                done = True
                break
        epoch += 1
    log("training finished", steps=step, best_val_PER=round(best, 4),
        minutes=round((time.time() - start) / 60, 1))


def predict(args):
    dev = device()
    path = Path(args.model)
    m, proc = load_model(path, dev)
    df = load_manifest(args)
    sub = df[df.split == args.split] if args.split != "smoke" else df[df.split == "test"].head(4)
    batches = make_batches(sub, 24.0, shuffle=False)
    preds, l1s, probs = decode(m, proc, sub, batches, dev)
    order = [i for b in batches for i in b]
    out = sub.loc[order, ["utt_key", "speaker", "L1", "utt_id", "canonical", "perceived"]].copy()
    out = out.rename(columns={"canonical": "reference"})
    out["prediction_raw"] = preds
    out["L1_pred"] = l1s
    out["L1_probs"] = probs
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"predictions_{args.split}.csv"
    out.to_csv(f, index=False)
    log("predictions written", file=str(f), n=len(out),
        L1_acc=round(float((out.L1 == out.L1_pred).mean()), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "train", "predict"])
    ap.add_argument("--manifest", default="data/finetune_manifest.csv")
    ap.add_argument("--l2arctic", default="data/l2arctic_release_v5.0")
    ap.add_argument("--ram-gb", type=float, default=16)
    ap.add_argument("--max-hours", type=float, default=3.0)
    ap.add_argument("--max-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--l1-weight", type=float, default=0.2)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--model", default=str(CKPT))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    if a.cmd == "smoke":
        OUT = Path("results/finetune_smoke")
        train(a, smoke=True)
        a.model, a.split = "models/ar_w2v2_smoke", "smoke"
        predict(a)
        log("SMOKE TEST OK")
    elif a.cmd == "train":
        train(a)
    else:
        predict(a)
