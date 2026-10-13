"""Self-checks that run without the model or the corpus.

1. TextGrid parsing + stress-aware ARPAbet mapping + id-based file pairing on a
   synthetic L2-ARCTIC folder (one speaker deliberately missing a transcript).
2. End-to-end `analyze` on a synthetic legacy outputs.csv in which errors are
   injected ONLY on phones missing from each L1: the hypothesis test must
   detect the effect, and a control with no injected effect must not.
"""
import json, random, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import canonical_ipa_from_textgrid, segment, SPEAKER_L1  # noqa
from run_pfer_experiment import iter_utterances, align  # noqa

TG = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "the cat"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 7
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "DH"
        intervals [3]:
            xmin = 0.2
            xmax = 0.3
            text = "AH0"
        intervals [4]:
            xmin = 0.3
            xmax = 0.4
            text = "K"
        intervals [5]:
            xmin = 0.4
            xmax = 0.5
            text = "AE1"
        intervals [6]:
            xmin = 0.5
            xmax = 0.6
            text = "T"
        intervals [7]:
            xmin = 0.6
            xmax = 1.0
            text = ""
'''


def test_parsing_and_pairing():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for spk in ["ABA", "BWC"]:
            base = root / spk
            for sub in ["wav", "textgrid"]:
                (base / sub).mkdir(parents=True)
            for u in ["arctic_a0001", "arctic_a0002", "arctic_a0003"]:
                (base / "wav" / f"{u}.wav").write_bytes(b"")
                if not (spk == "BWC" and u == "arctic_a0002"):
                    (base / "textgrid" / f"{u}.TextGrid").write_text(TG)
        items = list(iter_utterances(root))
        assert len(items) == 5, items
        assert all(w.stem == t.stem for _, _, w, t, _ in items), "audio/transcript misaligned"
        ipa = canonical_ipa_from_textgrid(items[0][3])
        assert ipa == ["ð", "ə", "k", "æ", "t"], ipa
    assert segment("t͡ʃeɪnd͡ʒ") == ["tʃ", "eɪ", "n", "dʒ"]
    assert segment("ɡiːtʰ") == ["ɡ", "iː", "tʰ"]
    path, ed = align(["k", "æ", "t"], ["kʰ", "a", "t"])
    assert ed == 1, (path, ed)
    print("parsing/pairing/alignment OK")


def make_legacy_csv(path, effect: bool, seed=0):
    rng = random.Random(seed)
    inv = pd.read_csv(ROOT / "results" / "inventory_phone_by_L1.csv", index_col=0)
    phones = [p for p in inv.index if len(p) == 1]
    rows = []
    for l1 in ["Arabic", "Mandarin", "Hindi", "Korean", "Spanish", "Vietnamese"]:
        missing = set(inv.index[inv[l1].isin(["absent", "allophone", "parts"])])
        for _ in range(400):
            ref = [rng.choice(phones) for _ in range(30)]
            hyp = []
            for p in ref:
                p_err = 0.15 + (0.35 if (effect and p in missing) else 0)
                hyp.append(rng.choice(["a", "o", "e"]) if rng.random() < p_err else p)
            rows.append({"predictions": "".join(hyp), "transcripts": "".join(ref),
                         "orthgraphic transcript": "x", "ethnicity": l1, "pfer": 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)


def run_analyze(csv, out):
    r = subprocess.run([sys.executable, str(ROOT / "src" / "run_pfer_experiment.py"), "analyze",
                        "--predictions", str(csv), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads((out / "hypothesis_test.json").read_text())


def test_analyze_detects_injected_effect():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for effect in (True, False):
            out = d / f"out_{effect}"
            (out / "figures").mkdir(parents=True)
            (out / "inventory_phone_by_L1.csv").write_text(
                (ROOT / "results" / "inventory_phone_by_L1.csv").read_text())
            csv = d / f"outputs_{effect}.csv"
            make_legacy_csv(csv, effect)
            h = run_analyze(csv, out)
            diff, p = h["mean_within_phone_difference_in_error_rate"], h["permutation_p_value_two_sided"]
            print(f"injected effect={effect}: diff={diff:.3f}  p={p:.4f}")
            if effect:
                assert 0.25 < diff < 0.45 and p < 0.01
            else:
                assert abs(diff) < 0.05 and p > 0.05


if __name__ == "__main__":
    test_parsing_and_pairing()
    test_analyze_detects_injected_effect()
    print("ALL TESTS PASSED")
