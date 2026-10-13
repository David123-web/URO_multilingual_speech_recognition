"""Check the IPA model downloads and runs on this Mac (no corpus needed).
Uses macOS `say` to synthesise a few ARCTIC sentences."""
import subprocess, sys, tempfile, time
from pathlib import Path
import librosa, torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
MODEL = "ctaguchi/wav2vec2-large-xlsr-japlmthufielta-ipa1000-ns"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", dev, "| torch", torch.__version__)
t = time.time()
proc = Wav2Vec2Processor.from_pretrained(MODEL)
model = Wav2Vec2ForCTC.from_pretrained(MODEL).to(dev).eval()
print(f"model loaded in {time.time()-t:.0f}s; vocab size {len(proc.tokenizer.get_vocab())}")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
(out / "model_vocab.json").write_text(__import__("json").dumps(sorted(proc.tokenizer.get_vocab()), ensure_ascii=False))
for s in ["Author of the danger trail, Philip Steels, etc.", "Typhoid, did I tell you?", "Will we ever forget it?"]:
    wav = Path(tempfile.mkdtemp()) / "x.wav"
    subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000", s], check=True)
    a, _ = librosa.load(wav, sr=16000)
    t = time.time()
    with torch.no_grad():
        ids = model(proc(a, sampling_rate=16000, return_tensors="pt").input_values.to(dev)).logits.argmax(-1)
    print(f"{s!r:50} -> {proc.batch_decode(ids)[0]}   ({time.time()-t:.2f}s)")
print("SMOKE TEST OK")
