import json
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

ROOT_IN = Path("D:\Downloads")
ROOT_OUT = Path("D:\Downloads\standardized")
ROOT_OUT.mkdir(parents=True, exist_ok=True)
CLASS_FOLDERS = ["Deepship", "watkins","dosits"]
PROPOSAL = Path("reports/normalization_proposal.json")

# Defaults if proposal not found
TARGET_SR = 48000
TARGET_CH = 1
TARGET_SUBTYPE = "PCM_16"

if PROPOSAL.exists():
    p = json.loads(PROPOSAL.read_text())
    TARGET_SR = int(p.get("proposed_sampling_rate_hz", TARGET_SR))
    TARGET_CH = int(p.get("proposed_channels", TARGET_CH))
    # enforce 16-bit
    TARGET_SUBTYPE = "PCM_16"

def load_audio(path, target_sr):
    # librosa returns mono float32; use it, then re-channel later if needed
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y.astype(np.float32), target_sr

def ensure_channels(y, target_ch):
    if target_ch == 1:
        return y
    # duplicate mono to stereo if needed
    return np.stack([y, y], axis=0)  # shape (2, n)

def peak_limit(x, peak=0.99):
    m = np.max(np.abs(x)) + 1e-9
    if m > peak:
        x = x / m * peak
    return x

def write_wav(path, x, sr, subtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (n, ) for mono or (n, ch) for interleaved; we used channel-first for stereo above
    if x.ndim == 2:
        x = x.T  # (n, ch)
    sf.write(str(path), x.astype(np.float32), sr, subtype=subtype)

def main():
    count = 0
    for cls in CLASS_FOLDERS:
        in_dir = ROOT_IN / cls
        out_dir = ROOT_OUT / cls
        if not in_dir.exists():
            print(f"Skip missing: {in_dir}")
            continue
        for wav in sorted(in_dir.rglob("*.wav")):
            y, sr = load_audio(str(wav), TARGET_SR)
            y = ensure_channels(y, TARGET_CH)
            y = peak_limit(y, 0.99)
            rel = wav.relative_to(ROOT_IN)
            out_path = ROOT_OUT / rel
            write_wav(out_path, y, TARGET_SR, TARGET_SUBTYPE)
            count += 1
            if count % 50 == 0:
                print(f"Standardized {count} files...")
    print(f"Done. Standardized {count} files into {ROOT_OUT}")

if __name__ == "__main__":
    main()
