import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa

ROOT_STD = Path("D:\\Downloads\\standardized")
ROOT_FEAT = Path("D:\\Downloads\\features")
ROOT_FEAT.mkdir(parents=True, exist_ok=True)
CLASS_FOLDERS = ["Deepship", "watkins","dosits"]

N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 20000  # adjust if hydrophone bandwidth is lower

rows = []

def compute_logmel(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32), sr

def main():
    for cls in CLASS_FOLDERS:
        for wav in sorted((ROOT_STD/cls).rglob("*.wav")):
            S_db, sr = compute_logmel(str(wav))
            rel = wav.relative_to(ROOT_STD).with_suffix(".npy")
            out_npy = ROOT_FEAT / rel
            out_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_npy, S_db)
            rows.append({"class_name": cls,
                         "wav_path": str(wav),
                         "sr": sr,
                         "logmel_path": str(out_npy),
                         "frames": int(S_db.shape[1])})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT_FEAT/"logmel_index.csv", index=False)
    print("Saved:", ROOT_FEAT/"logmel_index.csv")

if __name__ == "__main__":
    main()
