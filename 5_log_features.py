# cache_features.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

SR = 48000
N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 20000

ROOT = Path("D:\\Downloads\\mixes")
FEAT_DIR = ROOT / "features_logmel"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

def wav_to_logmel(path):
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )
    logS = librosa.power_to_db(S + 1e-12, ref=1.0).astype(np.float32)
    return logS

def process_split(split_csv):
    df = pd.read_csv(ROOT / split_csv)
    out_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fpath = row["file_path"]
        feat_path = (FEAT_DIR / split_csv.replace(".csv","") / (Path(fpath).stem + ".npy"))
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        if not feat_path.exists():
            M = wav_to_logmel(fpath)
            np.save(feat_path, M)
        out_paths.append(str(feat_path))
    df["feature_path"] = out_paths
    df.to_csv(FEAT_DIR / split_csv, index=False)

def main():
    for split in ["train.csv","val.csv","test.csv"]:
        process_split(split)
    print("Features cached at:", FEAT_DIR)

if __name__ == "__main__":
    main()
