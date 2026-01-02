# 6_train_crnn_memsafe_fixed.py
import os, math, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path("D:\\Downloads\\mixes")
FEAT_ROOT = ROOT / "features_logmel"
N_CLASSES = 4
LR = 1e-3
EPOCHS = 12
BATCH = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lightweight spectrogram params
SR = 48000
N_MELS = 64
N_FFT = 2048
HOP = 1024    # ~47 fps
FMIN = 20
FMAX = 20000
MAX_SEC = 30.0 # optional training crop

def ensure_feats_cached(split_csv):
    import librosa
    FEAT_DIR = FEAT_ROOT / split_csv.replace(".csv","")
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / split_csv)
    outs = []
    for _,row in tqdm(df.iterrows(), total=len(df), desc=f"cache {split_csv}"):
        wav = row["file_path"]
        outp = FEAT_DIR / (Path(wav).stem + ".npy")
        if not outp.exists():
            y, _ = librosa.load(wav, sr=SR, mono=True)
            if len(y) > int(MAX_SEC*SR):
                s = (len(y) - int(MAX_SEC*SR))//2
                y = y[s:s+int(MAX_SEC*SR)]
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                               n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
            logS = librosa.power_to_db(S + 1e-12, ref=1.0).astype(np.float32)
            np.save(outp, logS)
        outs.append(str(outp))
    df["feature_path"] = outs
    (FEAT_ROOT / split_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEAT_ROOT / split_csv, index=False)

class MelSet(Dataset):
    def __init__(self, split_csv):
        self.df = pd.read_csv(FEAT_ROOT / split_csv)
        self.targets = self.df[["has_vessel","has_marine_animal","has_natural_sound","has_other_anthropogenic"]].values.astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        M = np.load(self.df.iloc[i]["feature_path"])  # [M,T]
        x = torch.from_numpy(M).unsqueeze(0)          # [1,M,T]
        y = torch.from_numpy(self.targets[i])
        return x, y

class CRNN(nn.Module):
    def __init__(self, n_mels=N_MELS, n_classes=N_CLASSES):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 24, (3,3), padding=1), nn.ReLU(), nn.GroupNorm(6,24), nn.MaxPool2d((2,2)),
            nn.Conv2d(24, 48, (3,3), padding=1), nn.ReLU(), nn.GroupNorm(6,48), nn.MaxPool2d((2,2)),
        )
        # # infer GRU input_size dynamically
        # with torch.no_grad():
        #     dummy = torch.zeros(1,1,n_mels, 300)  # 300 frames dummy
        #     z = self.cnn(dummy)                   # [1,C,M/4,T/4]
        #     _, C, M_red, T_red = z.shape
        #     self._feat_dim = C * M_red            # per time-step feature size
        # self.rnn_hidden = 96
        # self.rnn = nn.GRU(input_size=self._feat_dim, hidden_size=self.rnn_hidden,
        #                   num_layers=1, batch_first=True, bidirectional=True)
        # self.head = nn.Linear(self.rnn_hidden*2, n_classes)
        self.rnn_hidden = 96
        self.n_classes = n_classes
        self.rnn = None
        self.head = None

    # def forward(self, x):
    #     z = self.cnn(x)                        # [B,C,M/4,T/4]
    #     B,C,M,T = z.shape
    #     z = z.permute(0,3,1,2).contiguous().view(B, T, -1)  # [B,T,C*M]
    #     print("GRU input shape:", z.shape, "Expected input_size:", self._feat_dim)
    #     if z.size(-1) != self._feat_dim:
    #         raise RuntimeError(f"GRU input.size(-1) must be equal to input_size. Expected {self._feat_dim}, got {z.size(-1)}")
    #     z,_ = self.rnn(z)
    #     #z,_ = self.rnn(z)                      # [B,T,2H]
    #     z = z.mean(dim=1)                      # [B,2H]
    #     return self.head(z)

    
    def forward(self, x):
        z = self.cnn(x)                        # [B,C,M/4,T/4]
        B,C,M,T = z.shape
        z = z.permute(0,3,1,2).contiguous().view(B, T, -1)  # [B,T,C*M]
        if self.rnn is None:
            self._feat_dim = z.size(-1)
            self.rnn = nn.GRU(input_size=self._feat_dim, hidden_size=self.rnn_hidden,
                              num_layers=1, batch_first=True, bidirectional=True).to(z.device)
            self.head = nn.Linear(self.rnn_hidden*2, self.n_classes).to(z.device)
        z,_ = self.rnn(z)                      # [B,T,2H]
        z = z.mean(dim=1)                      # [B,2H]
        return self.head(z)
    
def run():
    for s in ["train.csv","val.csv"]:
        ensure_feats_cached(s)
    train_ds = MelSet("train.csv"); val_ds = MelSet("val.csv")
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = CRNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    best = 1e9; OUT = ROOT / "models"; OUT.mkdir(exist_ok=True)

    for ep in range(1, EPOCHS+1):
        model.train(); tr=0
        for x,y in tqdm(train_dl, desc=f"train {ep}"):
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr += float(loss.item())*x.size(0)
        tr /= len(train_ds)
        model.eval(); vl=0
        with torch.no_grad():
            for x,y in tqdm(val_dl, desc=f"val {ep}"):
                x,y = x.to(DEVICE), y.to(DEVICE)
                vl += float(crit(model(x), y).item())*x.size(0)
        vl /= len(val_ds)
        print({"epoch":ep,"train_loss":tr,"val_loss":vl})
        if vl < best:
            best = vl
            torch.save(model.state_dict(), OUT / "crnn_clip_memsafe_fixed.pt")
    print("Saved:", OUT / "crnn_clip_memsafe_fixed.pt")

if __name__ == "__main__":
    run()


