# make_splits.py
import json, random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path("D:\\Downloads\\mixes")
JSON_PATH = ROOT / "dataset.json"
OUT_DIR = ROOT
SEED = 42

def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # build per-file label presence vector
    ann = pd.DataFrame(data["annotations"])
    aud = pd.DataFrame(data["audios"])
    # presence flags per file
    piv = ann.pivot_table(index="audio_id", columns="category_id", values="id", aggfunc="count", fill_value=0)
    for cid in [1,2,3,4]:
        if cid not in piv.columns:
            piv[cid] = 0
    piv = piv[[1,2,3,4]].clip(upper=1)
    piv.columns = ["has_vessel","has_marine_animal","has_natural_sound","has_other_anthropogenic"]
    df = aud.merge(piv, left_on="id", right_index=True, how="left").fillna(0)
    # stratify using a hash of presence vector
    df["strat_key"] = (df["has_vessel"].astype(int).astype(str) +
                       df["has_marine_animal"].astype(int).astype(str) +
                       df["has_natural_sound"].astype(int).astype(str) +
                       df["has_other_anthropogenic"].astype(int).astype(str))
    X = df.index.values
    y = df["strat_key"].values
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, hold_idx = next(sss1.split(X, y))
    df_train = df.iloc[train_idx].copy()
    df_hold = df.iloc[hold_idx].copy()
    # split hold 50/50 into val/test
    Xh = df_hold.index.values
    yh = df_hold["strat_key"].values
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
    val_idx, test_idx = next(sss2.split(Xh, yh))
    df_val = df_hold.iloc[val_idx].copy()
    df_test = df_hold.iloc[test_idx].copy()
    # save
    df_train.to_csv(OUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUT_DIR / "test.csv", index=False)
    print("Saved splits to:", OUT_DIR)

if __name__ == "__main__":
    main()
