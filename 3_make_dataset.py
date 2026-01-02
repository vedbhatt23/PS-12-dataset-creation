import os, json, math, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import soundfile as sf
import librosa

# ---------------- CONFIG ----------------SHA256:byga2czKn8ZtAJs53IpjHdlFLnqUdbhBG2YtsX8grDk
N_MIXES = 1000
SEED = 777
SR = 48000
BIT_DEPTH = "PCM_16"
TARGET_PEAK = 0.98

ROOT_STD = Path("D:\\Downloads\\standardized")
CLASS_DIRS = {
    1: ROOT_STD / "vessel",
    2: ROOT_STD / "marine_animal",
    3: ROOT_STD / "natural_sound",
    4: ROOT_STD / "other_anthropogenic",
}
AMBIENCE_DIR = Path("D:\\Downloads\\ambience")

# Output
OUT_DIR = Path("D:\\Downloads\\mixes")
OUT_DIR.mkdir(parents=True, exist_ok=True)
APP_JSON = OUT_DIR / "dataset.json"

MIX_DURATION_S = 60.0
MIN_GAP_S = 2.0
MAX_GAP_S = 10.0
EVENTS_PER_CLASS = 1

# Stronger SNR ranges (dB) for audibility
SNR_DB = {
    1: (14, 20),  # vessel
    2: (12, 20),  # marine_animal
    3: (12, 20),  # natural_sound
    4: (12, 20),  # other_anthropogenic
}

# Minimum RMS floors
FG_MIN_RMS = 5e-3    # raise foreground floor
AMB_MIN_RMS = 2e-3   # require audible ambience slice

ROUND_TO_SEC = True
# ----------------------------------------

random.seed(SEED)
np.random.seed(SEED)

def list_wavs(folder: Path) -> List[Path]:
    return sorted(folder.rglob("*.wav"))

def load_wav(path: Path, sr: int) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)

def ensure_len(y: np.ndarray, sr: int, min_s=1.0, max_s=5.0) -> np.ndarray:
    min_len = int(min_s * sr)
    max_len = int(max_s * sr)
    if len(y) < min_len:
        y = np.pad(y, (0, min_len - len(y)))
    if len(y) > max_len:
        start = random.randint(0, len(y) - max_len)
        y = y[start:start+max_len]
    return y.astype(np.float32)

def peak_limit(x: np.ndarray, peak=0.98) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-9)
    return x if m <= peak else (x / m * peak).astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def scale_to_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    cur = rms(x)
    if cur < 1e-12:
        # inject tiny noise to avoid hard zero
        n = np.random.normal(0, 1e-6, size=x.shape).astype(np.float32)
        x = x + n
        cur = rms(x)
    return (x * (target_rms / cur)).astype(np.float32)

def choose_ambience_window(amb: np.ndarray, need: int, tries=20) -> Tuple[int, np.ndarray]:
    for _ in range(tries):
        if len(amb) <= need:
            start = 0
        else:
            start = random.randint(0, len(amb) - need)
        sl = amb[start:start+need]
        if rms(sl) >= AMB_MIN_RMS:
            return start, sl
    # fallback: boost slice to reach AMB_MIN_RMS
    start = 0
    sl = amb[:need].copy()
    if rms(sl) < AMB_MIN_RMS:
        sl = scale_to_rms(sl, AMB_MIN_RMS)
    return start, sl

def schedule_non_overlapping(durations: List[float], total: float, min_gap: float, max_gap: float, max_tries=1000):
    for _ in range(max_tries):
        t = 0.0
        times = []
        ok = True
        for d in durations:
            gap = random.uniform(min_gap, max_gap)
            start = t + gap
            end = start + d
            if end > total:
                ok = False
                break
            times.append((start, end))
            t = end
        if ok:
            return times
    return None

def round_and_fix_non_overlap(events: List[Dict]):
    ev = []
    for a in events:
        s = int(round(a["start_time"]))
        e = int(round(a["end_time"]))
        if e <= s:
            continue
        ev.append({"s": float(s), "e": float(e), "category_id": a["category_id"], "score": a["score"]})
    ev.sort(key=lambda x: x["s"])
    fixed = []
    last_e = -1.0
    for a in ev:
        s, e = a["s"], a["e"]
        if s < last_e:
            s = last_e
            if e <= s:
                e = s + 1.0
        fixed.append({"start_time": s, "end_time": e, "duration": e - s,
                      "category_id": a["category_id"], "score": a["score"]})
        last_e = e
    return fixed

def cycle_paths(paths: List[Path]):
    i = 0
    n = len(paths)
    while True:
        yield paths[i % n]
        i += 1

def synthesize_one(idx: int, class_iters: Dict[int, any], amb_wavs: List[Path]) -> Tuple[str, float, List[Dict]]:
    # ambience assembly
    amb_src = load_wav(random.choice(amb_wavs), SR)
    need = int(MIX_DURATION_S * SR)
    if len(amb_src) < need:
        reps = int(math.ceil(need / max(1, len(amb_src))))
        amb_src = np.tile(amb_src, reps)
    amb_src = amb_src[:need].astype(np.float32)
    # normalize ambience to a baseline RMS so SNR behaves
    amb_target_rms = max(AMB_MIN_RMS, 0.01)  # ~ -40 dBFS region
    amb_src = scale_to_rms(amb_src, amb_target_rms)

    # choose one fg per class
    fgs = []
    info = []
    for cid in [1,2,3,4]:
        src = next(class_iters[cid])
        y = load_wav(src, SR)
        y = ensure_len(y, SR, 1.0, 5.0)
        # impose a strong foreground RMS baseline before SNR scaling
        y = scale_to_rms(y, FG_MIN_RMS)
        fgs.append(y)
        info.append({"cid": cid, "src": src, "dur": len(y)/SR})

    # schedule times
    times = schedule_non_overlapping([m["dur"] for m in info], MIX_DURATION_S, MIN_GAP_S, MAX_GAP_S)
    if times is None:
        # evenly spread fallback
        times = []
        t = 0.0
        for m in info:
            t += MIN_GAP_S
            times.append((t, min(MIX_DURATION_S, t + m["dur"])))
            t = times[-1][1]

    mix = amb_src.copy()
    ann = []
    for (y, m), (s, e) in zip(zip(fgs, info), times):
        start_i = int(round(s * SR))
        end_i = start_i + len(y)
        if end_i > len(mix):
            end_i = len(mix)
            start_i = end_i - len(y)
            if start_i < 0:
                y = y[:end_i]
                start_i = 0
        # ensure bg slice is non-silent
        bg = mix[start_i:end_i].copy()
        if rms(bg) < AMB_MIN_RMS:
            # patch a louder window from ambience
            _, repl = choose_ambience_window(amb_src, len(bg))
            mix[start_i:end_i] = repl
            bg = mix[start_i:end_i].copy()
        # SNR scaling
        lo, hi = SNR_DB[m["cid"]]
        snr = random.uniform(lo, hi)
        bg_r = rms(bg)
        target_fg_r = max(bg_r * (10.0 ** (snr / 20.0)), FG_MIN_RMS)
        y_scaled = scale_to_rms(y, target_fg_r)
        mixed = bg + y_scaled
        mix[start_i:end_i] = peak_limit(mixed, TARGET_PEAK)
        ann.append({
            "category_id": m["cid"],
            "start_time": float(s),
            "end_time": float(s + len(y)/SR),
            "score": 0.99
        })

    mix = peak_limit(mix, TARGET_PEAK)

    if ROUND_TO_SEC:
        ann = round_and_fix_non_overlap(ann)
    else:
        for a in ann:
            a["duration"] = float(a["end_time"] - a["start_time"])

    fname = f"mix_{idx:04d}.wav"
    sf.write(str(OUT_DIR / fname), mix, SR, subtype=BIT_DEPTH)
    return fname, MIX_DURATION_S, ann

def build_json(audios_meta, annotations_all):
    data = {
        "info": {"description": "Grand Challenge UDA", "version": "1.0", "year": 2025},
        "audios": [],
        "categories": [
            {"id": 1, "name": "vessel"},
            {"id": 2, "name": "marine_animal"},
            {"id": 3, "name": "natural_sound"},
            {"id": 4, "name": "other_anthropogenic"}
        ],
        "annotations": []
    }
    for i,(fname, dur) in enumerate(audios_meta, start=1):
        data["audios"].append({
            "id": i,
            "file_name": fname,
            "file_path": str((OUT_DIR / fname)).replace("\\","/"),
            "duration": float(dur)
        })
    ann_id = 1
    for i,anns in enumerate(annotations_all, start=1):
        for a in anns:
            data["annotations"].append({
                "id": ann_id,
                "audio_id": i,
                "category_id": int(a["category_id"]),
                "start_time": float(a["start_time"]),
                "end_time": float(a["end_time"]),
                "duration": float(a["duration"]),
                "score": float(a["score"])
            })
            ann_id += 1
    return data

def main():
    class_files = {cid: list_wavs(p) for cid,p in CLASS_DIRS.items()}
    for cid,files in class_files.items():
        if not files:
            raise RuntimeError(f"No WAVs for class {cid} in {CLASS_DIRS[cid]}")
    amb_files = list_wavs(AMBIENCE_DIR)
    if not amb_files:
        raise RuntimeError(f"No ambience WAVs in {AMBIENCE_DIR}")

    class_iters = {cid: cycle_paths(class_files[cid]) for cid in class_files}

    aud_meta, all_anns = [], []
    for i in range(1, N_MIXES+1):
        fname, dur, anns = synthesize_one(i, class_iters, amb_files)
        # enforce all 4 classes present
        present = {a["category_id"] for a in anns}
        tries = 0
        while present != {1,2,3,4} and tries < 5:
            fname, dur, anns = synthesize_one(i, class_iters, amb_files)
            present = {a["category_id"] for a in anns}
            tries += 1
        aud_meta.append((fname, dur))
        all_anns.append(anns)
        if i % 25 == 0:
            print(f"Generated {i}/{N_MIXES}")

    data = build_json(aud_meta, all_anns)
    with open(APP_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print("Wrote:", APP_JSON)

if __name__ == "__main__":
    main()
