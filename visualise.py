# import numpy as np
# import matplotlib.pyplot as plt

# # Path to one of your saved .npy files
# npy_file = r"D:\Downloads\features\Deepship\Passengership\20171020a-125\201024.npy"

# # Load the spectrogram
# S_db = np.load(npy_file)

# # Plot it
# plt.figure(figsize=(10, 4))
# plt.imshow(S_db, aspect='auto', origin='lower', cmap='magma')  # or 'viridis'
# plt.colorbar(label="dB")
# plt.title("Log-Mel Spectrogram")
# plt.xlabel("Frames (time)")
# plt.ylabel("Mel bins (frequency)")
# plt.tight_layout()
# plt.show()


#TEST FOR A SINGLE FILE CODE BELOW 
#______________________________________________________________________________________________

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

# ==== CONFIG ====
IN_FILE = Path("D:\Downloads\mixes\mix_0001.wav")   # <-- change to your file
OUT_DIR = Path("D:\\Downloads\\features111")     # where to save .npy
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mel-spectrogram params
N_MELS = 128
N_FFT = 2048
HOP = 512
FMIN = 20
FMAX = 20000  # adjust if hydrophone bandwidth is lower

def compute_logmel(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32), sr

def main():
    if not IN_FILE.exists():
        print("File not found:", IN_FILE)
        return

    S_db, sr = compute_logmel(str(IN_FILE))

    # Save as numpy array
    out_npy = OUT_DIR / (IN_FILE.stem + ".npy")
    np.save(out_npy, S_db)
    print("Saved log-mel spectrogram as:", out_npy)

    # ==== Visualization ====
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP,
        x_axis="time", y_axis="mel", fmin=FMIN, fmax=FMAX
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Mel Spectrogram")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
