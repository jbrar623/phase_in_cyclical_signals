import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert

AUDIO_IN  = "bad_bunny/sounds/DtMF - Bad Bunny (128k).wav"
AUDIO_OUT = "bad_bunny/sounds/hilbert"
os.makedirs(AUDIO_OUT, exist_ok=True)


def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.max(np.abs(x)) + 1e-12)


def save_wav(name, signal, sr):
    path = os.path.join(AUDIO_OUT, f"{name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(norm(signal), -1, 1) * 32767))
    print(f"saved: {path}")


# load: use one channel only (stereo → mono)
sr, data = wavfile.read(AUDIO_IN)
channel = data[:, 1].astype(np.float32)
channel = norm(channel)

# analytic signal via Hilbert transform
# z(t) = channel(t) + i·H[channel(t)] = A(t)·e^(iφ(t))
H        = hilbert(channel)
envelope = np.abs(H).astype(np.float32)       # A(t) — instantaneous amplitude
eps      = 1e-12
y        = (H / (envelope + eps)).astype(np.float32)   # H/|H| — phase only

# save all three
save_wav("H_real",   H.real,    sr)   # Hilbert transform: 90° phase shift
save_wav("abs_H",    envelope,  sr)   # amplitude envelope
save_wav("y_real",   y.real,    sr)   # phase-only signal (H/|H|)
