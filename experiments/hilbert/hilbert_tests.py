

import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import hilbert

AUDIO_OUT = "experiments/hilbert/data/tests"
os.makedirs(AUDIO_OUT, exist_ok=True)

fs       = 44100
duration = 5.0
t        = np.arange(int(fs * duration)) / fs


def norm(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float32)
    return (x / (np.max(np.abs(x)) + eps)).astype(np.float32)


def save_wav(name, signal, sr=fs):
    path = os.path.join(AUDIO_OUT, f"{name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(norm(signal), -1, 1) * 32767))
    print(f"saved: {path}")


def save_stereo(name, left, right, sr=fs):
    stereo = np.column_stack([norm(left), norm(right)])
    path = os.path.join(AUDIO_OUT, f"{name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(stereo, -1, 1) * 32767))
    print(f"saved: {path}")


def fractional_hilbert(x, H, alpha):
    """
    Rotate the analytic signal by angle alpha.
    x_alpha = cos(α)·x + sin(α)·H[x]
    At α=0: original. At α=π/2: Hilbert. At α=π: polarity flip.
    """
    return (np.cos(alpha) * x + np.sin(alpha) * H).astype(np.float32)


def play_static_rotations(x, H, angles=None, clip_seconds=0.75,
                           gap_seconds=0.15, stereo=False):
    """
    Play a sequence of static fractional Hilbert rotations.
    Each angle produces a short clip for direct comparison.
    stereo=True puts x_α in left ear and x_(α+π/2) in right — more audible.
    """
    if angles is None:
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)  # 0,45,90...315°

    n_clip = int(fs * clip_seconds)
    n_gap  = int(fs * gap_seconds)

    for a in angles:
        x_a = norm(fractional_hilbert(x, H, a))
        if stereo:
            x_b = norm(fractional_hilbert(x, H, a + np.pi / 2))
            out = np.column_stack((x_a[:n_clip], x_b[:n_clip])).astype(np.float32)
        else:
            out = x_a[:n_clip]
        sd.play(out, samplerate=fs)
        sd.wait()
        if n_gap > 0:
            silence = np.zeros((n_gap, 2) if stereo else n_gap, dtype=np.float32)
            sd.play(silence, samplerate=fs)
            sd.wait()


# build signals 

freq = 440.0
x    = np.sin(2 * np.pi * freq * t).astype(np.float32)
H    = np.imag(hilbert(x)).astype(np.float32)

# analytic signal and continuous rotation
z         = x + 1j * H
alpha_sweep = np.linspace(0, 4 * np.pi, len(x))   # two full rotations
x_rot     = np.real(np.exp(1j * alpha_sweep) * z).astype(np.float32)

# save outputs 

save_stereo("sine_vs_hilbert",     x,     H)        # should sound identical
save_stereo("sine_vs_continuous_rotation", x, x_rot) # sweeping phase
save_wav("hilbert_quadrature",     H)               # H[x] alone

for deg in [0, 45, 90, 135, 180]:
    alpha = np.radians(deg)
    save_stereo(f"rotation_{deg:03d}deg", x, fractional_hilbert(x, H, alpha))

# listening (uncomment to play)

# play_static_rotations(x, H, clip_seconds=0.75, gap_seconds=0.15, stereo=False)
# play_static_rotations(x, H, clip_seconds=0.8,  gap_seconds=0.2,  stereo=True)