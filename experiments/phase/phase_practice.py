import librosa
import sounddevice as sd
from scipy.signal import hilbert
import numpy as np

# Available example files:
# brahms, nutcracker, choice, fishin, humpback,
# libri1/2/3, vibeace, pistachio, robin, sweetwaltz
filename = librosa.ex('pistachio')

y, sr = librosa.load(filename, sr=None)  # preserve original sample rate
y = y.astype(np.float32)

# Compute quadrature component via Hilbert transform
analytic = hilbert(y)
h = np.imag(analytic).astype(np.float32)
h /= (np.max(np.abs(h)) + 1e-12)  # normalize to prevent clipping

sd.play(h, sr)
sd.wait()


# Triangle wave stereo experiment:
# t = np.linspace(0, 1, sr := 44100)
# triangle = sawtooth(2 * np.pi * 440 * t, width=0.5)  # width=0.5 → triangle
# stereo = np.column_stack([triangle, -triangle])       # mirrored L/R