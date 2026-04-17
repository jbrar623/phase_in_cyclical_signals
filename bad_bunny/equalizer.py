import numpy as np
import librosa
import soundfile as sf

# EQ bands: (center_hz, gain_db, Q)
# Default values — gentle, clearly audible but not extreme
bands = [
    (200,   +4.0, 1.0),   # low bass boost
    (1000,  -3.0, 2.0),   # mid cut
    (4000,  +5.0, 1.5),   # presence boost
    (10000, +3.0, 1.0),   # air boost
]

FRAME_SIZE = 2048   # N
HOP_SIZE   = 512    # H


# load and normalize              
x, sr = librosa.load("bad_bunny/sounds/DtMF - Bad Bunny (128k).wav", sr=None, mono=True)
x_norm = x / (np.max(np.abs(x)) + 1e-12)

# STFT                        
D   = librosa.stft(x_norm, n_fft=2048, hop_length=512)
M   = np.abs(D)
phi = np.angle(D)

# frequency bin centres                
f_k = np.arange(2048 // 2 + 1) * sr / 2048

# EQ bands: (center_hz, gain_db, Q)              
bands = [
    (200,   +4.0, 1.0),
    (1000,  -3.0, 2.0),
    (4000,  +5.0, 1.5),
    (10000, +3.0, 1.0),
]

G_total = np.ones(len(f_k))
for f_c, gain_db, Q in bands:
    g     = 10 ** (gain_db / 20)
    sigma = f_c / Q
    G_total *= 1 + (g - 1) * np.exp(-0.5 * ((f_k - f_c) / sigma) ** 2)

#   apply gain, reconstruct          
D_hat = M * G_total[:, np.newaxis] * np.exp(1j * phi)
y     = librosa.istft(D_hat, hop_length=512, length=len(x))

sf.write("bad_bunny/sounds/output_bb_eq.wav", y / (np.max(np.abs(y)) + 1e-12), sr)