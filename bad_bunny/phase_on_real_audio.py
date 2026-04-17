"""
Left channel = original, Right channel = modified.
"""

import numpy as np
from scipy.io import wavfile
import os

AUDIO_FILE = "bad_bunny/sounds/DtMF - Bad Bunny (128k).wav"       # path to your WAV
MAX_SECONDS = 30.0            # trim to this length (keeps file sizes manageable)

os.makedirs("bad_bunny/sounds/output_wavs", exist_ok=True)

# load
sr, data = wavfile.read(AUDIO_FILE)
if data.ndim == 2:
    data = data.mean(axis=1)                     # stereo → mono
data = data.astype(np.float32)
data /= np.max(np.abs(data)) + 1e-12             # normalise to ±1
data = data[:int(MAX_SECONDS * sr)]              # trim
N = len(data)
print(f"Loaded: {AUDIO_FILE}  ({N/sr:.1f}s at {sr} Hz)")

# helpers 

def norm(x):
    return (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32) 

def save(name, left, right):
    stereo = np.column_stack([norm(left), norm(right)])
    wavfile.write(f"bad_bunny/sounds/output_wavs/{name}.wav", sr, np.int16(stereo * 32767))
    print(f"  saved: {name}.wav  (L=original, R=modified)")


# error handling for even/uneven match 
# audio is 25.3s × 44100 = 1,114,230 samples, even, 
# but N//2 - 1 gives 557,114 elements 
# while rp[N//2+1:] has 557,115 slots.

data = data[:int(MAX_SECONDS * sr)]
if len(data) % 2 != 0:
    data = data[:-1]
N = len(data)


#  FFT 
X     = np.fft.fft(data)
mag   = np.abs(X)
phase = np.angle(X)

#  experiments 

# 1. Round-trip: should be identical to original (check)
recon = np.fft.ifft(X).real.astype(np.float32)
save("1_roundtrip_identical", data, recon)

# 2. Zero phase: keep all magnitudes, set every phase to 0
#    Same frequencies, same loudness per frequency, but all timing destroyed
zero_phase = np.fft.ifft(mag).real.astype(np.float32)
save("2_zero_phase", data, zero_phase)

# 3. Random phase: keep magnitudes, randomise all phases
#    Sounds like noise even though every frequency is still there
rng = np.random.default_rng(seed=42)
rand = rng.uniform(-np.pi, np.pi, N // 2 - 1)
rp = np.zeros(N)
rp[1:N//2]   =  rand
rp[N//2+1:]  = -rand[::-1]           # conjugate symmetry for real output
random_phase = np.fft.ifft(mag * np.exp(1j * rp)).real.astype(np.float32)
save("3_random_phase", data, random_phase)

# 4. Time reversal: flips the entire phase spectrum, magnitude unchanged
#    Attack and decay swap, should be clearly audible difference
reversed_audio = data[::-1].copy()
save("4_time_reversed", data, reversed_audio)

# 5. Scramble only mid frequencies (500–4000 Hz): most musically destructive
freqs = np.fft.fftfreq(N, d=1/sr)
new_phase = phase.copy()
pos = np.where((freqs >= 500) & (freqs < 4000))[0]
neg = np.where((freqs <= -500) & (freqs > -4000))[0]
r = rng.uniform(-np.pi, np.pi, len(pos))
new_phase[pos] = r
new_phase[neg] = -r[::-1]
mid_scrambled = np.fft.ifft(mag * np.exp(1j * new_phase)).real.astype(np.float32)
save("5_mid_band_phase_scrambled", data, mid_scrambled)

print("\nDone. Open output_wavs/ and listen with headphones.")
print("L = original, R = modified in every file.")


# mono versions for presentation — just the modified signal
os.makedirs("bad_bunny/sounds/output_wavs/output_mono", exist_ok=True)

def save_mono_presentation(name, modified):
    wavfile.write(f"bad_bunny/sounds/output_wavs/output_mono/{name}.wav",
                  sr, np.int16(np.clip(norm(modified), -1, 1) * 32767))
    print(f"  saved mono: {name}.wav")

save_mono_presentation("2_zero_phase", zero_phase)
save_mono_presentation("3_random_phase", random_phase)
save_mono_presentation("4_time_reversed", reversed_audio)
save_mono_presentation("5_mid_band_scrambled", mid_scrambled)