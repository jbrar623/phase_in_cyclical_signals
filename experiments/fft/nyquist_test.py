import numpy as np
import os 

os.makedirs("experiments/fft/data/nyquist", exist_ok=True)

# Audio-rate settings
fs_truth = 44100.0      # Standard audio sample rate
T = 2.0                 # 2 seconds of audio
N_truth = int(T * fs_truth)
t_truth = np.arange(N_truth) / fs_truth

# Low sampling rate (still demonstrating aliasing)
fs_low = 8000.0         # Common for telephone quality
nyquist_low = fs_low / 2.0  # 4000 Hz

# Frequencies in the audible range
f_ok = 440.0            # A4 note - below Nyquist
f_alias = 5500.0        # Above Nyquist (4000 Hz), will alias
noise_std = 0.02

# Generate signal
x_truth = (
    np.sin(2*np.pi*f_ok*t_truth)
    + 0.6*np.sin(2*np.pi*f_alias*t_truth)
    + noise_std*np.random.randn(N_truth)
)

# Low-rate sampling
N_low = int(T * fs_low)
t_low = np.arange(N_low) / fs_low
idx = np.round(t_low * fs_truth).astype(int)
idx = np.clip(idx, 0, N_truth - 1)
x_low = x_truth[idx]

# FFT to IFFT on low nyquist data
X_low = np.fft.fft(x_low)
x_low_recon = np.fft.ifft(X_low).real

# Reconstruction error
err_ifft = x_low_recon - x_low
rmse_ifft = np.sqrt(np.mean(err_ifft**2))
max_ifft = np.max(np.abs(err_ifft))

# Aliased frequency
freqs = np.fft.fftfreq(N_low, d=1/fs_low)
mag = np.abs(X_low)
peak_idx = np.argmax(mag[1:]) + 1
peak_freq = abs(freqs[peak_idx])

# Save outputs with sample rate metadata
tag = "test4_nyquist"
np.savetxt(f"experiments/fft/data/nyquist/{tag}_truth_highrate.csv", x_truth, delimiter=",")
np.savetxt(f"experiments/fft/data/nyquist/{tag}_lowrate_samples.csv", x_low, delimiter=",")
np.savetxt(f"signal_comparison1/data/nyquist/{tag}_lowrate_ifft.csv", x_low_recon, delimiter=",")

# Save sample rates for the converter to use
np.savetxt(f"experiments/fft/data/nyquist/{tag}_sample_rates.csv", 
           [fs_truth, fs_low], 
           delimiter=",")

with open(f"experiments/fft/data/nyquist/{tag}_metrics.txt", "w") as f:
    f.write("Nyquist / Sampling Rate FFT to IFFT test\n\n")
    f.write(f"fs_truth = {fs_truth} Hz\n")
    f.write(f"fs_low = {fs_low} Hz\n")
    f.write(f"Nyquist frequency = {nyquist_low} Hz\n")
    f.write(f"f_ok = {f_ok} Hz\n")
    f.write(f"f_alias (true) = {f_alias} Hz\n\n")
    f.write("FFT peak frequency from low-rate samples:\n")
    f.write(f"Observed peak ≈ {peak_freq:.2f} Hz\n\n")
    f.write("FFT to IFFT reconstruction error on low-rate samples:\n")
    f.write(f"RMSE = {rmse_ifft:.6e}\n")
    f.write(f"MaxAbs = {max_ifft:.6e}\n")

print("fs_low:", fs_low, "Hz | Nyquist:", nyquist_low, "Hz")
print("True high frequency:", f_alias, "Hz")
print("Observed FFT peak (aliased):", peak_freq, "Hz")
print("IFFT RMSE:", rmse_ifft)