import numpy as np
from scipy.io import wavfile
import os


# AUDIO SETUP
fs = 44100  # Standard audio sample rate
duration = 10.0  # 10 seconds
t = np.arange(int(fs * duration)) / fs

# Create output directory
os.makedirs("experiments/hilbert/data/sawtooth_experiments", exist_ok=True)


# helpers
def fourier_sawtooth(t, freq, n_harmonics=50, amplitude=1.0):
    """
    Generate sawtooth wave using Fourier series

    Mathematical form: f(t) = (2A/π) * Σ((-1)^(n+1) * sin(n*w*t) / n)
    """
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        wave += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * freq * t)
    return (2 * amplitude / np.pi) * wave


def fourier_sawtooth_shifted(t, freq, phase_shift, n_harmonics=50, amplitude=1.0):
    """
    Uniform phase offset added to every harmonic term (NOT a pure time shift).
    """
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        wave += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * freq * t + phase_shift)
    return (2 * amplitude / np.pi) * wave


def fourier_sawtooth_one_harmonic_phase_shift(t, freq, k=2, alpha=np.pi / 2, n_harmonics=50, amplitude=1.0):
    """
    Apply a phase shift only to harmonic k (e.g., k=2 for second harmonic).
    """
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        phase = alpha if n == k else 0.0
        wave += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * freq * t + phase)
    return (2 * amplitude / np.pi) * wave


def sawtooth_direct(t, freq, amplitude=1.0):
    """
    Generate a sawtooth wave directly in time (not from Fourier series).
    Produces a ramp from -amplitude to +amplitude each period.
    """
    T = 1.0 / freq
    frac = (t % T) / T
    wave = 2 * frac - 1
    return amplitude * wave


def normalize_amplitude(wave):
    """Normalize wave so max abs value is 1.0"""
    return wave / np.max(np.abs(wave))


def save_wav(filename, data, sample_rate=44100):
    """Save normalized audio data as 16-bit WAV file"""
    normalized = data / np.max(np.abs(data))
    audio_data = np.int16(normalized * 32767)
    wavfile.write(filename, int(sample_rate), audio_data)
    print(f"  Saved: {filename}")



# Wave A vs Wave B
fundamental_freq = 220.0  # A3 note - easily hearable
w = 2 * np.pi * fundamental_freq

wave_A = np.sin(w * t) + 0.5 * np.sin(2 * w * t)
wave_B = np.sin(w * t) + 0.5 * np.cos(2 * w * t)

alpha = np.pi / 3  # 60 degree shift
wave_B_alpha = np.sin(w * t) + 0.5 * np.cos(2 * w * t + alpha)

print(f"Fundamental frequency: {fundamental_freq} Hz")
print(f"Wave A: sin(ωt) + 0.5*sin(2ωt)")
print(f"Wave B: sin(ωt) + 0.5*cos(2ωt)  [90° phase shift on 2nd harmonic]")
print(f"Wave B_alpha: sin(ωt) + 0.5*cos(2ωt + {alpha:.3f})  [{np.degrees(alpha):.1f}° shift]")
print()

wave_A_norm = normalize_amplitude(wave_A)
wave_B_norm = normalize_amplitude(wave_B)
wave_B_alpha_norm = normalize_amplitude(wave_B_alpha)

print(f"  Wave A max amplitude: {np.max(np.abs(wave_A_norm)):.6f}")
print(f"  Wave B max amplitude: {np.max(np.abs(wave_B_norm)):.6f}")
print(f"  Wave B_alpha max amplitude: {np.max(np.abs(wave_B_alpha_norm)):.6f}")
print()

save_wav("signal_comparison2/data/sawtooth_experiments/wave_A_reference.wav", wave_A_norm, fs)
save_wav("signal_comparison2/data/sawtooth_experiments/wave_B_90deg_shift.wav", wave_B_norm, fs)
save_wav("signal_comparison2/data/sawtooth_experiments/wave_B_alpha_shift.wav", wave_B_alpha_norm, fs)

# stereo A/B waves, better to listen 
save_wav(
    "signal_comparison2/data/sawtooth_experiments/stereo_A_left_B_right.wav",
    np.column_stack([wave_A_norm, wave_B_norm]),
    fs,
)

diff_A_B = wave_A_norm - wave_B_norm
save_wav("signal_comparison2/data/sawtooth_experiments/difference_A_minus_B.wav", diff_A_B, fs)
print()



# Sawtooth Phase Shift for one harmonic 
sawtooth_freq = 220.0
n_harm = 50

saw_fourier = fourier_sawtooth(t, sawtooth_freq, n_harmonics=n_harm)

k = 2
alpha_saw = np.pi / 2
saw_fourier_phase_shifted = fourier_sawtooth_one_harmonic_phase_shift(
    t, sawtooth_freq, k=k, alpha=alpha_saw, n_harmonics=n_harm
)

print(f"Sawtooth frequency: {sawtooth_freq} Hz")
print(f"Number of harmonics: {n_harm}")
print("Regular: Standard Fourier series sawtooth")
print(f"Phase-shifted: ONLY harmonic k={k} shifted by 90° (π/2)")
print()

saw_fourier_norm = normalize_amplitude(saw_fourier)
saw_fourier_phase_shifted_norm = normalize_amplitude(saw_fourier_phase_shifted)

save_wav("signal_comparison2/data/sawtooth_experiments/saw_fourier_regular.wav", saw_fourier_norm, fs)
save_wav(
    f"signal_comparison2/data/sawtooth_experiments/saw_fourier_h{k}_phaseShift_90deg.wav",
    saw_fourier_phase_shifted_norm,
    fs,
)

# stereo regular vs shifted harmonic
save_wav(
    f"signal_comparison2/data/sawtooth_experiments/stereo_saw_regular_left_h{k}shift_right.wav",
    np.column_stack([saw_fourier_norm, saw_fourier_phase_shifted_norm]),
    fs,
)
#first harmonic is same but second is shifted 

diff_phase = saw_fourier_norm - saw_fourier_phase_shifted_norm
save_wav(
    f"signal_comparison2/data/sawtooth_experiments/diff_saw_fourier_h{k}_phaseShift_90deg.wav",
    diff_phase,
    fs,
)
print()


# Sawtooth time reversal (Forward vs Backward)
saw_forward = saw_fourier.copy()
saw_backward = saw_fourier[::-1].copy()

print(f"Sawtooth frequency: {sawtooth_freq} Hz")
print("Forward: Normal time progression")
print("Backward: Time-reversed (samples flipped)")
print()

saw_forward_norm = normalize_amplitude(saw_forward)
saw_backward_norm = normalize_amplitude(saw_backward)

save_wav("signal_comparison2/data/sawtooth_experiments/saw_forward.wav", saw_forward_norm, fs)
save_wav("signal_comparison2/data/sawtooth_experiments/saw_backward.wav", saw_backward_norm, fs)

# stereo forward vs backward
save_wav(
    "signal_comparison2/data/sawtooth_experiments/stereo_saw_forward_left_backward_right.wav",
    np.column_stack([saw_forward_norm, saw_backward_norm]),
    fs,
)

diff_time_rev = saw_forward_norm - saw_backward_norm
save_wav("signal_comparison2/data/sawtooth_experiments/diff_time_reversal.wav", diff_time_rev, fs)
print()


# Amplitude Inversion (Up-ramp vs Down-ramp)
saw_direct = sawtooth_direct(t, sawtooth_freq, amplitude=1.0)
saw_inverted = -saw_direct

print(f"Sawtooth frequency: {sawtooth_freq} Hz")
print("Up-ramp: Linear rise from -1 to +1")
print("Down-ramp: Linear fall from +1 to -1 (amplitude inversion)")
print()

saw_direct_norm = normalize_amplitude(saw_direct)
saw_inverted_norm = normalize_amplitude(saw_inverted)

save_wav("signal_comparison2/data/sawtooth_experiments/saw_upramp.wav", saw_direct_norm, fs)
save_wav("signal_comparison2/data/sawtooth_experiments/saw_downramp.wav", saw_inverted_norm, fs)

#saw up ramp in right ear 
#saw down ramp in left ear 

diff_amplitude_inv = saw_direct_norm - saw_inverted_norm
save_wav("signal_comparison2/data/sawtooth_experiments/diff_amplitude_inversion.wav", diff_amplitude_inv, fs)
print()



# COMPARISON
rms_A_B = np.sqrt(np.mean((wave_A_norm - wave_B_norm) ** 2))
rms_phase = np.sqrt(np.mean((saw_fourier_norm - saw_fourier_phase_shifted_norm) ** 2))
rms_time_rev = np.sqrt(np.mean((saw_forward_norm - saw_backward_norm) ** 2))
rms_amplitude_inv = np.sqrt(np.mean((saw_direct_norm - saw_inverted_norm) ** 2))

print("RMS Differences:")
print(f"  Wave A vs B (phase shift on 2nd harmonic): {rms_A_B:.6f}")
print(f"  Sawtooth phase shift (harmonic {k} only): {rms_phase:.6f}")
print(f"  Time reversal (forward vs backward): {rms_time_rev:.6f}")
print(f"  Amplitude inversion (up vs down ramp - direct sawtooth inverted): {rms_amplitude_inv:.6f}")
print()

corr_A_B = np.corrcoef(wave_A_norm, wave_B_norm)[0, 1]
corr_phase = np.corrcoef(saw_fourier_norm, saw_fourier_phase_shifted_norm)[0, 1]
corr_time_rev = np.corrcoef(saw_forward_norm, saw_backward_norm)[0, 1]
corr_amplitude_inv = np.corrcoef(saw_direct_norm, saw_inverted_norm)[0, 1]

print("Correlations:")
print(f"  Wave A vs B: {corr_A_B:.6f}")
print(f"  Sawtooth phase shift (harmonic {k} only): {corr_phase:.6f}")
print(f"  Time reversal: {corr_time_rev:.6f}")
print(f"  Amplitude inversion: {corr_amplitude_inv:.6f}")
print("  (1.0 = identical, 0.0 = uncorrelated, -1.0 = inverted)")
print()


with open("signal_comparison2/data/sawtooth_experiments/analysis_metrics.txt", "w") as f:
    f.write("RESULTS\n\n")

    f.write("SETUP:\n")
    f.write(f"Sample rate: {fs} Hz\n")
    f.write(f"Duration: {duration} seconds\n")
    f.write(f"Fundamental frequency: {fundamental_freq} Hz\n")
    f.write(f"Sawtooth harmonics: {n_harm}\n\n")

    f.write("WAVE DEFINITIONS:\n")
    f.write("Wave A: sin(ωt) + 0.5*sin(2ωt)\n")
    f.write("Wave B: sin(ωt) + 0.5*cos(2ωt)  [90° phase shift on 2nd harmonic]\n\n")

    f.write("SAWTOOTH FOURIER SERIES:\n")
    f.write("f(t) = (2A/π) * Σ[(-1)^(n+1) * sin(n*ω*t) / n] for n=1 to 50\n\n")

    f.write("NUMERIC RESULTS\n\n")

    f.write("RMS DIFFERENCES:\n")
    f.write(f"Wave A vs B (phase shift on 2nd harmonic): {rms_A_B:.6f}\n")
    f.write(f"Sawtooth phase shift (harmonic {k} only): {rms_phase:.6f}\n")
    f.write(f"Time reversal (forward vs backward): {rms_time_rev:.6f}\n")
    f.write(f"Amplitude inversion (up vs down ramp): {rms_amplitude_inv:.6f}\n\n")

    f.write("CORRELATIONS:\n")
    f.write(f"Wave A vs B: {corr_A_B:.6f}\n")
    f.write(f"Sawtooth phase shift (harmonic {k} only): {corr_phase:.6f}\n")
    f.write(f"Time reversal: {corr_time_rev:.6f}\n")
    f.write(f"Amplitude inversion: {corr_amplitude_inv:.6f}\n")
    f.write("(1.0 = identical, 0.0 = uncorrelated, -1.0 = inverted)\n")
