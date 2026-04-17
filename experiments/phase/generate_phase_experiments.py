"""
Generates stereo WAV files for phase perception experiments.
Left channel = reference signal, Right channel = phase-manipulated signal.
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, sawtooth as scipy_sawtooth
import os

# Setup 
fs = 44100
duration = 5.0
t = np.arange(int(fs * duration)) / fs
OUT = "experiments/phase/results"
os.makedirs(OUT, exist_ok=True)

def norm(x):
    return (x / (np.max(np.abs(x)) + 1e-12)).astype(np.float32)

def save(name, left, right=None, note=""):
    """Save stereo (or mono) wav. Prints filename + note."""
    left = norm(left)
    if right is not None:
        right = norm(right)
        data = np.column_stack([left, right])
    else:
        data = left
    out16 = np.int16(data * 32767)
    path = os.path.join(OUT, name + ".wav")
    wavfile.write(path, fs, out16)
    print(f"  {name}.wav   {note}")


def fourier_sawtooth(t, freq, n_harmonics=40, amplitude=1.0):
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        wave += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * freq * t)
    return (2 * amplitude / np.pi) * wave


print("\n  1: Pure Sine — Phase-Blind Baseline ")
# A pure sine wave and its 90° phase shift (cosine) should sound IDENTICAL
freq = 440.0
sine = np.sin(2 * np.pi * freq * t)
cosine = np.cos(2 * np.pi * freq * t)  # 90° shift

save("01_sine_vs_cosine_90deg",
     sine, cosine,
     "L=sin(440), R=cos(440). Should sound IDENTICAL (phase blind baseline)")

save("01_sine_mono", sine, note="mono reference 440 Hz sine")


print("\n 2: Hilbert Quadrature — Stereo Rotation ")
# x in left ear, H[x] (Hilbert = 90° rotated) in right ear
# Same spectrum, different phase. 
x = sine.astype(np.float32)
H = np.imag(hilbert(x)).astype(np.float32)

save("02_hilbert_stereo_sine",
     x, H,
     "L=sine, R=Hilbert(sine). ")

# Now with a richer signal: sawtooth
saw = fourier_sawtooth(t, 220.0, n_harmonics=40)
H_saw = np.imag(hilbert(saw)).astype(np.float32)
save("03_hilbert_stereo_sawtooth",
     saw, H_saw,
     "L=sawtooth, R=Hilbert(sawtooth). Richer harmonics, is it more audible?")


print("\n EXPERIMENT 3: Fractional Hilbert Rotation (Static Angles) ")
# x_alpha = cos(alpha)*x + sin(alpha)*H[x]
# Rotate the analytic signal by a fixed angle for each file
angles_deg = [0, 45, 90, 135, 180]
for deg in angles_deg:
    alpha = np.radians(deg)
    x_alpha = np.cos(alpha) * x + np.sin(alpha) * H
    save(f"04_frac_hilbert_{deg:03d}deg",
         x, x_alpha,
         f"L=original sine, R=rotated by {deg}°")

# 180° is a polarity inversion: should be audible in stereo (comb filter / null)
save("04_frac_hilbert_180deg_STEREO_NULL",
     x, -x,
     "L=sine, R=-sine. 180° inversion. Should cancel if mixed to mono (anti-phase).")


print("\n EXPERIMENT 4: Slow Continuous Rotation (sweeping alpha) ")
# alpha sweeps 0→2π over the full clip: you hear the phase 'rotate'
alpha_sweep = np.linspace(0, 2 * np.pi, len(x))
z = x + 1j * H
x_rot = np.real(np.exp(1j * alpha_sweep) * z).astype(np.float32)
save("05_continuous_rotation_sine",
     x, x_rot,
     "L=original, R=continuously rotating phase (0->360deg).")

saw32 = saw.astype(np.float32)
H_saw32 = H_saw.astype(np.float32)
z_saw = saw32 + 1j * H_saw32
x_rot_saw = np.real(np.exp(1j * alpha_sweep) * z_saw).astype(np.float32)
save("05_continuous_rotation_sawtooth",
     saw32, x_rot_saw,
     "L=sawtooth, R=continuously rotating phase.")


print("\n EXPERIMENT 5: Polarity Inversion ")
# Direct amplitude inversion: a 180° phase flip
# Different from a time-delay phase shift
for sig, name, label in [
    (sine, "sine_440", "sine 440 Hz"),
    (saw, "sawtooth_220", "sawtooth 220 Hz"),
]:
    save(f"06_polarity_{name}",
         norm(sig), norm(-sig),
         f"L={label}, R=polarity inverted. Headphones vs speakers.")


print("\n EXPERIMENT 6: Time Delay vs Phase Shift ")
# A time delay shifts ALL frequencies by DIFFERENT phase amounts (phase = -2πf*delay)
# A phase shift shifts ALL frequencies by the SAME phase amount

delay_ms = 1.0  # 1 ms delay
delay_samples = int(delay_ms * 1e-3 * fs)

sine_delayed = np.zeros_like(sine)
sine_delayed[delay_samples:] = sine[:-delay_samples]
save("07_delay_vs_original_1ms",
     sine, sine_delayed,
     "L=sine, R=1ms delayed sine. Delay≠phase shift for broadband signals.")

# ITD (interaural time difference) — how we locate sounds spatially
# ~600µs = 90° azimuth (ear-to-ear distance ~21cm, sound travels ~343m/s)
itd_samples = int(0.0006 * fs)  # 600 µs
saw_itd = np.zeros_like(saw)
saw_itd[itd_samples:] = saw[:-itd_samples]
save("08_ITD_sawtooth_600us",
     saw, saw_itd,
     "L=sawtooth, R=600µs delayed (ITD ~90deg azimuth). Sound should appear to shift LEFT.")


print("\n EXPERIMENT 7: Harmonic Phase — Hearing the Shape ")
# Shift only the 2nd harmonic of a two-component signal
# Classic phase audibility test
w = 2 * np.pi * 220.0
wave_A = np.sin(w * t) + 0.5 * np.sin(2 * w * t)        # standard
wave_B = np.sin(w * t) + 0.5 * np.cos(2 * w * t)         # 2nd harmonic at 90°
wave_C = np.sin(w * t) - 0.5 * np.sin(2 * w * t)         # 2nd harmonic at 180°

save("09_harmonic_phase_A_vs_B",
     wave_A, wave_B,
     "L=sin(w)+0.5sin(2w), R=sin(w)+0.5cos(2w). 90deg on 2nd harmonic.")
save("09_harmonic_phase_A_vs_C",
     wave_A, wave_C,
     "L=sin(w)+0.5sin(2w), R=sin(w)-0.5sin(2w). 180deg on 2nd harmonic.")

# Sawtooth variant with harmonic phase shifts
saw_full = fourier_sawtooth(t, 220.0, n_harmonics=40)
def saw_shift_harmonic(t, freq, k, alpha, n_harmonics=40, amplitude=1.0):
    wave = np.zeros_like(t)
    for n in range(1, n_harmonics + 1):
        phase = alpha if n == k else 0.0
        wave += ((-1) ** (n + 1) / n) * np.sin(2 * np.pi * n * freq * t + phase)
    return (2 * amplitude / np.pi) * wave

for k, deg in [(2, 90), (2, 180), (3, 90)]:
    alpha = np.radians(deg)
    saw_shifted = saw_shift_harmonic(t, 220.0, k=k, alpha=alpha, n_harmonics=40)
    save(f"10_saw_h{k}_{deg:03d}deg_shift",
         saw_full, saw_shifted,
         f"Sawtooth: L=normal, R=harmonic {k} shifted {deg}°.")


print("\n EXPERIMENT 8: Forward vs Backward (Time Reversal) ")
# Reversing time changes phase spectrum but preserves magnitude
saw_fwd = norm(saw_full)
saw_bwd = norm(saw_full[::-1])
save("11_sawtooth_forward_vs_backward",
     saw_fwd, saw_bwd,
     "L=sawtooth forward, R=time-reversed. Phase spectrum flipped, magnitude same.")


print("\n EXPERIMENT 9: Minimum Phase vs Zero Phase Reconstruction ")
# Compare original signal to FFT→phase zeroed→IFFT (zero-phase, keeps magnitude only)
# This isolates what the phase spectrum contributes to the SOUND SHAPE

sig = saw_full
X = np.fft.fft(sig)
mag = np.abs(X)

# Zero-phase: keep magnitude, set all phases to 0
X_zerophase = mag * np.exp(1j * 0)
sig_zerophase = np.fft.ifft(X_zerophase).real.astype(np.float32)

save("12_original_vs_zerophase_saw",
     norm(sig), norm(sig_zerophase),
     "L=sawtooth, R=magnitude only (phase zeroed).")

# Same for a speech-like AM signal
carrier = 800.0
mod = np.sin(2 * np.pi * 3.0 * t)  # 3 Hz AM
am_sig = (1 + 0.8 * mod) * np.sin(2 * np.pi * carrier * t)
X_am = np.fft.fft(am_sig)
X_am_zp = np.abs(X_am) * np.exp(1j * 0)
am_zp = np.fft.ifft(X_am_zp).real.astype(np.float32)
save("12_am_vs_zerophase",
     norm(am_sig), norm(am_zp),
     "L=AM signal, R=magnitude only (phase zeroed).")


print("\n EXPERIMENT 10: Chorus / Phaser Simulation ")
# Chorus: slightly pitch-shifted + delayed copy mixed with original
# Creates 'thickening' — the beating between slightly detuned copies
detune_hz = 3.0   # slight detuning
chorus_t = t + 0.001 * np.sin(2 * np.pi * 0.5 * t)   # slow LFO modulates delay
chorus_sig = np.sin(2 * np.pi * freq * chorus_t).astype(np.float32)
chorus_mix = 0.5 * (sine + chorus_sig)
save("13_chorus_effect_sine",
     norm(sine), norm(chorus_mix),
     "L=dry sine, R=chorus mix (LFO-modulated delay). ")

# Phaser: all-pass filtered signal (phase shifted notches) mixed with dry
# Simplified phaser using Hilbert to create a comb
num_stages = 4
phaser_sig = sine.copy()
for _ in range(num_stages):
    phaser_sig = 0.5 * (phaser_sig + np.imag(hilbert(phaser_sig)))
phaser_sig = phaser_sig.astype(np.float32)
save("14_phaser_simulation_sine",
     norm(sine), norm(phaser_sig),
     "L=dry, R=phaser (cascaded Hilbert all-pass stages).")


print(f"\n✓ All files saved to: {OUT}")
print(f"  Total: {len(os.listdir(OUT))} files")
