import numpy as np
from scipy.signal import hilbert, sawtooth
from scipy.io import wavfile
import os


# Setup
fs = 44100.0
duration = 10.0
t = np.arange(int(fs * duration)) / fs

os.makedirs("experiments/hilbert/data/hilbert_analysis", exist_ok=True)

 
# EXAMPLE: Simple Sine Wave

freq = 440.0  # A4 note
sine_wave = np.sin(2 * np.pi * freq * t)

# Apply Hilbert transform
analytic_signal = hilbert(sine_wave)

# Extract amplitude and phase
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))

# For a pure sine wave, amplitude should be constant, phase should be linear
print(f"Sine wave frequency: {freq} Hz")
print(f"Mean amplitude envelope: {np.mean(amplitude_envelope):.4f} (should be ~1.0)")
print(f"Amplitude variation (std): {np.std(amplitude_envelope):.6f} (should be ~0)")
print()

# Instantaneous frequency (derivative of phase)
instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi) * fs
print(f"Mean instantaneous frequency: {np.mean(instantaneous_freq):.2f} Hz")
print(f"Expected: {freq} Hz")
print()


# EXAMPLE: changing amplitude, amplitude modulated signal 
# Carrier frequency and modulation frequency
carrier_freq = 1000.0  # 1 kHz carrier
mod_freq = 5.0  # 5 Hz modulation (slow enough to hear "wobbling")

# Create AM signal: (1 + m*cos(2πf_m*t)) * cos(2πf_c*t)
modulation_depth = 0.8
envelope = 1 + modulation_depth * np.cos(2 * np.pi * mod_freq * t)
am_signal = envelope * np.cos(2 * np.pi * carrier_freq * t)

# Extract envelope (instantaneous amplitude over time) using Hilbert transform
am_analytic = hilbert(am_signal)
extracted_envelope = np.abs(am_analytic)

print(f"Original envelope frequency: {mod_freq} Hz")
print(f"Carrier frequency: {carrier_freq} Hz")
print()

# Check how well we recovered the envelope
envelope_error = np.sqrt(np.mean((envelope - extracted_envelope)**2))
print(f"Envelope recovery RMSE: {envelope_error:.6f}")
#low error means successfully extracted envelope



# EXAMPLE: phase comparison 

# Create the two waves 
fund_freq = 220.0
w = 2 * np.pi * fund_freq

wave_A = np.sin(w * t) + 0.5 * np.sin(2 * w * t)
wave_B = np.sin(w * t) + 0.5 * np.cos(2 * w * t)  # 90° shift on 2nd harmonic

# Get analytic signals
analytic_A = hilbert(wave_A)
analytic_B = hilbert(wave_B)

# Extract phases
phase_A = np.unwrap(np.angle(analytic_A))
phase_B = np.unwrap(np.angle(analytic_B))

# Phase difference
phase_difference = phase_A - phase_B

print(f"Wave A: sin(wt) + 0.5*sin(2wt)")
print(f"Wave B: sin(wt) + 0.5*cos(2wt)")
print()
print(f"Mean phase difference: {np.mean(phase_difference):.4f} radians")
print(f"                     = {np.degrees(np.mean(phase_difference)):.2f} degrees")
print(f"Std of phase difference: {np.std(phase_difference):.4f}")



# Sawtooth Wave 

# Simple sawtooth from scipy
sawtooth_wave = sawtooth(2 * np.pi * 220 * t)

# Hilbert analysis
saw_analytic = hilbert(sawtooth_wave)
saw_phase = np.unwrap(np.angle(saw_analytic))
saw_inst_freq = np.diff(saw_phase) / (2 * np.pi) * fs

print(f"Sawtooth fundamental: 220 Hz")
print(f"Mean instantaneous frequency: {np.mean(saw_inst_freq):.2f} Hz")
print()
#Sawtooth has many harmonics, so instantaneous frequency varies more than a pure sine wave  


old = [1, 2,3,4,5,6]
new = hilbert(old)
print(new)
#print(new.imag) 
# imag_part = [ 2.30940108 -1.15470054 -1.15470054 -1.15470054 -1.15470054  2.30940108]



# Save WAV files for listening
def save_wav(filename, data, sample_rate=44100):
    normalized = data / np.max(np.abs(data))
    audio_data = np.int16(normalized * 32767)
    wavfile.write(filename, int(sample_rate), audio_data)

# Save AM signal and envelope
save_wav("experiments/hilbert/data/hilbert_analysis/am_signal.wav", am_signal, fs)
# Make envelope audible by using it to modulate a tone 
envelope_as_sound = extracted_envelope * np.sin(2 * np.pi * 440 * t)
save_wav("signal_comparison2/data/hilbert_analysis/extracted_envelope.wav", envelope_as_sound, fs)

# Save phase-shifted waves
save_wav("experiments/hilbert/data/hilbert_analysis/wave_A.wav", wave_A, fs)
save_wav("experiments/hilbert/data/hilbert_analysis/wave_B.wav", wave_B, fs)

