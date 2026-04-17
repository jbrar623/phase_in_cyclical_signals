import os
import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import lfilter

AUDIO_IN  = "bad_bunny/sounds/DtMF - Bad Bunny (128k).wav"
AUDIO_OUT = "bad_bunny/sounds"
os.makedirs(AUDIO_OUT, exist_ok=True)


# helpers 

def norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.max(np.abs(x)) + 1e-12)


def load_audio(path, max_seconds=30):
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return norm(data[:int(max_seconds * sr)]), sr


def save_stereo(name, dry, wet, sr):
    """Save dry (left) and wet (right) as stereo WAV for headphone comparison."""
    stereo = np.column_stack([norm(dry), norm(wet)])
    path = os.path.join(AUDIO_OUT, f"{name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(stereo, -1, 1) * 32767))
    print(f"saved: {path}")


def save_mono(name, signal, sr):
    """Save a single processed signal as mono WAV."""
    path = os.path.join(AUDIO_OUT, f"{name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(norm(signal), -1, 1) * 32767))
    print(f"saved: {path}")


# simple_phaser 
# First-order IIR allpass filter, minimal implementation.
# Transfer function: H(z) = (z^-1 - a) / (1 - a·z^-1)
# Pole at z = a. Flat magnitude response: phase only.
# LFO sweeps the pole position a over time, moving the phase shift frequency.
# Mixing dry + processed creates notches where the two are ~180° out of phase.
#
# Simplest possible phaser: good for understanding the core idea.
# Sounds thinner than allpass_phaser() because first-order sections have
# a gentler phase slope and produce shallower notches.

def simple_phaser(x, sr, lfo_rate=0.5, depth=0.9, n_stages=4, wet_mix=0.5):
    n   = len(x)
    lfo = depth * np.sin(2 * np.pi * lfo_rate * np.arange(n) / sr)
    out = x.copy().astype(np.float64)
    for _ in range(n_stages):
        y = np.zeros_like(out)
        for i in range(1, n):
            a    = lfo[i]
            y[i] = -a * out[i] + out[i-1] + a * y[i-1]
        out = y
    return (wet_mix * x + (1 - wet_mix) * out).astype(np.float32)


# allpass_phaser 
# Biquad (second-order) allpass phaser: the full time-domain implementation.
# Same algorithm as Audacity, hardware phasers, guitar pedals.
# Biquad gives a steeper phase slope than first-order → deeper, more musical notches.
#
# Each stage is a second-order allpass biquad:
#   H(z) = (b0 + b1·z^-1 + b2·z^-2) / (1 + a1·z^-1 + a2·z^-2)
# where |H(ω)| = 1 for all ω — magnitude is completely flat.
# Phase shifts steeply around the center frequency f0.
#
# Processed in blocks of 64 samples — LFO updates filter coefficients
# per block rather than per sample (inaudible difference, much faster).

def _allpass_biquad_coeffs(f0, sr, q=0.707):
    w0    = 2 * np.pi * f0 / sr
    cos_w = np.cos(w0)
    sin_w = np.sin(w0)
    alpha = sin_w / (2 * q)
    b = np.array([1 - alpha, -2 * cos_w, 1 + alpha])
    a = np.array([1 + alpha, -2 * cos_w, 1 - alpha])
    return b / a[0], a / a[0]


def allpass_phaser(x, sr, n_stages=4, lfo_rate=0.4, lfo_depth=0.7,
                   center_hz=1200.0, width_hz=800.0, wet_mix=0.5, q=0.707):
    x       = x.astype(np.float64)
    N       = len(x)
    y_wet   = np.zeros(N)
    t       = np.arange(N) / sr
    lfo     = np.sin(2 * np.pi * lfo_rate * t)
    f_sweep = np.clip(center_hz + lfo_depth * width_hz * lfo, 20.0, sr / 2 - 1)
    states  = [np.zeros(2) for _ in range(n_stages)]
    block   = 64
    pos     = 0
    while pos < N:
        end   = min(pos + block, N)
        chunk = x[pos:end]
        f0    = float(f_sweep[min(pos + block // 2, N - 1)])
        out   = chunk.copy()
        for s in range(n_stages):
            b, a           = _allpass_biquad_coeffs(f0, sr, q=q)
            out, states[s] = lfilter(b, a, out, zi=states[s])
        y_wet[pos:end] = out
        pos = end
    return ((1 - wet_mix) * x + wet_mix * y_wet).astype(np.float32)


# fft_phaser 
# STFT-domain phaser: modifies phase directly per frequency bin.
# Phase is not a filter side effect here: it is explicitly computed and added.
# *** This is the best sounding implementation of the three ***
#
# Key step per frame:
#   lfo   = sin(2π · lfo_rate · t)           oscillates -1 to +1
#   delta = phase_depth · lfo · weight        Gaussian-weighted per bin
#   new_phase = original_phase + delta        direct phase addition
#
# The Gaussian weight (sigma from FWHM conversion: sigma = width / 2.355)
# concentrates the modification around center_hz, leaving other bins untouched.
#
# Saved as stereo (dry left, wet right) AND as a processed stereo WAV
# matching the original stereo format of the Bad Bunny track.
#
# This sits between the time-domain phaser and the phase vocoder:
#   - like the phaser:   LFO sweep creates audible notches
#   - like the vocoder:  operates directly on STFT phase values
#   - unlike the vocoder: phase is modified arbitrarily, not propagated coherently

def fft_phaser(x, sr, frame_size=2048, hop_size=512, lfo_rate=0.4,
               phase_depth=1.2, center_hz=1200, width_hz=1800):
    x      = norm(x)
    D      = librosa.stft(x, n_fft=frame_size, hop_length=hop_size)
    mag    = np.abs(D)
    phase  = np.angle(D)
    freqs  = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    sigma  = max(width_hz / 2.355, 1.0)
    weight = np.exp(-0.5 * ((freqs - center_hz) / sigma) ** 2)
    new_phase = phase.copy()
    for frame in range(D.shape[1]):
        time_sec              = (frame * hop_size) / sr
        lfo                   = np.sin(2 * np.pi * lfo_rate * time_sec)
        new_phase[:, frame]   = phase[:, frame] + phase_depth * lfo * weight
    D_new = mag * np.exp(1j * new_phase)
    return norm(librosa.istft(D_new, hop_length=hop_size, length=len(x)))


def fft_phaser_from_wav(audio_file, out_name="fft_phaser", max_seconds=30):
    """
    Loads a WAV file, processes each channel independently through fft_phaser,
    and saves as a proper stereo WAV: preserving the original stereo format.
    Also saves a headphone comparison version (dry left, wet right).
    """
    sr, data = wavfile.read(audio_file)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    data = data[:int(max_seconds * sr)]

    if data.ndim == 2:
        # process each channel independently: preserves stereo imaging
        left  = fft_phaser(data[:, 0], sr)
        right = fft_phaser(data[:, 1], sr)
    else:
        left  = fft_phaser(data, sr)
        right = left.copy()

    # save processed stereo (both channels phased)
    stereo = np.column_stack([left, right])
    path = os.path.join(AUDIO_OUT, f"{out_name}.wav")
    wavfile.write(path, sr, np.int16(np.clip(stereo, -1, 1) * 32767))
    print(f"saved: {path}")


# run all three on Bad Bunny 

data, sr = load_audio(AUDIO_IN)

# simple and allpass saved as stereo comparison (dry left, wet right)
save_stereo("phaser_simple",  data, simple_phaser(data, sr),  sr)
save_stereo("phaser_allpass", data, allpass_phaser(data, sr), sr)

# fft_phaser saved as proper stereo: the best/more obvious sounding version
fft_phaser_from_wav(AUDIO_IN, out_name="phaser_fft")

# process each channel independently, save as true stereo phased output
sr, raw = wavfile.read(AUDIO_IN)
raw = raw.astype(np.float32) / np.iinfo(raw.dtype).max
raw = raw[:int(30 * sr)]
left  = allpass_phaser(raw[:, 0], sr)
right = allpass_phaser(raw[:, 1], sr)
stereo = np.column_stack([norm(left), norm(right)])
wavfile.write(os.path.join(AUDIO_OUT, "phaser_allpass_stereo.wav"), sr,
              np.int16(np.clip(stereo, -1, 1) * 32767))
print("saved: phaser_allpass_stereo.wav")