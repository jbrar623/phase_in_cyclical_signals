import numpy as np
import librosa
import soundfile as sf
import os


def phase_vocoder(filename, time_stretch=1.5, output_dir="sounds"):
    """
    Phase vocoder using librosa's implementation.
    time_stretch > 1 = slower, < 1 = faster.

    Uses librosa's refined algorithm which includes:
    - Peak picking: identifies spectral peaks and treats them separately
    - Phase locking: bins near a peak inherit the peak's phase correction
      rather than computing their own, which preserves harmonic relationships
    These refinements make it sound clean and natural.
    """
    y, sr = librosa.load(filename, sr=None, mono=True)
    y_stretched = librosa.effects.time_stretch(y, rate=1/time_stretch)

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/phase_vocoder_{time_stretch}x.wav"
    sf.write(out_path, y_stretched, sr)
    print(f"saved: {out_path}")
    return y_stretched, sr


def manual_vocoder(filename, time_stretch=1.5, n_fft=2048, hop_length=512, output_dir="sounds"):
    """
    Manual phase vocoder — exposes the raw algorithm explicitly.

    INTENTIONAL LIMITATIONS:
    - No peak picking: every bin computes its own phase correction independently
    - No phase locking: harmonically related bins can drift out of alignment
      with each other, causing the characteristic "phasey" or "underwater" sound
    - Simple nearest-frame magnitude interpolation rather than weighted blending

    These limitations mean this will sound WORSE than the librosa version.
    This difference demonstrates what phase coherence between bins sounds like when it breaks.
    The artifacts you hear are phase relationships going wrong in exactly the
    same way as the random-phase demo, but gradually rather than all at once.

    THE KEY STEP — phase propagation (the entire vocoder idea in 3 lines):
        delta_phase = phase[:, next] - phase[:, current] - bin_freqs
            → how much did phase advance beyond what was expected?
        delta_phase = wrap to [-π, π]
            → keep the deviation bounded (phase wraps at ±π)
        phase_acc += bin_freqs + delta_phase
            → accumulate: expected advance + corrected deviation

    """
    y, sr = librosa.load(filename, sr=None, mono=True)

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    phase = np.angle(D)

    n_frames = D.shape[1]
    n_bins = D.shape[0]

    # expected phase advance per hop for each frequency bin
    # bin k has center frequency k * sr / n_fft
    # in hop_length samples: expected advance = 2π * k * hop_length / n_fft
    bin_freqs = np.arange(n_bins) * 2 * np.pi * hop_length / n_fft

    n_frames_out = int(n_frames * time_stretch)
    phase_acc = phase[:, 0].copy()
    D_out = np.zeros((n_bins, n_frames_out), dtype=complex)

    for i in range(n_frames_out):
        # map output frame back to nearest input frame
        input_frame = i / time_stretch
        frame_idx = min(int(input_frame), n_frames - 2)

        # build output frame: input magnitude, accumulated output phase
        D_out[:, i] = mag[:, frame_idx] * np.exp(1j * phase_acc)

        # phase propagation — the core vocoder step
        delta_phase = phase[:, frame_idx + 1] - phase[:, frame_idx] - bin_freqs
        delta_phase = delta_phase - 2 * np.pi * np.round(delta_phase / (2 * np.pi))
        phase_acc += bin_freqs + delta_phase

    y_out = librosa.istft(D_out, hop_length=hop_length, length=int(len(y) * time_stretch))
    y_out = y_out / (np.max(np.abs(y_out)) + 1e-12)

    os.makedirs(output_dir, exist_ok=True)
    out_path = f"{output_dir}/manual_vocoder_{time_stretch}x.wav"
    sf.write(out_path, y_out, sr)
    print(f"saved: {out_path}")
    return y_out, sr


# usage
phase_vocoder("bad_bunny/sounds/DtMF - Bad Bunny (128k).wav", time_stretch=1.5)
manual_vocoder("bad_bunny/sounds/DtMF - Bad Bunny (128k).wav", time_stretch=1.5)