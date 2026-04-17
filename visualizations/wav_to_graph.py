import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

def plot_spectrogram(filename, n_fft=4096, hop=64, title=None, output_dir=None):
    y, sr = librosa.load(filename, sr=None)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure()
    librosa.display.specshow(D_db, sr=sr, hop_length=hop, x_axis="time", y_axis="hz")
    plt.title(title or filename)
    plt.colorbar(format='%+2.0f dB')
    #plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename))[0]
        save_path = os.path.join(output_dir, f"{base}_spectrogram.png")
        plt.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")

    #plt.show()

def plot_partial_waveform(filename, output_dir=None, start_sec=0, duration_sec=0.05, label=None):
    y, sr = librosa.load(filename, sr=None, mono=True)
    start_sample = int(start_sec * sr)
    end_sample = int((start_sec + duration_sec) * sr)
    chunk = y[start_sample:end_sample]
    t = np.linspace(start_sec, start_sec + duration_sec, len(chunk))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, chunk, linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(label or os.path.basename(filename))
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename))[0].replace(".", "_")
        path = os.path.join(output_dir, f"{base}_waveform.png")
        plt.savefig(path, dpi=150)
        print(f"saved: {path}")

    #plt.show()


def plot_phase_scatter(filename, time_sec=1.0, n_fft=4096, hop=64,
                       freq_limit_hz=8000, output_dir=None):
    y, sr = librosa.load(filename, sr=None, mono=True)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop)

    frame_idx = min(int(time_sec * sr / hop), D.shape[1] - 1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    phase = np.angle(D[:, frame_idx])
    mag = np.abs(D[:, frame_idx])

    mask = freqs <= freq_limit_hz
    mag_norm = mag[mask] / (mag[mask].max() + 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    sc = ax.scatter(freqs[mask], phase[mask], s=mag_norm * 40 + 1,
                    c=mag_norm, cmap="plasma", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="Relative magnitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase angle (radians)")
    ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_title(f"{os.path.basename(filename)} — phase at t={time_sec:.2f}s")
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename))[0].replace(".", "_")
        path = os.path.join(output_dir, f"{base}_phase_scatter.png")
        plt.savefig(path, dpi=150)
        print(f"saved: {path}")

    #plt.show()


def plot_waveform(filename, output_dir=None, label=None):
    y, sr = librosa.load(filename, sr=None, mono=True)
    t = np.arange(len(y)) / sr

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, y, linewidth=0.3, alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(label or os.path.basename(filename))
    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(filename))[0].replace(".", "_")
        path = os.path.join(output_dir, f"{base}_waveform_full.png")
        plt.savefig(path, dpi=150)
        print(f"saved: {path}")

    #plt.show()


# usage

files = [
    "bad_bunny/sounds/DtMF - Bad Bunny (128k).wav",
    "bad_bunny/sounds/hilbert/H_real.wav",
    "bad_bunny/sounds/hilbert/y_real.wav",
    "bad_bunny/sounds/hilbert/abs_H.wav",
    "bad_bunny/sounds/output_bb_eq.wav",
    "bad_bunny/sounds/output_wavs/2_zero_phase.wav",
]

for f in files:
    plot_spectrogram(f, output_dir="visualizations/spectograms")
    plot_waveform(f, output_dir="visualizations/waveforms")
    plot_phase_scatter(f, time_sec=5.0, output_dir="visualizations/phase_scatter")