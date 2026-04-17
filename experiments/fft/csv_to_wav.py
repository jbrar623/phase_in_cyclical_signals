# csv_to_wav_converter.py

import numpy as np
from scipy.io import wavfile
import os

def csv_to_wav(csv_path, wav_path, sample_rate=44100, repeat=1):
    """
    Convert a CSV file containing signal data to a WAV file.
    
    Args:
        csv_path: Path to input CSV file
        wav_path: Path to output WAV file
        sample_rate: Sampling rate in Hz (default: 44100)
    """
    # Load signal
    signal = np.loadtxt(csv_path, delimiter=",")

    # Repeat signal to extend duration
    if repeat > 1:
        signal = np.tile(signal, repeat)

    # Safe normalization (avoid divide-by-zero)
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    # Convert to 16-bit 
    audio_data = np.int16(signal * 32767)

    # Write WAV
    wavfile.write(wav_path, int(sample_rate), audio_data)

    # Print duration for verification
    duration = len(signal) / sample_rate
    #print(f"Converted {csv_path} -> {wav_path} | duration: {duration:.2f}s at {sample_rate} Hz")


if __name__ == "__main__":
    # Convert regular data folder (fs = 44100)
    data_dir = "experiments/fft/data"
    default_fs = 44100
    
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".csv"):
                csv_path = os.path.join(data_dir, filename)
                wav_path = csv_path.replace(".csv", ".wav")
                csv_to_wav(csv_path, wav_path, default_fs)
    
    # Convert nyquist folder with appropriate sample rates
    nyquist_dir = "experiments/fft/data/nyquist"
    
    if os.path.exists(nyquist_dir):
        # Load the sample rates
        rates_file = os.path.join(nyquist_dir, "test4_nyquist_sample_rates.csv")
        if os.path.exists(rates_file):
            rates = np.loadtxt(rates_file, delimiter=",")
            fs_truth = int(rates[0])
            fs_low = int(rates[1])
        else:
            # default values
            fs_truth = 44100
            fs_low = 8000
        
        # Convert files with correct sample rates
        for filename in os.listdir(nyquist_dir):
            if filename.endswith(".csv") and filename != "test4_nyquist_sample_rates.csv":
                csv_path = os.path.join(nyquist_dir, filename)
                wav_path = csv_path.replace(".csv", ".wav")
                
                # Use correct sample rate
                if "highrate" in filename or "truth" in filename:
                    csv_to_wav(csv_path, wav_path, fs_truth, 4)
                elif "lowrate" in filename:
                    csv_to_wav(csv_path, wav_path, fs_low, 4)