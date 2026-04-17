import os
import numpy as np

# setup

# fs = 200.0
# # sampling rate
# N = 1024
# # number of samples
# t = np.arange(N) / fs
# # time axis

fs = 44100.0  # Standard audio sampling rate (44.1 kHz)
N = 44100     # 1 second of audio at 44.1 kHz
t = np.arange(N) / fs

# Keep frequencies in the audible range 
f1 = 10.0
f2 = 70.0
noise_std = 0.02

# nyquist = fs/2


#generating synthetic data, long array - used chat gpt for this function 
x = np.sin(2*np.pi*f1*t) + 0.6*np.sin(2*np.pi*f2*t) + noise_std*np.random.randn(N)

#helper function to make comparing our multiple examples easier, saving files, etc 
def comparison_metrics(name, a, b):
    os.makedirs("experiments/fft/data", exist_ok=True)
    # new folder for organization 

    err = b - a
    rmse = np.sqrt(np.mean(err**2))
    max_abs = np.max(np.abs(err))

    #saving the fft and ifft results, and their comparisons to files to read/access
    np.savetxt(f"experiments/fft/data/{name}_original.csv", a, delimiter=",")
    np.savetxt(f"experiments/fft/data/{name}_recon.csv", b, delimiter=",")
    with open(f"experiments/fft/data/{name}_metrics.txt", "w") as f:
        f.write(f"{name}\n")
        f.write(f"RMSE = {rmse:.6e}\n")
        f.write(f"MaxAbs = {max_abs:.6e}\n")
    print(name, "| RMSE:", rmse, "| MaxAbs:", max_abs)


# 1: Identity (should be perfect match)

X1 = np.fft.fft(x)
x1_recon = np.fft.ifft(X1).real
comparison_metrics("test1_identity", x, x1_recon)

#error is 10 to the power of -16, almost 0, negligible, this can be considered a perfect reconstruction 

# 2: Windowed signal compared to original (will differ)

w = np.hanning(N)
# hann window, could also do rectangle, which is no window, or hamming (might be better), and there are others too 
# similar to chunking but it multplies instead of just taking a piece of the entire signal
# both modify amplitude, and the frequencies, prety much changing the signal, 
# so of course the results will be different 
x_win = x * w

X2 = np.fft.fft(x_win)
x2_recon = np.fft.ifft(X2).real  # reconstructs x_win, not x

# original x vs reconstruction of windowed version (mismatch)
comparison_metrics("test2_windowed_vs_original", x, x2_recon)
# error is more significant, 0.5, since we are comparing the original signal, vs the fft to ifft on the window, not actually the same thing 

# compare x_win vs x2_recon, will match correctly
comparison_metrics("test2_windowed_correct_compare", x_win, x2_recon)



# 3: Chunking (FFT to IFFT matches the chunk; chunk differs from entire signal)

chunk_len = 256
chunk_start = 200
chunk_end = min(chunk_start + chunk_len, N)

#has spectral leakage since there are abrupt edges to the end and beginning of our chunk

x_chunk = x[chunk_start:chunk_end]

# pad if chunk hits the end (keeps chunk_len consistent)
if x_chunk.size < chunk_len:
    x_chunk = np.pad(x_chunk, (0, chunk_len - x_chunk.size))

X3 = np.fft.fft(x_chunk)
x3_recon = np.fft.ifft(X3).real

# chunk vs reconstructed chunk (should be perfect match)
comparison_metrics("test3_chunking", x_chunk, x3_recon)
 