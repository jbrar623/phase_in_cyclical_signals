# Understanding Phase in Cyclical Signals Using Sound

## Research Context

This project is part of a broader research question:

**How can listening and visualization be used together to help people develop an intuitive understanding of phase in cyclical signals?**

Phase is fundamental to signal processing, but it is also one of the hardest concepts to build intuition for. Most tools prioritize magnitude (like spectrograms), while phase is either hidden or treated as secondary. At the same time, phase is not directly audible by itself, which makes it harder to understand.

This project approaches that gap directly by combining:
- listening experiments, what does a phase change sound like?
- visualizations, what does phase look like?
- mathematical framing, what is actually happening under the hood?

The goal is to make phase intuitive, the way frequency and other signal concepts are.

---

## Two Components

This repository is one of two parts of the project. The other is a **dictionary** that bridges the gap between mathematical definitions and audio engineering application. Terms like phaser, phase vocoder, instantaneous frequency, and the Hilbert transform each have precise mathematical meanings, but they also have practical identities in music production and audio engineering that are not often connected to that math. The dictionary attempts to define each term formally and then explain what is mathematically happening to the sound. The code here is the experimental side: it provides audio that supports and explains those definitions. 

---

## What This Project Is Doing

When you compute a Fourier transform, you get a complex-valued representation:
- magnitude → how much of each frequency exists
- phase → how those frequencies are aligned in time

Most visual tools (like spectrograms) discard phase and only show magnitude. This project focuses on that missing part.

Everything here is built around the same decomposition:

```python
D = librosa.stft(x)
M = np.abs(D)
phi = np.angle(D)
```

From there, each experiment answers a version of the same question: what happens when we isolate, manipulate, destroy, or preserve phase, and how does that affect what we hear and see?

---

## Methodology: Listening + Visualization

The core design choice in this project is pairing audio perception with visual representation.

**Listening**

Almost all experiments are exported as stereo WAV files:
- left channel → original
- right channel → modified

This allows direct A/B comparison in real time. These comparisons are best heard in headphones. 

**Visualization**

The project uses three complementary views:

- **Spectrogram** (STFT magnitude) — based on the Short-Time Fourier Transform, shows what we usually see (magnitude only)
- **Waveform** (time domain) — shows amplitude over time
- **Phase scatter plots** — explicitly show phase vs frequency at an instance

Spectrograms show what changes in magnitude look like. These experiments show what phase changes sound like, and what we have been missing visually.

---

## Project Structure

The repository is organized sequentially:
1. Foundations to explain what FFT/IFFT actually do
2. Synthetic experiments to isolate phase in simple signals
3. Real audio to test whether the same ideas hold in music
4. Visualization to compare what we hear vs what we see

---

## Key Experimental Themes

### 1. Phase vs Magnitude
- Zero-phase reconstruction → signal becomes unrecognizable
- Random phase → noise-like output with identical spectrum
- Equalization → magnitude changes without touching phase

**Insight:** Magnitude alone is not enough to reconstruct meaningful sound.

### 2. Phase Is Often Not Directly Audible
- Hilbert transform (90° phase shift) sounds nearly identical to the original
- Sine vs cosine → perceptually indistinguishable

**Insight:** We are largely "phase-deaf" to uniform phase shifts.

### 3. Phase Becomes Audible Through Interaction
- Phaser effect → phase shift + dry signal = audible notches
- Harmonic phase shifts → change waveform shape
- Stereo phase differences → create spatial effects

**Insight:** Phase is not heard directly, it is heard through interference, cancellation, and spatialization.

### 4. Phase Encodes Structure
- Phase vocoder → requires careful phase propagation to avoid artifacts
- Broken phase → audible smearing and noise
- Phase-only reconstruction → structured but unnatural sound

**Insight:** Phase contains the temporal relationships that make sound coherent.

### 5. Time vs Phase Are Not the Same
- A small delay and a phase shift are equivalent for a pure tone, but not for broadband signals

**Insight:** Phase is frequency-dependent timing, not just a shift in time.

---

## Files (Brief Overview)

**Foundations:**
`fft_on_data.py`, `nyquist_test.py`, `csv_to_wav.py`

**Synthetic experiments:**
`phase_practice.py`, `sawtooth_phase.py`, `hilbert_practice.py`, `hilbert_tests.py`, `generate_phase_experiments.py`

**Real audio (Bad Bunny):**
`phase_on_real_audio.py`, `equalizer.py`, `hilbert_transform.py`, `phaser.py`, `phase_vocoder.py`

**Visualization:**
`wav_to_graph.py`

Each file isolates one concept, then connects it back to listening.

---

## How This Connects to the Research Question

This project suggests that:

- **Listening alone is not enough** since phase changes are often subtle or hidden
- **Visualization alone is not enough**, most common tools ignore phase
- **Together, they fill the gap**, listening reveals perceptual effects, visualization explains why they occur

The combination allows phase to move from abstract math → observable behavior → intuitive understanding.


Author:
Joban Brar 
jbrar23@mtroyal.ca 

Supervisors:
Charles Hepler, Dr. Peter Zizler