#!/usr/bin/env python
"""
Wavelet-based time–frequency analysis of eye-tracking trials.

Pipeline:
1. Load pre-segmented trials
2. Compute CWT (single-trial + optionally across trials)
3. Extract power, cross-power, coherence
4. Visualise:
   - time-averaged spectra
   - single-trial spectrograms
   - cross-spectrograms
   - coherence maps
"""

import h5py
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =========================================================
# CONFIGURATION
# =========================================================

TRIALS_PATH = "data/processed/trials.h5"

FS = 1000  # Hz

WAVELET = "cmor1.0-0.5"

FREQS_HZ = np.logspace(0, 2.5, 100)  # 1–300 Hz approx

OUTPUT_DIR = "plots"


# =========================================================
# LOAD DATA
# =========================================================

with h5py.File(TRIALS_PATH, "r") as f:
    x_trials = np.array(f["x_trials"])
    y_trials = np.array(f["y_trials"])
    fs = int(np.array(f["fs"]))

n_trials = x_trials.shape[0]
n_time = x_trials.shape[-1]

print(f"Loaded {n_trials} trials | {n_time} timepoints | fs={fs}")


# =========================================================
# WAVELET HELPERS
# =========================================================

def compute_scales(freqs, fs, wavelet):
    """Convert target frequencies to wavelet scales."""
    fc = pywt.central_frequency(wavelet)
    return fc * fs / freqs


def cwt(signal, freqs, fs, wavelet):
    """Continuous wavelet transform."""
    scales = compute_scales(freqs, fs, wavelet)
    W, _ = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    return W


def wavelet_power(W):
    return np.abs(W) ** 2


# =========================================================
# SMOOTHING (MAKES COHERENCES BETTER DEFINED)
# =========================================================

def smooth_timefreq(x, sigma_f=2, sigma_t=50):
    """2D Gaussian smoothing (freq x time)."""
    return gaussian_filter(x, sigma=(sigma_f, sigma_t))


# =========================================================
# SINGLE-TRIAL ANALYSIS
# =========================================================

def analyse_single_trial(x_trials, trial_idx=0):
    """Compute and plot wavelet representations for one trial."""

    left = x_trials[trial_idx, 0, :]
    right = x_trials[trial_idx, 1, :]

    Wl = cwt(left, FREQS_HZ, fs, WAVELET)
    Wr = cwt(right, FREQS_HZ, fs, WAVELET)

    Pl = wavelet_power(Wl)
    Pr = wavelet_power(Wr)
    cross = Wl * np.conj(Wr)

    # Smooth the auto-spectra
    S_Pl = smooth_timefreq(Pl)
    S_Pr = smooth_timefreq(Pr)

    # Smooth the complex cross-spectrum (real and imag separately)
    S_cross_real = smooth_timefreq(np.real(cross))
    S_cross_imag = smooth_timefreq(np.imag(cross))
    S_cross_sq = S_cross_real**2 + S_cross_imag**2

    # Compute coherence from the smoothed components
    coh_smooth = S_cross_sq / (S_Pl * S_Pr + 1e-12)

    # ----------------------------
    # Time-averaged spectrum
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(FREQS_HZ, Pl.mean(axis=1), label="Left")
    plt.plot(FREQS_HZ, Pr.mean(axis=1), label="Right")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"Trial {trial_idx} - Time-averaged spectrum")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/time_avg_spectrum.png", dpi=150)
    plt.close()

    # ----------------------------
    # Spectrogram (left eye)
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, FREQS_HZ, Pl,
                   shading="auto", cmap="magma")
    plt.yscale("log")
    plt.colorbar(label="Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Left eye wavelet power")
    plt.savefig(f"{OUTPUT_DIR}/spectrogram_left.png", dpi=150)
    plt.close()

    # ----------------------------
    # Cross-spectrogram
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, FREQS_HZ, np.abs(cross),
                   shading="auto", cmap="plasma")
    plt.yscale("log")
    plt.colorbar(label="Cross power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Cross wavelet power")
    plt.savefig(f"{OUTPUT_DIR}/cross_spectrum.png", dpi=150)
    plt.close()

    # ----------------------------
    # Coherence (smoothed)
    # ----------------------------

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, FREQS_HZ, coh_smooth,
                   shading="auto", cmap="viridis", vmin=0, vmax=1)
    plt.yscale("log")
    plt.colorbar(label="Coherence")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Wavelet coherence (smoothed)")
    plt.savefig(f"{OUTPUT_DIR}/coherence.png", dpi=150)
    plt.close()


def analyse_tremor_single_trial(x_trials, trial_idx=0, fs=1000):
    left_pos = x_trials[trial_idx, 0, :]
    right_pos = x_trials[trial_idx, 1, :]

    # differentiating
    # flattening the 1/f spectrum 
    left_vel = np.gradient(left_pos) * fs
    right_vel = np.gradient(right_pos) * fs

    # isolate gamma
    freqs_tremor = np.logspace(np.log10(1), np.log10(200), 80)

    # wavelet transform
    Wl = cwt(left_vel, freqs_tremor, fs, WAVELET)
    Wr = cwt(right_vel, freqs_tremor, fs, WAVELET)

    Pl = wavelet_power(Wl)
    Pr = wavelet_power(Wr)
    cross = Wl * np.conj(Wr)

    # smoothing for coherence
    sigma_f = 5
    sigma_t = 10

    S_Pl = gaussian_filter(Pl, sigma=(sigma_f, sigma_t))
    S_Pr = gaussian_filter(Pr, sigma=(sigma_f, sigma_t))
    
    S_cross_real = gaussian_filter(np.real(cross), sigma=(sigma_f, sigma_t))
    S_cross_imag = gaussian_filter(np.imag(cross), sigma=(sigma_f, sigma_t))
    S_cross_sq = S_cross_real**2 + S_cross_imag**2

    # Calculate true coherence
    coh_smooth = S_cross_sq / (S_Pl * S_Pr + 1e-12)

    # ----------------------------
    # Time-averaged spectrum
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(freqs_tremor, Pl.mean(axis=1), label="Left")
    plt.plot(freqs_tremor, Pr.mean(axis=1), label="Right")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title(f"Trial {trial_idx} - Time-averaged velocity spectrum")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/velocity_time_avg_spectrum.png", dpi=150)
    plt.close()

    # ----------------------------
    # Spectrogram (left eye)
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, freqs_tremor, Pl,
                   shading="auto", cmap="magma")
    plt.yscale("log")
    plt.colorbar(label="Power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Left eye wavelet velocity power")
    plt.savefig(f"{OUTPUT_DIR}/velocity_spectrogram_left.png", dpi=150)
    plt.close()

    # ----------------------------
    # Cross-spectrogram
    # ----------------------------
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, freqs_tremor, np.abs(cross),
                   shading="auto", cmap="plasma")
    plt.yscale("log")
    plt.colorbar(label="Cross power")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Cross wavelet velocity power")
    plt.savefig(f"{OUTPUT_DIR}/velocity_cross_spectrum.png", dpi=150)
    plt.close()

    # ----------------------------
    # Coherence (smoothed)
    # ----------------------------

    plt.figure(figsize=(8, 4))
    plt.pcolormesh(np.arange(n_time)/fs, freqs_tremor, coh_smooth,
                   shading="auto", cmap="viridis", vmin=0, vmax=1)
    plt.yscale("log")
    plt.colorbar(label="Coherence")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Wavelet velocity coherence (smoothed)")
    plt.savefig(f"{OUTPUT_DIR}/velocity_coherence.png", dpi=150)
    plt.close()

# =========================================
# POPULATION-LEVEL SPECTRUM
# =========================================

def population_spectrum(x_trials):
    """Average wavelet power across trials (left eye only)."""

    all_power = []

    for i in range(n_trials):
        W = cwt(x_trials[i, 0, :], FREQS_HZ, fs, WAVELET)
        all_power.append(wavelet_power(W))

    all_power = np.array(all_power)

    mean_power = all_power.mean(axis=0)
    median_power = np.median(all_power, axis=0)

    # Mean spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(FREQS_HZ, mean_power.mean(axis=1), label="Mean")
    plt.plot(FREQS_HZ, median_power.mean(axis=1), label="Median")
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Population spectrum (left eye)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/population_spectrum.png", dpi=150)
    plt.close()


# =========================================================
# RUN ANALYSIS
# =========================================================

if __name__ == "__main__":

    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running single-trial analysis...")
    analyse_single_trial(x_trials, trial_idx=0)

    print("Running single-trial velocity analysis...")
    analyse_tremor_single_trial(x_trials, trial_idx=0)

    # print("Running population analysis...")
    # population_spectrum(x_trials)

    print("Done.")
