import numpy as np
import librosa
import scipy.signal
from SWIPE import *

def find_glottal_cycles(audio, sr):
    """
    Identify glottal cycles by detecting fundamental frequency (F0)
    and extracting period durations (inverse of F0).
    """
    ## Estimate fundamental frequency using SWIPE
    f0, t, s = swipep(audio, sr, np.array([25, 400]), 0.0015, 1/96, 0.1, float('-inf'))
    
    #f0, voiced_flag, voiced_probs = librosa.pyin(audio , fmin=20, fmax=750, sr=sr)
    # Convert f0 to period length in samples (T = 1 / F0)
    period_samples = sr / f0
    period_samples = period_samples[~np.isnan(period_samples)]  # Remove NaN values

    return period_samples

def calculate_jitter_rap(periods):
    """Computes Jitter (RAP) from extracted glottal periods."""
    N = len(periods)
    if N < 3:
        return None  # Not enough cycles to compute Jitter RAP

    # Compute numerator: Average absolute difference from the moving average of three periods
    numerator = np.sum(np.abs(periods[1:-1] - np.convolve(periods, np.ones(3)/3, 'valid')))
    numerator /= (N - 2)

    # Compute denominator: Mean period duration
    denominator = np.mean(periods)

    # Compute Jitter RAP (%)
    jitter_rap = (numerator / denominator) * 100 if denominator != 0 else 0
    return jitter_rap

def jitter_rap_from_wav(file_path):
    """Loads the WAV file, extracts glottal periods, and computes Jitter RAP."""
    audio, sr = librosa.load(file_path, sr=None)  # Load audio

    # Detect glottal cycles (extract fundamental periods)
    periods = find_glottal_cycles(audio, sr)

    # Compute Jitter RAP
    jitter_rap_value = calculate_jitter_rap(periods)
    return jitter_rap_value

# Example Usage
wav_file = "K1024_7.1-2-a_1.wav"  #Replace with your actual file
jitter_rap_value = jitter_rap_from_wav(wav_file)

print(f"Jitter (RAP): {jitter_rap_value:.6f} %")
