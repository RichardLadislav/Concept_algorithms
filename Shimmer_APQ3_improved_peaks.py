import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
from SWIPE import * 
def find_glottal_cycles(audio, sr, f0_min=75, f0_max=300):
    """
    Identify glottal cycles by detecting fundamental frequency (F0)
    and extracting peak positions per cycle.
    """
    # Estimate fundamental frequency using librosa
    #f0, voiced_flag, voiced_probs = librosa.pyin(audio , fmin=20, fmax=750, sr=sr)
    f0, t, s = swipep(audio, sr, np.array([25,400]),0.1,1/96,0.1,float('-inf'))
        
    # Convert f0 to period length in samples
    period_samples = sr / f0
    period_samples = period_samples[~np.isnan(period_samples)]  # Remove NaN values
    
    # Find cycle boundaries
    cycle_positions = np.cumsum(period_samples).astype(int)
    cycle_positions = cycle_positions[cycle_positions < len(audio)]  # Keep within bounds
    
    return cycle_positions

def extract_glottal_peaks(audio, cycle_positions):
    """
    Extract maximum absolute peak values and their positions within detected glottal cycles.
    """
    peaks = []
    
    for i in range(len(cycle_positions) - 1):
        cycle_segment = audio[cycle_positions[i]:cycle_positions[i + 1]]
        if len(cycle_segment) > 0:
            peak_value = np.max(np.abs(cycle_segment))  # Get max absolute amplitude
            peaks.append(peak_value)
    
    return np.array(peaks)

def calculate_apq3(amplitudes):
    """Computes Shimmer (APQ3) from the extracted amplitudes using the given formula."""
    N = len(amplitudes)
    if N < 3:
        return None  # Not enough cycles to compute APQ3

    # Compute numerator: Average absolute difference
#    test_ampitudes = np.abs(amplitudes[1:-1])

#    test_convolve = np.convolve(amplitudes, np.ones(3)/3, 'valid') 
    numerator = np.sum(np.abs(amplitudes[1:-1] - np.convolve(amplitudes, np.ones(3)/3, 'valid')))
    numerator /= (N - 1)

    # Compute denominator: Mean amplitude
    denominator = np.mean(amplitudes)


    # Compute APQ3
    apq3 = (numerator / denominator) * 100 if denominator != 0 else 0
    return apq3

def shimmer_apq3_from_wav(file_path):
    """Loads the WAV file, extracts amplitudes, ad computes APQ3."""
    audio, sr = librosa.load(file_path, sr=None)  # Load audio
     # Detect glottal cycles
    cycle_positions = find_glottal_cycles(audio, sr)

    # Extract glottal peaks
    peaks = extract_glottal_peaks(audio, cycle_positions)
    apq3_value = calculate_apq3(peaks)  # Compute APQ3
    return apq3_value

# Example Usage
wav_file = "K1003_7.1-2-e_1.wav"  # Replace with your actual file
shimmer_apq3_value = shimmer_apq3_from_wav(wav_file)

print(f"Shimmer (APQ3): {shimmer_apq3_value}")

