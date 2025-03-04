import librosa
import numpy as np
import scipy.signal

def read_audio(file_path):
    """Load audio file and return waveform and sample rate."""
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def find_glottal_cycles(audio, sr, f0_min=75, f0_max=300):
    """Find glottal cycle positions using fundamental frequency estimation."""
    f0, voiced_flag, voiced_probs = librosa.pyin(audio , fmin=20, fmax=750, sr=sr)
    period_samples = sr / f0
    period_samples = period_samples[~np.isnan(period_samples)]  # Remove NaNs
    cycle_positions = np.cumsum(period_samples).astype(int)
    cycle_positions = cycle_positions[cycle_positions < len(audio)]  # Keep within bounds
    return cycle_positions

def extract_max_min_peaks(audio, cycle_positions):
    """Extract max and min peak values within each glottal cycle."""
    max_peaks, min_peaks = [], []
    
    for i in range(len(cycle_positions) - 1):
        segment = audio[cycle_positions[i]:cycle_positions[i + 1]]
        if len(segment) > 0:
            max_peaks.append(np.max(segment))  # Maximum peak
            min_peaks.append(np.min(segment))  # Minimum peak
    
    return np.array(max_peaks), np.array(min_peaks)

def apply_moving_average(peaks, window_size=3):
    """Apply a moving average to smooth the peaks."""
    return np.convolve(peaks, np.ones(window_size) / window_size, mode='valid')

def process_peaks(max_peaks, min_peaks):
    """Process peaks based on the flowchart conditions."""
    max_threshold = 0.7 * np.max(max_peaks)
    min_threshold = 0.7 * np.min(min_peaks)

    filtered_max = max_peaks[max_peaks > max_threshold]
    filtered_min = min_peaks[min_peaks < min_threshold]

    if (len(filtered_max) >= 10 or len(filtered_min) >= 10):
        return filtered_max, filtered_min
    else:
        max_ma = apply_moving_average(max_peaks)
        min_ma = apply_moving_average(min_peaks)

        max_final = max_ma[max_ma > 0.7 * np.max(max_ma)]
        min_final = min_ma[min_ma < 0.7 * np.min(min_ma)]

        return max_final, min_final

def compute_apq3(max_peaks, min_peaks):
    """Compute Shimmer (APQ3) from peaks."""
    N = len(max_peaks)
    
    if N < 3:
        print("Not enough glottal cycles to compute APQ3.")
        return None

    numerator = np.sum(np.abs(max_peaks[1:-1] - (max_peaks[:-2] + max_peaks[1:-1] + max_peaks[2:]) / 3)) / (N - 2)
    mean_amp = np.mean(max_peaks)

    apq3 = numerator / mean_amp if mean_amp != 0 else 0
    return apq3

if __name__ == "__main__":
    wav_file = "P1021_7.1-1-e_1.wav"  # Change this to your file
    audio, sr = read_audio(wav_file)

    cycle_positions = find_glottal_cycles(audio, sr)
    max_peaks, min_peaks = extract_max_min_peaks(audio, cycle_positions)
    max_peaks, min_peaks = process_peaks(max_peaks, min_peaks)

    apq3_value = compute_apq3(max_peaks, min_peaks)
    
    if apq3_value is not None:
        print(f"Shimmer (APQ3): {apq3_value:.6f}")
