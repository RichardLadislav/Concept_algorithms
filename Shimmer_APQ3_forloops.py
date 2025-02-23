import numpy as np
import librosa
import scipy.signal

def extract_glottal_amplitudes(audio, sr):
    """Extracts peak-to-peak amplitudes by detecting glottal cycle peaks."""
    peaks, _ = scipy.signal.find_peaks(audio, height=0)  # Detect peaks in waveform
    
    if len(peaks) < 3:
        return None  # Not enough cycles to compute APQ3

    # Compute cycle-to-cycle peak-to-peak amplitudes
    amplitudes = []
    for i in range(len(peaks) - 1):  # Iterate through detected cycles
        amp = abs(audio[peaks[i]] - audio[peaks[i+1]])  # Peak-to-peak difference
        amplitudes.append(amp)

    return amplitudes

def calculate_apq3(amplitudes):
    """Computes Shimmer (APQ3) from extracted amplitudes using the exact equation."""
    N = len(amplitudes)
    if N < 3:
        return None  # Not enough data points

    # Compute numerator using a rolling 3-cycle moving average
    sum_abs_diffs = 0
    for i in range(N - 2):  # Iterate over valid range for 3-cycle windows
        avg_three = (amplitudes[i] + amplitudes[i+1] + amplitudes[i+2]) / 3  # 3-cycle mean
        sum_abs_diffs += abs(amplitudes[i] - avg_three)

    numerator = sum_abs_diffs / (N - 2)  # Normalize over (N-2)

    # Compute denominator: Mean amplitude over N cycles
    denominator = sum(amplitudes) / N

    # Compute APQ3 and ensure correct scaling
    apq3 = (numerator / denominator) * 100 if denominator != 0 else 0
    return apq3

def shimmer_apq3_from_wav(file_path):
    """Loads the WAV file, extracts amplitudes, and computes APQ3."""
    audio, sr = librosa.load(file_path, sr=None)  # Load audio
    amplitudes = extract_glottal_amplitudes(audio, sr)  # Extract peak-to-peak amplitudes
    if amplitudes is None:
        return None
    apq3_value = calculate_apq3(amplitudes)  # Compute APQ3
    return apq3_value

# Example Usage
wav_file = "K1003_7.1-2-e_1.wav"  # Replace with your actual file
shimmer_apq3_value = shimmer_apq3_from_wav(wav_file)

print(f"Shimmer (APQ3): {shimmer_apq3_value:.2f}")

