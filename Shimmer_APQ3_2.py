import numpy as np
import librosa
import scipy.signal

def extract_amplitude_cycles(audio, sr):
    """Extracts peak-to-peak amplitudes for glottal cycles."""
    peaks, _ = scipy.signal.find_peaks(audio, height=0)  # Find peaks (potential glottal cycles)
    
    if len(peaks) < 3:
        return None  # Not enough cycles to compute APQ3

    amplitudes = np.abs(audio[peaks])  # Extract amplitude at peaks
    return amplitudes

def calculate_apq3(amplitudes):
    """Computes Shimmer (APQ3) from the extracted amplitudes using the given formula."""
    N = len(amplitudes)
    if N < 3:
        return None  # Not enough cycles to compute APQ3

    # Compute moving average over three consecutive cycles
    moving_avg = np.convolve(amplitudes, np.ones(3)/3, mode='valid')

    # Compute numerator: Mean of absolute deviations
    numerator = np.sum(np.abs(amplitudes[1:-1] - moving_avg)) / (N - 2)

    # Compute denominator: Mean amplitude
    denominator = np.mean(amplitudes)

    # Compute APQ3 and ensure correct scaling
    apq3 = (numerator / denominator) * 100 if denominator != 0 else 0
    return apq3

def shimmer_apq3_from_wav(file_path):
    """Loads the WAV file, extracts amplitudes, and computes APQ3."""
    audio, sr = librosa.load(file_path, sr=None)  # Load audio
    amplitudes = extract_amplitude_cycles(audio, sr)  # Extract peak-to-peak amplitudes
    if amplitudes is None:
        return None
    apq3_value = calculate_apq3(amplitudes)  # Compute APQ3
    return apq3_value

# Example Usage
wav_file = "K1003_7.1-2-a_1.wav"  # Replace with your actual file
shimmer_apq3_value = shimmer_apq3_from_wav(wav_file)

print(f"Shimmer (APQ3): {shimmer_apq3_value:.2f}")
