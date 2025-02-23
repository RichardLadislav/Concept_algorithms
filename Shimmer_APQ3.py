import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt
def extract_amplitude_peaks(audio, sr):
    """Extracts the amplitudes of glottal cycles by detecting peaks in the waveform."""
    #FIXME: pravdepodobne chyba v hladani amplitud, treba opravit nejakym chytrejsim spusobem 
    peaks, _ = scipy.signal.find_peaks(audio, height=0)  # Find peaks (potential cycle peaks)
    amplitudes = np.abs(audio[peaks])  # Extract amplitude at peaks
   # plt.plot(audio)
    #plt.scatter(peaks, audio[peaks], color='red')  # Show detected peaks
    #plt.show()
    return amplitudes


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
    amplitudes = extract_amplitude_peaks(audio, sr)  # Extract amplitude peaks
    apq3_value = calculate_apq3(amplitudes)  # Compute APQ3
    return apq3_value

# Example Usage
wav_file = "K1003_7.1-2-a_1.wav"  # Replace with your actual file
shimmer_apq3_value = shimmer_apq3_from_wav(wav_file)

print(f"Shimmer (APQ3): {shimmer_apq3_value}")

