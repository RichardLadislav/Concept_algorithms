import numpy as np
import librosa
import scipy.io.wavfile as wav
import scipy.signal as signal

def compute_hnr(signal, fs, frame_length=0.05, overlap=0.5, f0_min=75, f0_max=500):
    """
    Compute Harmonics-to-Noise Ratio (HNR) using autocorrelation with Hamming window.
    
    Parameters:
    - signal: The input audio signal
    - fs: Sampling frequency (Hz)
    - frame_length: Frame length in seconds (default: 40 ms)
    - overlap: Overlap percentage (default: 50%)
    - f0_min: Minimum fundamental frequency (Hz)
    - f0_max: Maximum fundamental frequency (Hz)

    Returns:
    - Mean HNR value across frames
    """

    frame_size = int(frame_length * fs)
    step_size = int(frame_size * (1 - overlap))  # Step size based on overlap
    num_frames = int((len(signal) - frame_size) / step_size) + 1

    hnr_values = []

    for i in range(num_frames):
        frame = signal[i * step_size : i * step_size + frame_size]
        
        if len(frame) < frame_size:
            continue

        # Apply Hamming window
        #frame = frame * np.hamming(len(frame))
        #frame = frame * np.hanning(len(frame))
        frame = frame * np.blackman(len(frame))
        # Compute normalized autocorrelation
        autocorr = np.correlate(frame, frame, mode='full') / np.dot(frame, frame)
        autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags

        # Find fundamental period (max autocorr peak within f0 range)
        min_period = int(fs / f0_max)  # Corresponds to max frequency
        max_period = int(fs / f0_min)  # Corresponds to min frequency

        if max_period >= len(autocorr):
            max_period = len(autocorr) - 1

        peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
        r_max = autocorr[peak_idx]  # Max autocorrelation peak

        # Compute HNR using the definition
        if r_max > 0:
            hnr = 10 * np.log10(r_max / (1 - r_max))
            hnr_values.append(hnr)

    if len(hnr_values) > 0:
        return np.mean(hnr_values)
    else:
        return float('nan')

if __name__ == "__main__":


    # Read input arguments
    file_path = "K1003_7.1-2-a_1.wav"  #Replace with your actual file
    overlap_percentage = 50 / 100  # Convert percentage to fraction

    # Read audio file
    fs, audio = wav.read(file_path)

#    audio, fs = librosa.load(file_path, sr=None)  # Load audio
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Compute HNR
    hnr_value = compute_hnr(audio, fs, overlap=overlap_percentage)

    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
