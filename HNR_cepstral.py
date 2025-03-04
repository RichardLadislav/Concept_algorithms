import numpy as np
from scipy.signal import welch  # Import welch from scipy.signal
import scipy.io.wavfile as wav
import librosa
import scipy.signal as signal

def pre_emphasis(signal, coeff=0.97):
    """Applies a pre-emphasis filter to boost high frequencies."""
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def frame_signal(signal, fs, frame_length=0.04, overlap=0.5):
    """Splits the signal into overlapping frames with a Hamming window."""
    frame_size = int(frame_length * fs)
    step_size = int(frame_size * (1 - overlap))
    
    num_frames = int((len(signal) - frame_size) / step_size) + 1
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        frame = signal[i * step_size : i * step_size + frame_size]
        if len(frame) < frame_size:
            continue
        frames[i, :] = frame * np.hamming(frame_size)
    
    return frames

def cepstral_pitch(frame, fs, f0_min=75, f0_max=500):
    """
    Estimates fundamental frequency using cepstral analysis.
    - Computes FFT, takes log magnitude, applies IFFT, and finds quefrency peak.
    """
    spectrum = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real

    # Convert quefrency to Hz limits
    min_quefrency = int(fs / f0_max)
    max_quefrency = int(fs / f0_min)

    # Find peak in valid quefrency range
    pitch_peak = np.argmax(cepstrum[min_quefrency:max_quefrency]) + min_quefrency
    return fs / pitch_peak  # Convert quefrency to Hz

def compute_hnr(signal, fs, frame_length=0.3, overlap=0.5, f0_min=75, f0_max=500, vad_threshold=0.01):
    """
    Computes HNR using cepstral pitch estimation.
    - Uses voiced frames only (energy-based VAD).
    - Estimates harmonic and noise power.
    """
    frames = frame_signal(signal, fs, frame_length, overlap)
    hnr_values = []

    for frame in frames:
        # Voice Activity Detection (VAD)
        energy = np.sum(frame ** 2) / len(frame)
        if energy < vad_threshold:
            continue  # Skip unvoiced frames

        # Estimate F0 using cepstrum
        f0 = cepstral_pitch(frame, fs, f0_min, f0_max)

        # Compute Harmonic and Noise power using PSD
        f, psd = welch(frame, fs, nperseg=len(frame) // 2)
        harmonic_power = np.sum(psd[f < f0])  # Sum power at harmonics
        noise_power = np.sum(psd[f >= f0])  # Sum power in non-harmonic region

        if noise_power > 0 and harmonic_power > 0:
            hnr = 10 * np.log10(harmonic_power / noise_power)
            hnr_values.append(hnr)

    return np.mean(hnr_values) if hnr_values else float('nan')

if __name__ == "__main__":


  #  if len(sys.argv) != 3:
  #      print("Usage: python hnr_cepstral.py <wav_file> <overlap_percentage>")
        #sys.exit(1)

    # Read input arguments
    file_path = "K1024_7.1-2-a_1.wav" 
    overlap_percentage = float(50) / 100  # Convert percentage to fraction

    # Read audio file
    audio, fs = librosa.load(file_path, sr = None) 

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio
    audio = audio / np.max(np.abs(audio))

    # Apply pre-emphasis
    audio = pre_emphasis(audio)

    # Compute HNR
    hnr_value = compute_hnr(audio, fs, overlap=overlap_percentage)

    if np.isnan(hnr_value):
        print("Could not compute HNR.")
    else:
        print(f"Mean HNR: {hnr_value:.2f} dB")
