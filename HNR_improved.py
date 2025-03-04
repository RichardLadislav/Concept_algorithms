import librosa
import numpy as np

def normalized_autocorrelatiob(frame):
    norm_factor = np.dot(frame, frame)
    autocorr  = np.correlate(frame, frame, mode = 'full') / norm_factor
    return autocorr[len(autocorr) // 2:]

def cepstral_pitch(frame, fs):
    spectrum = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.fft.ifft(log_spectrum).real
    
    # Limit search range based on expecter F0
    min_quefrency = int(fs/500)
    max_quefrency = int(fs/500)
    pitch_peak = np.argmax(cepstrum[min_quefrency:max_quefrency]) + min_quefrency
    
    return fs / pitch_peak


