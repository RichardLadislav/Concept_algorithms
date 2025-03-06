import numpy as np
from scipy.signal import resample
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import librosa
import SWIPE as sw
import matplotlib.pyplot as plt

def segmentation(audio, winlen, winover):
    wl = 0
    winover = int(np.floor(winover))
    #Check the winleght and overlap 
    if isinstance(winlen,np.ndarray) and winlen.ndim > 0:
        wl = len(winlen)
    else:
        wl = winlen
    # Obtain the number of lolumns 
    cols = int(np.ceil(len(audio)-winover)/(wl-winover))
    
    # Pad zeros if necessary
    if np.remainder(len(audio),wl != 0):
        audio = np.pad(audio,(0, (cols *wl) -len(audio)))
    # Zeros matrix
    segmentated = np.zeros((wl,cols))
    
    # Segmentation
    sel = np.arange(wl).reshape(-1,1)
    step = np.arange(0, (cols-1) * (wl-winover)+1, wl - winover)
    
    segmentated[:, :] = audio[sel + step]

    #Alpy window
    if isinstance(winlen, np.ndarray) and winlen.ndim > 0:
        segmentated  *=  winlen[:,np.newaxis]
    return segmentated

"""Zero crossing rate function"""
def ZCR(audio, winlen, winover):
    segmented_temp = segmentation(audio, winlen, winover)
    #segmented_temp = segmented.copy()
    segmented_temp[segmented_temp>=0] = 1
    segmented_temp[segmented_temp<0] = -1
    
    segmented_temp = np.abs(segmented_temp[:-1,:] - segmented_temp[1:, :])

    zcr = (np.sum(segmented_temp == 2, axis = 0) / (segmented_temp.shape[0] + 1)).reshape(1,-1)
    return zcr

def STE(audio, winlen, winover):
    """
    Compute Short-Time Energy (STE) for each frame.
    """
    segmented_ste = segmentation(audio, winlen, winover)
    ste = np.sum((np.abs(segmented_ste))**2, axis=0).reshape(1, -1)
    return ste

def compute_dynamic_threshold(ste, W=2):
    """
    Compute dynamic threshold T_E based on histogram and local maxima.
    """
    # Compute histogram of STE values
    hist, bin_edges = np.histogram(ste, bins=50, density=True)  # 50 bins for resolution
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(hist)
    
    if len(peaks) < 2:
        print("Warning: Less than two peaks found in STE histogram.")
        return np.percentile(ste,75)  # Fallback: Use mean STE if not enough peaks
    
    # Get first and second local maxima positions
    sorted_peaks = sorted(peaks[:2])  # Ensure they are in order
    M1 = bin_edges[sorted_peaks[0]]
    M2 = bin_edges[sorted_peaks[1]]

    # Compute threshold
    T_E = (W * M1 + M2) / (W + 1)
    
    return T_E



def vuv_detector(audio, fs, zcr, ste):
    """
    Voiced/unvoiced detection using ZCR, STE, and pitch.
    Instead of interpolating ZCR/STE, this resamples pitch to match their length.
    """
    sTHR1 = float('-inf')
    plim = [75, 400]  # Pitch limits
    winlen = 512 / fs

    # Compute dynamic threshold for STE
    T_E = compute_dynamic_threshold(ste, W=2)

    # Compute pitch using SWIPE'
    pitch, t, s = sw.swipep(audio, fs, np.array(plim), winlen, 1/96, 0.1, sTHR1)

    # Resample pitch to match ZCR/STE length
    num_frames = zcr.shape[1]  # Number of frames in ZCR/STE
    pitch_resampled = resample(pitch, num_frames)  # Resample pitch

    # Initialize VUV array
    vuv = np.zeros_like(pitch_resampled)

    # Apply voiced/unvoiced decision rules
    for i in range(num_frames):
        if zcr[0, i] < 0.1 and ste[0, i] > T_E:
            vuv[i] = 1  # Voiced
        elif 0.1 < zcr[0, i] < 0.3 and ste[0, i] < T_E:
            vuv[i] = -1 if pitch_resampled[i] <= 78 else 0  # Silence or Unvoiced

    return vuv, pitch_resampled, zcr, ste

def plot_waveform_vuv(audio, fs, vuv, frame_shift):
    """
    Plots the audio waveform and the corresponding voiced/unvoiced decisions (VUV) aligned in time.

    Parameters:
    - audio: The raw audio waveform.
    - fs: Sampling rate of the audio.
    - vuv: Voiced/unvoiced decisions (same length as STE/ZCR).
    - frame_shift: The time duration of one frame (based on window size and overlap).
    """

    # Time vector for audio waveform
    time_audio = np.linspace(0, len(audio) / fs, len(audio))

    # Time vector for VUV decisions (aligned with STE/ZCR frames)
    num_frames = len(vuv)
    time_vuv = np.arange(num_frames) * frame_shift  # Each frame corresponds to a step in time

    # Convert VUV into a visual representation (scale to waveform amplitude)
    vuv_plot = (vuv * np.max(audio) * 0.8)  # Scale to 80% of max amplitude

    # Plot waveform and VUV decision
    plt.figure(figsize=(10, 4))
    plt.plot(time_audio, audio, label="Waveform", alpha=0.7)
    plt.step(time_vuv, vuv_plot, label="Voiced/Unvoiced (scaled)", where="mid", linewidth=2, color="r")

    # Labels and legend
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform and Voiced/Unvoiced Decision")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
        
    file_path  = "K1003_0.0_1.wav"
    audio, fs = librosa.load(file_path, sr=None)  # Load audio
    winlen = np.hamming(512)
    winover = 256
    #seg = segmentation(audio, winlen, winover)
    zcr = ZCR(audio, 512, winover)
    ste = STE(audio, winlen, winover)
    t_e = compute_dynamic_threshold(ste,2)
    vuv, p, zcr_i, ste_i,= vuv_detector(audio, fs, zcr, ste)
    #plt.plot(audio)
    # Compute frame shift (assuming fixed window and overlap)
    frame_shift = (512 - 256) / fs  # Frame step in seconds (winlen - winover)

    # Call the function to plot
    plot_waveform_vuv(audio, fs, vuv, frame_shift)
    """
    plt.figure(1)
    plt.plot(zcr.T)


    plt.figure(3)
    plt.plot(ste.T)
    plt.figure(4)
    plt.plot(ste_i.T)
    plt.figure(5)
    plt.plot(zcr_i.T)

    plt.figure(6)
    plt.plot(p)
    plt.plot(vuv*300)
    plt.show(block = True)
    """   