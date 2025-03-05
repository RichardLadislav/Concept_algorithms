import numpy as np
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
        audio = np.pad(audio,(0, cols *wl -len(audio)))
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
def ZCR(segmented):
    segmented_temp = segmented.copy()
    segmented_temp[segmented_temp>=0] = 1
    segmented_temp[segmented_temp<0] = -1
    
    segmented_temp = np.abs(segmented_temp[:-1,:] - segmented_temp[1:, :])

    zcr = (np.sum(segmented_temp == 2, axis =0) / (segmented_temp.shape[0] + 1)).reshape(1,-1)
    return zcr
def STE(segmented):
    ste = np.sum(np.power(np.abs(segmented),2))
    return ste

def vuv_detector(audio, segmented, fs, zcr, ste):
    
    sTHR1 = float('-inf')
    plim = [75,400]#pitch limitation
    winlen = 512/fs
    pitch, t ,s = sw.swipep(segmented,fs,np.array(plim),winlen , 1/96,0.1,sTHR1)
    vuv = np.zeros(np.shape(segmented))
    # Branching
    return pitch
    #vuv = 
if __name__ == "__main__":
        
    file_path  = "K1003_7.1-2-a_1.wav"
    audio, fs = librosa.load(file_path, sr=None)  # Load audio
    winlen = np.hamming(512)
    winover = 256
    seg = segmentation(audio, winlen, winover)
    zcr = ZCR(seg)
    ste = STE(seg)
    vuv = vuv_detector(audio, seg, fs, zcr, ste)
    plt.plot(zcr)
    plt.show()