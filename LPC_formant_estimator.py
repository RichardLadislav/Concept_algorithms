import sys
import numpy as np
import wave
import math as m
from scipy.signal import lfilter
import scipy
import librosa as lib


#from scipy.signal import hamming
#from scikits.talkbox import lpc

"""
Estimate formants using LPC.
"""

def get_formants(file_path):

    # Read from file.
    spf, Fs = lib.load(file_path, sr = None) # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav

    # Get file as np array.
    '''
    x = spf.readframes(-1)
    x = np.fromstring(x, 'Int16')
    '''
    x = spf
    # Get Hamming window.
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    x1 = x1.reshape(np.size(x1),1)
    # Get LPC.
    #Fs = spf.getframerate()
    ncoeff = 2 + Fs / 1000
    A = lib.lpc(x1, order=int(ncoeff))
    #y, sr = lib.load(lib.ex('libri1'), duration=0.020)
    #A = lib.lpc(y, order=2)
    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Get angles.
    angz = np.arctan2(np.imag(rts), np.real(rts))

    # Get frequencies.
    #Fs = spf.getframerate()
    frqs = sorted(angz * (Fs / (2 * m.pi)))

    return frqs

# get_formants(sys.argv[1])   
def main():

    formant_freq = get_formants("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//vowel_e_test.wav")
    print(f"formant frequencies {formant_freq}")

if __name__ == "__main__":
  
    main() 