from scipy.linalg import solve_toeplitz
import sys
import pandas as pd
import numpy as np
import wave
import math as m
from scipy.signal import lfilter
import scipy
import librosa as lib

def calculate_lpc(x, order):
    """Calculate LPC coefficients using the Levinson-Durbin recursion."""
    r = np.correlate(x, x, mode='full')[-len(x):]  # Autocorrelation
    R = r[:order]  # Autocorrelation matrix
    r = r[1:order+1]
    return np.append(1, -solve_toeplitz(R, r))  # LPC coefficients

def get_formants(file_path):
    # Read the file using librosa
    x, Fs = lib.load(file_path, sr=None)

    # Get Hamming window
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    
    # Determine LPC order
    ncoeff = int(2 + Fs / 1000)
    
    # Calculate LPC coefficients
    #A = calculate_lpc(x1, order=ncoeff)
    A = calculate_lpc(x1, order=ncoeff)

    # Find roots of the LPC polynomial
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) >= 0]

    # Calculate angles and frequencies
    angz = np.arctan2(np.imag(rts), np.real(rts))
    frqs = sorted(angz * (Fs / (2 * m.pi)))

    return frqs
def main():

    formant_freq = get_formants("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//vowel_e_test.wav")
     # Convert formant frequencies to a DataFrame
    df = pd.DataFrame(formant_freq, columns=["Formant Frequencies (Hz)"])
    
    # Save to CSV
    df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//formant_csv_vowel_e1.csv", index=False)
    print(f"formant frequencies {formant_freq}")

if __name__ == "__main__":
  
    main() 
