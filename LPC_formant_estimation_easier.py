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
    r_full = np.correlate(x, x, mode='full')[-len(x):]  # Full autocorrelation
    if len(r_full) < order:
        raise ValueError("Signal too short for the desired LPC order.")
    
    R = scipy.linalg.toeplitz(r_full[:order])           # Toeplitz matrix
    r_vec = r_full[1:order+1]                           # RHS vector
    coeffs = np.append(1, -np.linalg.solve(R, r_vec))   # LPC coefficients
    return coeffs
'''
def calculate_lpc(x, order):
    """Calculate LPC coefficients using the Levinson-Durbin recursion."""
    r = np.correlate(x, x, mode='full')[-len(x):]  # Autocorrelation
    #R = r[:order]  # Autocorrelation matrix
    R = scipy.linalg.toeplitz(r[:order])

    r = r[1:order+1]
    return np.append(1, -solve_toeplitz(R, r))  # LPC coefficients
'''
def get_formants(file_path):
    # Read the file using librosa

    ncoeff = int(2 + Fs / 1000) # Calculate order of LPC order
    # Get Hamming window
    w = np.hamming(len(x))

    # Apply window and high pass filter
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)
    #TODO: filtering with hamming window before or after SLIDING WINDOW 
    # 
    #  
    dt = 0.01 # duration of time window in seconds TODO: input of function
    samples_dt = dt * Fs  # 
    '''zero matrix intialization'''
    A_rows  = np.floor(len(x1)/samples_dt) # count number of rows for matrices``
    A = np.zeros((int(A_rows),ncoeff+1)) # intialization of A matrix, ncoeff +1 beacause when estimating LPC, theres added 1
    rts = np.zeros((int(A_rows),ncoeff), dtype=complex)
    angz = np.zeros((int(A_rows),ncoeff), dtype=complex)
    frqs = np.zeros((int(A_rows),ncoeff))
    for hop in range(0,int(A_rows*samples_dt),int(samples_dt)):
        
        x1_cut = x1[hop:hop+int(samples_dt)]


        A[int(hop/samples_dt),:] = calculate_lpc(x1_cut, order=ncoeff)
        rts[int(hop/samples_dt),:] = np.roots(A[int(hop/160),:])

        rts[int(hop/samples_dt),:] = np.where(0<=np.imag(rts[int(hop/160),:]),rts[int(hop/160),:],0)
        # Find roots of the LPC polynomial
        #rts[int(hop/160),:]= [r for r in rts[int(hop/160),:] if np.imag(r) >= 0]

        # Calculate angles and frequencies
        angz[int(hop/samples_dt),:] = np.arctan2(np.imag(rts[int(hop/160),:]), np.real(rts[int(hop/160),:]))
        #TODO: 17.11: vyreisit vypisovanie len kladnych realnych casti matice freq
        frqs[int(hop/samples_dt),:] = sorted(np.real(angz[int(hop/160),:]) * (Fs / (2 * m.pi)))

    return frqs
def main():

    formant_freq = get_formants("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//test_samples//P1021_7.1-1-e_1.wav")
     # Convert formant frequencies to a DataFrame
 #   df = pd.DataFrame(formant_freq, columns=["Formant Frequencies (Hz)"])
    
    # Save to CSV
 #   df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//1formants_K1003_7.1-2-a_1.csv", index=False)
    # df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//formant_csv_vowel_e1.csv", index=False)
    print(f"formant frequencies {formant_freq}")

if __name__ == "__main__":
  
    main() 
