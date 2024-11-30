from scipy.linalg import solve_toeplitz
import sys
import pandas as pd
import numpy as np
import wave
import math as m
from scipy.signal import lfilter
import scipy
import librosa as lib
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def calculate_lpc(x, order):
    """Calculate LPC coefficients using the Levinson-Durbin recursion."""
    r_full = np.correlate(x, x, mode='full')[-len(x):]  # Full autocorrelation
    if len(r_full) < order:
        raise ValueError("Signal too short for the desired LPC order.")
    
    R = scipy.linalg.toeplitz(r_full[:order])           # Toeplitz matrix
    r_vec = r_full[1:order+1]                           # RHS vector
    coeffs = np.append(1, -np.linalg.solve(R, r_vec))   # LPC coefficients
    return coeffs

def get_formants(x, Fs, dt, number_of_formants = None):
    """Calculate formant frequencies from LPC coefficients"""
    if number_of_formants:
        ncoeff = int(number_of_formants*2 + 2)  #2*expected number of formants +2
    else:
        ncoeff = 8
    #ncoeff = int(2 + Fs / 1000) # Calculate order of LPC orderVyo
   
    w = np.hamming(len(x)) # Get Hamming window
    # Apply window and high pass filter
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    # Handling of out of boundaries users input
    if int(len(x1)) < dt or dt <= 0: 
        dt =len(x1)/Fs  

    samples_dt = dt * Fs  # 
    '''zero matrix intialization'''
    A_rows  = np.floor(len(x1)/samples_dt) # Count number of rows for matrices``

    A = np.zeros((int(A_rows),ncoeff+1)) # Intialization of A matrix, ncoeff +1 beacause when estimating LPC, theres added 1
    rts = np.zeros((int(A_rows),ncoeff), dtype=complex) # Initialization of rts matrix
    angz = np.zeros((int(A_rows),ncoeff), dtype=complex) # Inicilization of angz matrix
    frqs = np.zeros((int(A_rows),ncoeff)) # IInicilization of angz matrix

    for hop in range(0,int(A_rows*samples_dt),int(samples_dt)):
        
        x1_cut = x1[hop:hop+int(samples_dt)]


        A[int(hop/samples_dt),:] = calculate_lpc(x1_cut, order=ncoeff)
        rts[int(hop/samples_dt),:] = np.roots(A[int(hop/samples_dt),:])

        rts[int(hop/samples_dt),:] = np.where(0<=np.imag(rts[int(hop/samples_dt),:]),rts[int(hop/samples_dt),:],0)
        # Find roots of the LPC polynomial
        # Calculate angles and frequencies
        angz[int(hop/samples_dt),:] = np.arctan2(np.imag(rts[int(hop/samples_dt),:]), np.real(rts[int(hop/samples_dt),:]))
        frqs[int(hop/samples_dt),:] = sorted(np.real(angz[int(hop/samples_dt),:]) * (Fs / (2 * m.pi)))
    
    zero_idx = np.argwhere(np.all(frqs[...,:] == 0, axis =0))
    formants = np.delete(frqs,zero_idx,axis=1)
    
    # Align formants across time
    formants = align_formants(formants)
    if number_of_formants:
        # Adjust for 1-based indexing (e.g., formant 1 is index 0)

        desired_indices = np.arange(0,3,1)
        #desired_indices = np.arange(0,number_of_formants,1)
        #desired_indices = [i -1  for i in desired_formants]
        formants = formants[:, desired_indices]
#        formants = align_formants(formants)
        
    return formants

def align_formants(formants):
    """
    Align formants row by row based on specified conditions.
    
    Parameters:
        formants (np.ndarray): 2D array where rows represent time frames and columns represent formants.
        
    Returns:
        np.ndarray: Adjusted formants matrix with aligned rows.
    """
    aligned_formants = np.copy(formants)  # Create a copy to avoid modifying the original

    num_rows, num_cols = formants.shape

    for r in range(num_rows):
        row = aligned_formants[r, :]
       # TODO: fix aligning algorithm  
        for c in range(num_cols - 2):  # Avoid index out of bounds for c + 2
            # Check condition: Leave row as it is
            if (row[c] != 0 or row[c + 1] == 0) and row[c + 2] >= 700:
                break  # No changes needed; move to the next row
            
            # Check condition: Align whole row to the left
            if row[c] == 0 and row[c + 1] <= 650:
                non_zero_values = row[row > 0]  # Extract all non-zero values
                aligned_formants[r, :] = np.zeros_like(row)  # Reset row
                aligned_formants[r, :len(non_zero_values)] = non_zero_values  # Align to the left
                break  # Alignment done; move to the next row
    
    return aligned_formants

def main():

    dt = 0.00625
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//test_samples//P1021_7.1-1-a_1.wav"
    x, Fs = lib.load(file_path, sr=None)
#    desired_formants = [1,2,3]
    num_of_formants = 10
    formant_freq = get_formants(x,Fs,dt,num_of_formants)
     # Convert formant frequencies to a DataFrame
 
    # Save to CSV
    # df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//formant_csv_vowel_e1.csv", index=False)
    print(f"formant frequencies {formant_freq}")
    # Time vector for each window
    time_vector = np.arange(0, len(formant_freq) * dt, dt)[:formant_freq.shape[0]]
    #columns = ["Time (s)"] + [f"Formant {i+1} (Hz)" for i in range(formant_freq.shape[1])]
    #data = np.column_stack((time_vector, formant_freq))
    #df = pd.DataFrame(data, columns=columns)

    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms///test_samples//formants_prelonged_vowel_a_patient_final.csv", index=False)

    # Plot all formant frequencies over time
    plt.figure(figsize=(10, 6))
    for i in range(formant_freq.shape[1]):  # Iterate over formant frequency columns
        plt.plot(time_vector, formant_freq[:, i], label=f'Formant {i+1}')

    plt.title("Formant Frequencies Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.show(block=True) 

if __name__ == "__main__":
  
    main() 
