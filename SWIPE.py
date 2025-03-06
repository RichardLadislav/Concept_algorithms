import pandas as pd
import numpy as np
from scipy import io, integrate, linalg,  signal
from scipy.sparse.linalg import cg, eigs
import math as m
import sympy as sym 
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
import librosa 
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os
# Check i  f 'plim' exists and is not None or empty, else set default value
def swipep(x,fs,plim,dt,dlog2p,dERBs,sTHR):
  #  plim = [30, 5000] if 'plim' not in locals() or not plim else plim

    # Check if 'dt' exists and is not None or empty, else set default value
   # dt = 0.01 if 'dt' not in locals() or not dt else dt

    # Check if 'dlog2p' exists and is not None or empty, else set default value
   # dlog2p = 1/96 if 'dlog2p' not in locals() or not dlog2p else dlog2p

    # Check if 'dERBs' exists and is not None or empty, else set default value
   # dERBs = 0.1 if 'dERBs' not in locals() or not dERBs else dERBs

    # Check if 'sTHR' exists and is not None or empty, else set default value
   # sTHR = float('-inf') if 'sTHR' not in locals() or not sTHR else sTHR
    
    t =np.arange( 0, len(x)/fs, dt)#[:,np.newaxis]  time vektor

    dc = 4 #Hop size 
    K = 2 #Parameter for size window

    #Define pitch candidates

    log2pc = np.arange(np.log2(plim[0]), np.log2(plim[1]), dlog2p).reshape(-1, 1) 
    pc = 2 ** log2pc
    S = np.zeros((len(pc),len(t))) # Pitch candidate strenght matrix

    # Determine PW - WSs
    divFS = [fs / x for x in plim] #variable so I can divide by list
    logWs = [round(m.log2(4 * K * df)) for df in divFS]
    #ws_arg =  np.arange(logWs[0], logWs[1], -1)
    ws = 2**  np.arange(logWs[0], logWs[1], -1)
    pO = 4 * K * fs / ws

    # Determine window sizes used by each pitch candidate
    d =  1 + log2pc - m.log2(4*K*fs/ws[0])
    # Create ERBs spaced frequencies (in Hertz)
    
    fERBs = erbs2hz(np.arange(hz2erbs(pc[0]/4), hz2erbs(fs/2),dERBs))[:,np.newaxis]
    for i in range(len(ws)):
        dn = round(dc * fs / pO[0]) #Hop size in samples
        # Zero pad signal
        xzp = np.concatenate([np.zeros((ws[i]//2,)), x.flatten(), np.zeros((dn + ws[i]//2,))])
        # Compute spectrum
        w = np.hanning(ws[i]) # Hann window
        o = max(0, round(ws[i] - dn))
        f, ti, X = spectrogram(xzp, fs=fs, window=w, nperseg=ws[i], noverlap=o, mode='complex') 
        X = X *np.sum(w)#*10E12
        # Interpolate at eqidistant ERBs steps
        # Perform interpolation
        # TO DO: ferb je hodnota musim posilat poradi prvku v liste 
        interp_func = CubicSpline(f, np.abs(X), extrapolate=False)
        # Calculate the interpolated magnitudes
        M = np.maximum(0, interp_func(fERBs) )  # Ensure non-negative values
        M = np.squeeze(M)# 
        #M = np.maximum(0, [interp_func(ferbs) for ferbs in fERBs] )  # Ensure non-negative values

        #M = np.maximum(0, [interp_func(ferbs) for ferbs in range(fERBs.shape[0])] )  # Ensure non-negative values
        L = [np.sqrt(ms) for ms in M]# Loudness
        # Select candidates that use this window size 
        # Loop over window 
        # Select candidates that use this window size
        if i == len(ws) - 1:
             j = np.where(d - i > -1)[0]
             k = np.where(d[j] - i < 0)[0]
        elif i == 0:
             j = np.where(d - i < 1)[0]
             k = np.where(d[j] - i > 0)[0]
        else:
             j = np.where(np.abs(d - i) < 1)[0]
             k = np.arange(len(j))   


         # Pitch strength for selected candidates
        Si = pitchStrengthAllCandidates(fERBs, L, pc[j])


        # Interpolate at desired times
        if Si.shape[1] > 1:

           Si = np.nan_to_num(Si, nan=0.0, posinf=0.0, neginf=0.0)
           interp_func = CubicSpline(ti, Si.T, extrapolate = False)


           #Si = [interp_func(i) for i in t]

           Si = interp_func(t) 

        else:

           Si = np.full((len(Si), len(t)), np.nan)


        # Calculate lambda and mu for weighting
        lambda_ = d[j[k]] - i

        mu = np.ones(j.shape).T

        mu[k] = 1 - np.abs(lambda_.T)

        # Update pitch strength matrix

        #help_dimensions= np.outer(mu.shape[0],np.ones(Si.shape[1]))*Si
        S[j, :] += np.outer(mu, np.ones(Si.T.shape[1])) * Si.T
        #S[j, :] += (mu * Si.T).T

    # Initialize pitch and strength ys with NaN
    p = np.full((S.shape[1], 1), np.nan)
    s = np.full((S.shape[1], 1), np.nan)

    # Loop over each time frame
    for j in range(S.shape[1]):
        # Find the maximum strength and its index
        s[j], i = np.max(S[:, j]), np.argmax(S[:, j])
    
        # Skip if the strength is below the threshold
        if s[j] < sTHR:
            continue
    
        # Handle boundary cases
        if i == 0 or i == len(pc) - 1:
            p[j] = pc[0]
        else:
             # Use neighboring points for interpolation
            I = np.arange(i-1, i+2)  # Indices for 3-point interpolation
            tc = 1.0 / pc[I]  # Convert pitch candidates to periods
            ntc = (tc / tc[1] - 1) * 2 * np.pi  # Normalize periods
        
            # Perform parabolic interpolation using polyfit
            S_help = S[I, j]
            ntc_help = np.squeeze(ntc)
            c = np.polyfit(np.squeeze(ntc), S_help, 2)
        
            # Generate fine-tuned frequency candidates for interpolation
            ftc = 1.0 / 2.0**np.arange(np.log2(pc[I[0]]), np.log2(pc[I[2]]) + 1/12/64, 1/12/64)
            nftc = (ftc / tc[1] - 1) * 2 * np.pi  # Normalize fine-tuned candidates Use the interpolated polynomial to find the fine-tuned maximum
            s[j], k = np.max(np.polyval(c, nftc)), np.argmax(np.polyval(c, nftc))
        
            # Convert the fine-tuned result back to pitch
            p[j] = 2 ** (np.log2(pc[I[0]]) + (k - 1) / (12 * 64))

    return p, t, s

def pitchStrengthAllCandidates(f,L,pc):

    """
    Calculate the pitch strength for all candidates.

    Parameters:
    f  -- Frequency y
    L  -- Loudness y
    pc -- Pitch candidates

    Returns:
    S  -- Pitch salience matrix
    """
    
    with np.errstate(divide= 'ignore', invalid = 'ignore'):
     
        L =  np.array(L)
        L = L / np.sqrt(np.sum(L ** 2, axis = 0, keepdims = True))
    #Create pitch salience matrix
    S = np.zeros((len(pc), L.shape[1]))

    for j in range(len(pc)):

        S[j,:] = pitchStrengthOneCandidate(f, L, pc[j])

    return S

def  pitchStrengthOneCandidate(f,L,pc):
    """
    Calculate the pitch strength for one pitch candidate.

    Parameters:
    f  -- Frequency y
    L  -- Loudness y
    pc -- Pitch candidate

    Returns:
    S  -- Pitch strength for this candidate
    """
    n = int(np.fix(f[-1]/pc - 0.75)) # Number of harmonics
    k = np.zeros(f.shape) # Kernel
    q = f / pc #Normalize frequency  w.r.t candidate

    for i in [1] + list(sym.primerange(n)):
        a = np.abs(q-i)
        p = a < 0.25 #Peaks weights
        k[p] = np.cos(2*np.pi * q[p]) /2

    k = k * np.sqrt(1. /f) # Aplly envelope
    k = k / np.linalg.norm(k[k > 0]) # K+-normalize kernel

    S = np.dot(k.T, L)

    return S

def hz2erbs(hz):
    """Convert frequency in Hz to ERBs."""
    return 21.4* np.log10(1+ hz / 229)

def erbs2hz(erbs):
    """Convert ERBs to frequency in Hz."""
    return (10** (erbs / 21.4)-1) * 229

# Testing part
def main():
    #audiorad
   # Load audio file
    #os.chdir("C:\\Users\\Richard Ladislav\\Desktop\\final countdown\\DP-knihovna pro parametrizaci reci - kod\\concepts_algorithms")
    os.chdir("C:\\Users\\Richard Ladislav\\Desktop\\final countdown\\DP-knihovna pro parametrizaci reci - kod\\statistic_evaluation\\perlonged_E\\P_e")
    filename ="P2111_7.1-2-e_1.wav" 

    #filename ="vowel_e_test.wav" 
    #filename ="vowel_e_test.wav" 
    #filename ="vowel_e_test.wav" 
    #filename = librosa.ex('trumpet')
    x, Fs = librosa.load(filename, sr=None)  # Load audio, maintain original sampling rate fmin = 75
    # Call the swipep-like function
    sTHR1 = float('-inf')
    plim = [75,300]#pitch limitation
    #TODO: upravit funkciu aby brala premenny pocet argumentov 
    p, t, s = swipep(x, Fs, np.array(plim), 0.035, 1/96,0.1,sTHR1)
    pitch_in_time = np.column_stack((t,p))
    df = pd.DataFrame(pitch_in_time, columns=["Time(s)"  , "Pitch "])

    # Save to CSV
    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepvowel_e_testts_algorithms//pitch_P1021_7.1-1-a_1.csv", index=False)
    #df.to_csv("C:\\Users\\Richard Ladislav\\Desktop\\final countdown\\DP-knihovna pro parametrizaci reci - kod\\statistic_evaluation\\perlonged_E\\SWIPE_e\\P2111_7.1-2-e_1_250hz.wav_swipe.csv", index=False)
    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//pitch_csv_vowel_e.csv", index=False)
    
    # Plot the pitch
    plt.plot(1000 * t, p)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Pitch Estimation using swipep-like Algorithm')
    plt.show(block=True)

if __name__ == "__main__":
  
    main()
