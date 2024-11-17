from SWIPE import *
from LPC_formant_estimation_easier import *

def main():

    dt = 0.01
    file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//test_samples//P1021_7.1-1-e_1.wav"
    x, Fs = lib.load(file_path, sr=None)
    formant_freq = get_formants(x,Fs,dt)
     # Convert formant frequencies to a DataFrame
    #df = pd.DataFrame(formant_freq, columns=["Formant Frequencies (Hz)"])
    
    # Save to CSV
    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//1formants_K1003_7.1-2-a_1.csv", index=False)
    # df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//formant_csv_vowel_e1.csv", index=False)
    print(f"formant frequencies {formant_freq}")
    # Time vector for each window
    time_vector = np.arange(0, len(formant_freq) * dt, dt)

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

    sTHR1 = float('-inf')
    plim = [75,500]#pitch limitation
    #TODO: upravit funkciu aby brala premenny pocet argumentov 
    p, t, s = swipep(x, Fs, np.array(plim), 0.01, 1/96,0.1,sTHR1)
    pitch_in_time = np.column_stack((t,p))
#    df = pd.DataFrame(pitch_in_time, columns=["Time(s)"  , "Pitch "])
    
    # Save to CSV
#    df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepvowel_e_testts_algorithms//pitch_P1021_7.1-1-a_1.csv", index=False)
    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//test_samples//pitch_csv_vowel_e.csv", index=False)
    #df.to_csv("C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concepts_algorithms//pitch_csv_vowel_e.csv", index=False)
    
    # Plot the pitch
    plt.plot(1000 * t, p)
    plt.xlabel('Time (ms)')
    plt.ylabel('Pitch (Hz)')
    plt.title('Pitch Estimation using swipep-like Algorithm')
    plt.show(block=True)

if __name__ == "__main__":
  
    main() 
