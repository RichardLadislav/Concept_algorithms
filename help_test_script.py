import numpy as np

vec  = 2, 4, 3, 8, 1, 6, 9
vec_conv =  np.convolve(vec, np.ones(5)/5, 'valid')
vec_diff = np.sum(np.abs(vec[2:-2]-vec_conv))
vec_nominator = vec_diff / (np.size(vec)-1)
#print(vec_conv)
vec_denominator = np.mean(vec)
apq3 = vec_nominator/vec_denominator *100
#print(vec_denominator)
print(apq3)
"VSECKO FAKIN FUNGUJEEEEEE"
