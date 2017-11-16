import numpy as np
from scipy.io.wavfile import read


"""
Function to convert wav file into feature vector.

Split song into quarters, only analyze 2nd and 3rd quarters.

Features Returned:
	- mean of absolute value of amplitude data
"""
def extract_features_from_wav(file_name):
	raw = read(file_name, 'rb')
	audio = raw[1]
	audio = audio[:,0:1]
	print audio.shape

	amplitude_abs_value = np.absolute(audio)

	mean = np.mean(amplitude_abs_value)
	std = np.std(amplitude_abs_value)

	
	




extract_features_from_wav('test_Everyday.wav')

