import numpy as np
from scipy.io.wavfile import read


"""
Function to convert wav file into feature vector.

Features Returned:
	- mean of absolute value of amplitude data
	- std of absolute value of amplitude data
	- variance of absolute value of amplitude data
	- we will split the song into 100 segments, take the mean amplitude for each segment, and take the std of these 100 means
		- this will give us a good measure of the consistency of the song volume (for example, if there are loud choruses and soft verses)
"""
def extract_features_from_wav(file_name):
	raw = read(file_name, 'rb')
	audio = raw[1]
	print audio.shape

	amplitude_abs_value = np.absolute(audio)

	# under25 = 0
	# between25and100 = 0
	# zero = 0
	# for indx,val in enumerate(amplitude_abs_value):
	# 	if val < 25 and val > 0:
	# 		under25 += 1
	# 	elif val > 24 and val < 101:
	# 		between25and100 += 1
	# 	elif val < 1:
	# 		zero += 1
	# print under25, between25and100, zero

	mean = np.mean(amplitude_abs_value)
	std = np.std(amplitude_abs_value)
	var = np.var(amplitude_abs_value)

	avg_of_100_segments = np.array([])

	print mean, std, var

	




extract_features_from_wav('data/metal/cleaned/Black_Sabbath_War_Pigs.wav')

