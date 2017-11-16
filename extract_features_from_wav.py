import numpy as np
from scipy.io.wavfile import read


def extract_features_from_wav(file_name):
	"""
	Function to convert wav file into feature vector.

	Features Returned:
		- mean of absolute value of amplitude data
		- std of absolute value of amplitude data
		- variance of absolute value of amplitude data
		- we will split the song into 100 segments, take the mean amplitude for each segment, and take the std of these 100 means
			- this will give us a good measure of the consistency of the song volume (for example, if there are loud choruses and soft verses)
	"""

	raw = read(file_name, 'rb')
	audio = raw[1]

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
	for i in xrange(100):
		amplitude_slice = amplitude_abs_value[len(amplitude_abs_value)*i/100:len(amplitude_abs_value)*(i+1)/100]
		m = np.mean(amplitude_slice)
		avg_of_100_segments = np.append(avg_of_100_segments, m)

	std_of_100_avgs = np.std(avg_of_100_segments)

	return (str(file_name), mean, std, var, std_of_100_avgs)

	




# extract_features_from_wav('data/metal/Black_Sabbath_War_Pigs.wav')
# extract_features_from_wav('data/metal/Black_Sabbath_Iron_Man.wav')
# extract_features_from_wav('data/metal/Black_Sabbath_Paranoid.wav')
# extract_features_from_wav('data/metal/Holy_Wars_The_Punishment_Due.wav')
# extract_features_from_wav('data/metal/Iron_Maiden_Hallowed_Be_Thy_Name.wav')









