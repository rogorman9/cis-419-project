
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





