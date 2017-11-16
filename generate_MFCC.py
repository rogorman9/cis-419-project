# from extract_features_from_wav import *
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc


def extract_features_MFCC(file_name):
	''' 
	similar to Alex's extract features, excepts only prepares a middle 
	chunk to test with the MFCC
	'''
	raw = read(file_name, 'rb')
	audio = raw[1]
	audio = audio[:,0:1]
	# print audio.shape
	n, d = audio.shape
	begin_slice = n / 2 - 5000000
	end_slice = n / 2 + 5000000
	middle_slice = audio[begin_slice:end_slice]
	# print(middle_slice.shape)
	return middle_slice

	# amplitude_abs_value = np.absolute(audio)
	# mean = np.mean(amplitude_abs_value)
	# std = np.std(amplitude_abs_value)

def gen_MFCC(file_name):
	
	audio_signal = extract_features_MFCC(file_name)
	print(audio_signal.shape)
	mfcc_feat = mfcc(audio_signal)
	print(mfcc_feat)
	print(mfcc_feat.shape)

MFCC = gen_MFCC('test_Everyday.wav')