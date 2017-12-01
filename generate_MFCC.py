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
	begin_slice = n / 2 - 500000
	end_slice = n / 2 + 500000
	middle_slice = audio[begin_slice:end_slice]
	# print(middle_slice.shape)
	return middle_slice


def gen_MFCC(file_name):
	# test with whole song and rate
	rate, audio_signal = read(file_name)
	print "Shape: ", audio_signal.shape
	begin_slice = audio_signal.shape[0] / 2 - 3500000
	end_slice = audio_signal.shape[0] / 2 + 3500000
	sig = audio_signal[begin_slice:end_slice]
	mfcc_feat = mfcc(sig, rate, winlen=0.25, winstep=0.1,nfft=11025)

	# test with middle slice of specified size
	# audio_signal = extract_features_MFCC(file_name)
	# mfcc_feat = mfcc(audio_signal)
	
	return mfcc_feat


def gen_data_from_MFCC():
	MFCC = gen_MFCC('data/metal/Black_Sabbath_Paranoid.wav')
	print(MFCC.shape)
	print(np.mean(MFCC[1]))
	MFCC2 = gen_MFCC('data/metal/Black_Sabbath_War_Pigs.wav')
	# print(MFCC2.shape)
	# print(MFCC2[2])
	print(np.mean(MFCC2[1]))
	# MFCC3 = gen_MFCC('data/classical/Sprint_Allegro.wav')
	MFCC4 = gen_MFCC('data/classical/Toccata.wav')


gen_data_from_MFCC()
