# from extract_features_from_wav import *
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc


def gen_MFCC(file_name):
	# test with whole song and rate
	rate, audio_signal = read(file_name)
	# Alter size of slice, smaller sizes may be quicker and give higher accuracy but could be prone to overfitting
	begin_slice = audio_signal.shape[0] / 2 - 20000000
	end_slice = audio_signal.shape[0] / 2 + 20000000
	# print "Shape: ", audio_signal.shape
	begin_slice = audio_signal.shape[0] / 2 - 3500000
	end_slice = audio_signal.shape[0] / 2 + 3500000
	sig = audio_signal[begin_slice:end_slice]
	# winlen is 
	# mfcc_feat = mfcc(sig, rate)
	mfcc_feat = mfcc(sig, rate, winlen=0.25, winstep=0.1, nfft=11025)

	# test with middle slice of specified size
	# audio_signal = extract_features_MFCC(file_name)
	# mfcc_feat = mfcc(audio_signal)
	
	return mfcc_feat