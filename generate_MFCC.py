# from extract_features_from_wav import *
import numpy as np
from scipy.io.wavfile import read
from python_speech_features import mfcc


def gen_MFCC(file_name):
	# test with whole song and rate
	rate, audio_signal = read(file_name)
	# Alter size of slice, smaller sizes may be quicker and give higher accuracy but could be prone to overfitting
	# print('name: ', file_name, 'length of song: ', audio_signal.shape[0])
	begin_slice = audio_signal.shape[0] / 2 - 3500000
	end_slice = audio_signal.shape[0] / 2 + 3500000
	# begin_slice = audio_signal.shape[0] / 2 - 2000000
	# end_slice = audio_signal.shape[0] / 2 + 2000000
	sig = audio_signal[begin_slice:end_slice]
	# winlen is 
	# mfcc_feat = mfcc(sig, rate)
	mfcc_feat = mfcc(sig, rate, winlen=0.25, winstep=0.1, numcep=13, nfft=11025)
	# mfcc_feat = mfcc(sig, rate, winlen=0.1, winstep=0.005, numcep=13, nfft=11025)

	# test with middle slice of specified size
	
	return mfcc_feat