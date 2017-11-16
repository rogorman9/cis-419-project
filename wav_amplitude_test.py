import wave 
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

raw = read('test_Everyday.wav', 'rb')
audio = raw[1]

print type(audio), len(audio), audio.shape
print len(audio) / 4, len(audio) * 2 / 4


n,d = audio.shape

# for i in xrange(n):
# 	row = audio[i]
# 	if row[0] or row[1]:
# 		print i, row


















