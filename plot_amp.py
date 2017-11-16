
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys


plt.rcParams['agg.path.chunksize'] = 10000

spf = wave.open('data/metal/cleaned/Black_Sabbath_War_Pigs.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
fs = spf.getframerate()

#If Stereo
if spf.getnchannels() == 2:
    print 'Just mono files'
    sys.exit(0)


Time=np.linspace(0, len(signal)/fs, num=len(signal))

plt.figure(1)
plt.title('Amplitude vs. time (s)')
plt.plot(Time,signal)
plt.show()