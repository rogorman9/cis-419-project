import numpy as np
from alex_generate_MFCC import gen_MFCC_Tune
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
import os


classical_path = os.path.join("data", "classical")
metal_path = os.path.join("data", "metal")

def tune_params(model, winlen_range=[0.025], winstep_range=[0.01], nfft_range=[11025], numcep_range=[13]):
	"""
	Tunes the parameters winlen, winstep, nfft, and numcep
	Default values are shown above in case we only want to tune one of these four values

	Returns:
		Best values for winlen, winstep, nfft, and numcep, all in a tuple
	"""

	best_accuracy = 0
	best_params = (0, 0, 0, 0)
	cntr = 1
	for winl in winlen_range:
		for wins in winstep_range:
			for nf in nfft_range:
				for numc in numcep_range:

					print "Trial " + str(cntr)

					X = []
					y = []

					for audio_file in os.listdir(classical_path):
						X.append(gen_MFCC_Tune(os.path.join(classical_path, audio_file), winlen=winl, winstep=wins, nfft=nf, numcep=numc).flatten().tolist())
						y.append("classical")

					for audio_file in os.listdir(metal_path):
						X.append(gen_MFCC_Tune(os.path.join(metal_path, audio_file), winlen=winl, winstep=wins, nfft=nf, numcep=numc).flatten().tolist())
						y.append("metal")

					X = np.array(X)

					scores = []
					for i in range(10):
						score = cross_val_score(model, X, y, cv=5)
						scores += list(score)
					avg_cv_accuracy = np.mean(scores)

					print "Accuracy " + str(cntr) + ": " + str(avg_cv_accuracy) + ", Params: " + str((winl, wins, nf, numc))
					cntr += 1

					if avg_cv_accuracy > best_accuracy:
						best_accuracy = avg_cv_accuracy
						best_params = (winl, wins, nf, numc)
						
	print best_params, best_accuracy
	return best_params


if __name__ == "__main__":
	# Tune parameters of the MFCC, in order to give us a good idea of what range these values should lie within

	# Ideal winlen = 0.25
	winlen_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0]
	# Ideal winstep = 0.1
	winstep_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
	# Ideal nfft = 11025
	nfft_range = [10000, 11025, 12500, 15000]
	# Ideal numcep value = 13
	numcep_range = [3, 6, 9, 12, 13, 15, 18, 21]

	model = AdaBoostClassifier()
	tuned_values = tune_params(model, winlen_range, winstep_range, nfft_range, numcep_range)

	print tuned_values

















