import numpy as np
from alex_generate_MFCC import gen_MFCC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


classical_path = os.path.join("data", "classical")
metal_path = os.path.join("data", "metal")

def tune_params(model, winlen_range, winstep_range, nfft_range, numcep_range):
	"""
	Tunes the parameters winlen, winstep, nfft, and numcep

	Returns:
		Best values for winlen, winstep, nfft, and numcep, all in a tuple
	"""

	best_accuracy = 0
	best_params = (0, 0, 0, 0)
	for winl in winstep_range:
		for wins in winstep_range:
			for nf in nfft_range:
				for numc in numcep_range:
					X = []
					y = []

					for audio_file in os.listdir(classical_path):
						test = gen_MFCC(os.path.join(classical_path, audio_file), winlen=winl, winstep=wins, nfft=nf, numcep=numc)
						X.append(gen_MFCC(os.path.join(classical_path, audio_file)).flatten().tolist())
						y.append("classical")

					for audio_file in os.listdir(metal_path):
						X.append(gen_MFCC(os.path.join(metal_path, audio_file)).flatten().tolist())
						y.append("metal")

					X = np.array(X)
					X_train, X_test, y_train, y_test = train_test_split(X, y)

					scores = []
					for i in range(10):
						score = cross_val_score(model, X, y, cv=5)
						scores += list(score)
					avg_cv_accuracy = np.mean(scores)
					if avg_cv_accuracy > best_accuracy:
						best_accuracy = avg_cv_accuracy
						best_params = (winl, wins, nf, numc)
						
	print best_params, best_accuracy
	return best_params



















