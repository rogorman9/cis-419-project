import numpy as np 
from extract_features_from_wav import extract_features_from_wav

# First, we must load in the data
metal1 = extract_features_from_wav('data/metal/Black_Sabbath_Iron_Man.wav')
metal1 = np.append(metal1, "metal")
metal1 = metal1.reshape((1, len(metal1)))[0]
metal2 = extract_features_from_wav('data/metal/Black_Sabbath_Paranoid.wav')
metal2 = np.append(metal2, "metal")
metal2 = metal2.reshape((1, len(metal2)))[0]
metal3 = extract_features_from_wav('data/metal/Black_Sabbath_War_Pigs.wav')
metal3 = np.append(metal3, "metal")
metal3 = metal3.reshape((1, len(metal3)))[0]
metal4 = extract_features_from_wav('data/metal/Holy_Wars_The_Punishment_Due.wav')
metal4 = np.append(metal4, "metal")
metal4 = metal4.reshape((1, len(metal4)))[0]
metal5 = extract_features_from_wav('data/metal/Iron_Maiden_Hallowed_Be_Thy_Name.wav')
metal5 = np.append(metal5, "metal")
metal5 = metal5.reshape((1, len(metal5)))[0]
metal6 = extract_features_from_wav('data/metal/Iron_Maiden_The_Trooper.wav')
metal6 = np.append(metal6, "metal")
metal6 = metal6.reshape((1, len(metal6)))[0]
metal7 = extract_features_from_wav('data/metal/Judas_Priest_Painkiller.wav')
metal7 = np.append(metal7, "metal")
metal7 = metal7.reshape((1, len(metal7)))[0]
metal8 = extract_features_from_wav('data/metal/Metallica_Enter_Sandman.wav')
metal8 = np.append(metal8, "metal")
metal8 = metal8.reshape((1, len(metal8)))[0]
metal9 = extract_features_from_wav('data/metal/Metallica_Master_Of_Puppets.wav')
metal9 = np.append(metal9, "metal")
metal9 = metal9.reshape((1, len(metal9)))[0]
metal10 = extract_features_from_wav('data/metal/Metallica_One.wav')
metal10 = np.append(metal10, "metal")
metal10 = metal10.reshape((1, len(metal10)))[0]

classical1 = extract_features_from_wav('data/classical/Bride.wav')
classical1 = np.append(classical1, "classical")
classical1 = classical1.reshape((1, len(classical1)))[0]
classical2 = extract_features_from_wav('data/classical/Canon_in_D_Major.wav')
classical2 = np.append(classical2, "classical")
classical2 = classical2.reshape((1, len(classical2)))[0]
classical3 = extract_features_from_wav('data/classical/Eine_Kleine_Nachtmusik.wav')
classical3 = np.append(classical3, "classical")
classical3 = classical3.reshape((1, len(classical3)))[0]
classical4 = extract_features_from_wav('data/classical/Fifth.wav')
classical4 = np.append(classical4, "classical")
classical4 = classical4.reshape((1, len(classical4)))[0]
classical5 = extract_features_from_wav('data/classical/Hallelujah.wav')
classical5 = np.append(classical5, "classical")
classical5 = classical5.reshape((1, len(classical5)))[0]
classical6 = extract_features_from_wav('data/classical/Moonlight_Sonata.wav')
classical6 = np.append(classical6, "classical")
classical6 = classical6.reshape((1, len(classical6)))[0]
classical7 = extract_features_from_wav('data/classical/Mountain_King.wav')
classical7 = np.append(classical7, "classical")
classical7 = classical7.reshape((1, len(classical7)))[0]
classical8 = extract_features_from_wav('data/classical/Ode_to_Joy.wav')
classical8 = np.append(classical8, "classical")
classical8 = classical8.reshape((1, len(classical8)))[0]
classical9 = extract_features_from_wav('data/classical/Sprint_Allegro.wav')
classical9 = np.append(classical9, "classical")
classical9 = classical9.reshape((1, len(classical9)))[0]
classical10 = extract_features_from_wav('data/classical/Toccata.wav')
classical10 = np.append(classical10, "classical")
classical10 = classical10.reshape((1, len(classical10)))[0]


data = np.stack((metal1, metal2, metal3, metal4, metal5, metal6, metal7, metal8, metal9, metal10, 
				classical1, classical2, classical3, classical4, classical5, classical6, classical7,
				classical8, classical9, classical10), axis=0)
print data, data.shape

np.savetxt('metal_classical_data.dat', data, delimiter=",", fmt='%s')


