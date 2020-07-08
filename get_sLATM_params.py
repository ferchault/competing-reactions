#!/usr/bin/env python3

import sys
import time
import random
from datetime import datetime
import numpy as np
from copy import deepcopy
import qml
from qml.math import cho_solve
from qml.representations import *
#from qml.wrappers import get_atomic_kernels_gaussian 
from qml.kernels import get_local_kernels_gaussian
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
import itertools
from time import time
from sklearn.model_selection import KFold

# Function to parse datafile to a dictionary
def get_energies(filename):
	""" Returns a dictionary with heats of formation for each xyz-file.
	"""

	f = open(filename, "r")
	lines = f.readlines()
	f.close()

	energies = dict()

	for line in lines:
		tokens = line.split()

		xyz_name = tokens[1]
		Ebind = float(tokens[2])

#		if Ebind < 100 and Ebind > 0: energies[xyz_name] = Ebind
		energies[xyz_name] = Ebind

	return energies

if __name__ == "__main__":
	# file containing xyz file names (without extension .xyz) and according proerties 
	data  = get_energies("train_sn2.txt")

	# Generate a list of qml.Compound() objects
	mols = []
	mols_test = []

	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
		mol.read_xyz("/home/heinen/PhD/projects/sn_e/paper/raw_data/xyz/" + xyz_file)
		mol.properties = data[xyz_file]
		mols.append(mol)

	mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in mols])

	for mol in mols:
		mol.generate_slatm(mbtypes, local=False)

	X = np.asarray([ mol.representation for mol in mols ])

	sigma = [3.2,6.4,12.8,25.6,102.4,204.8,409.6,819.2,1638.4]
	ll = [1e-1, 1e-3, 1e-5, 1e-7]

	Yprime = np.asarray([ mol.properties for mol in mols])

	kf = KFold(n_splits=5)
	kf.get_n_splits(X)

	print(kf)
	for j in range(len(sigma)):
		for l in ll:
			maes = []
			for train_index, test_index in kf.split(X):
				K      = gaussian_kernel(X[train_index], X[train_index], sigma[j])
				K_test = gaussian_kernel(X[train_index], X[test_index],  sigma[j])

				Y = Yprime[train_index]

				C = deepcopy(K)
				C[np.diag_indices_from(C)] += l

				alpha = cho_solve(C, Y)

				Yss  = np.dot(K_test.T, alpha)
				diff = Yss- Yprime[test_index]
				mae  = np.mean(np.abs(diff))
				maes.append(mae)

			print( str(l) + ',' + str(sigma[j]) + "," + str(sum(maes)/len(maes)) )
