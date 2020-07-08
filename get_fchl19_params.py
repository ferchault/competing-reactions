#!/usr/bin/env python3

import sys
import random
from datetime import datetime
import numpy as np
from copy import deepcopy
import qml
from qml.kernels import get_local_symmetric_kernels
from qml.kernels import get_local_kernels
from qml.representations import generate_fchl_acsf
from qml.math import cho_solve
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
		hof = float(tokens[2])

#		if hof < 100 and hof > 0: energies[xyz_name] = hof
		energies[xyz_name] = hof

	return energies

if __name__ == "__main__":
	# Parse file containing PBE0/def2-TZVP heats of formation and xyz filenames
	data  = get_energies("train_sn2.txt")
	# Generate a list of fml.Molecule() objects
	mols = []

	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
		mol.read_xyz("/home/heinen/PhD/projects/sn_e/paper/raw_data/xyz/" + xyz_file)
		mol.properties = data[xyz_file]
		mols.append(mol)

	sigma = [0.1 * 2**i for i in range(10)]#[20.48]
	ll = [1e-1, 1e-3, 1e-5]

	x = []
	q = []
	x_test = []
	q_test = []

	lst = [1,5,6,7,8,9,17,35]
	for mol in mols:
		x1 = generate_fchl_acsf(mol.nuclear_charges, mol.coordinates, gradients=False, pad=21, elements=lst)
		x.append(x1)
		q.append(mol.nuclear_charges)

	X		= np.array(x)
	Q		= np.asarray(q)

	K      = get_local_symmetric_kernels(X, Q, sigma)
	Yprime = np.asarray([ mol.properties for mol in mols ])

	kf = KFold(n_splits=5)
	kf.get_n_splits(X)

	for j in range(len(sigma)):
		for l in ll:
			maes = []
			for train_index, test_index in kf.split(X):
				K_train = K[j][train_index][:,train_index]
				K_test  = K[j][train_index][:,test_index]

				Y = Yprime[train_index]

				C = deepcopy(K_train)
				C[np.diag_indices_from(C)] += l

				alpha = cho_solve(C, Y)

				Yss  = np.dot(K_test.T, alpha)
				diff = Yss- Yprime[test_index]
				mae  = np.mean(np.abs(diff))
				maes.append(mae)

			print( str(l) + ',' + str(sigma[j]) + "," + str(sum(maes)/len(maes)) )
