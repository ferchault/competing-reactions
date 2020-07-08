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
	data2 = get_energies( "test_sn2.txt")

	# Generate a list of qml.Compound() objects
	mols = []
	mols_test = []

	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
		mol.read_xyz("/home/heinen/PhD/projects/sn_e/paper/raw_data/xyz/" + xyz_file)
		mol.properties = data[xyz_file]
		mols.append(mol)

	for xyz_file in sorted(data2.keys()):
		mol = qml.Compound()
		mol.read_xyz("/home/heinen/PhD/projects/sn_e/paper/raw_data/xyz/" + xyz_file)
		mol.properties = data2[xyz_file]
		mols_test.append(mol)

	mbtypes = get_slatm_mbtypes([mol.nuclear_charges for mol in mols+mols_test])

	# Generate representation for each molecule
	print("\n -> generate representation")
	start = time()
	for mol in mols:
		mol.generate_slatm(mbtypes, local=False)
	for mol in mols_test:
		mol.generate_slatm(mbtypes, local=False)
	end = time()
	print(str(end-start))

	X = np.asarray([ mol.representation for mol in mols ])
	X_test = np.asarray([ mol.representation for mol in mols_test ])
#	X = np.concatenate([mol.representation for mol in mols])
#	X_test = np.concatenate([mol.representation for mol in mols_test])
#	M = np.array([mol.natoms for mol in mols])
#	M_test = np.array([mol.natoms for mol in mols_test])
#
	# test set size
	# training set size
	N	 = [225,450,900,1800]
	#N	 = [125,250,500,1000]
	ll = [1e-1, 1e-3, 1e-5, 1e-7]
	total = len(mols)
	nModels = 10
	print(len(mols_test))

	sigma = [12.8,25.6,102.4,204.8,409.6,819.2]

	#print "\n -> loading the Kernel " + str(sigma)

#	K	= get_local_kernels_gaussian(X, X, M, M, sigma)
#	K_test	= get_local_kernels_gaussian(X, X_test, M, M_test, sigma)

	Yprime = np.asarray([ mol.properties for mol in mols])
	Y_test = np.asarray([ mol.properties for mol in mols_test])

	random.seed(667)

	# loop over sigmas (if necessary)
	for j in range(len(sigma)):
		print("\n -> calculating the Kernel ")
		start = time()
		K      = gaussian_kernel(X,X,sigma[j])
		K_test = gaussian_kernel(X,X_test,sigma[j])
		end = time()
		print(str(end-start) + "\n")
		# loop over training set sizes

		for l in ll:
			print("\n")
			for train in N:
				test = total - train
				maes = []
				# cross validation
				for i in range(nModels):
					split = list(range(total))
					# shuffle compounds (for cross validation)
					random.shuffle(split)

					# choose training and test set
					training_index = split[:train]

					# properties
					Y = Yprime[training_index]

					C = deepcopy(K[training_index][:,training_index])
#					C = deepcopy(K[j][training_index][:,training_index])
					C[np.diag_indices_from(C)] += l

					alpha = cho_solve(C, Y)

					Yss = np.dot((K_test[training_index]).T, alpha)
#					Yss = np.dot((K_test[j][training_index]).T, alpha)
					diff = Yss	- Y_test
					mae = np.mean(np.abs(diff))
					maes.append(mae)
#s	 = np.std(maes)/np.sqrt(nModels)

					# print sigma, training set size, mean MAE and standard deviation
				print(str(l) + "\t" + str(sigma[j]) + "\t" + str(train) + "\t" + str(sum(maes)/len(maes)))
