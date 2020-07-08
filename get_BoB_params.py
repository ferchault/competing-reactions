#!/usr/bin/env python3

import sys
import time
from datetime import datetime
import random
#import cPickle
import numpy as np
from copy import deepcopy
import qml
from qml.representations import *
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
import itertools
from time import time

from sklearn.model_selection import KFold

def get_energies(filename):
	""" returns dic with energies for xyz files
	"""
	f = open(filename, "r")
	lines = f.readlines()
	f.close()

	energies = dict()

	for line in lines:
		tokens = line.split()
		xyz_name = tokens[1]
		Ebind = float(tokens[2])
		#if Ebind < 100 and Ebind > 0: energies[xyz_name] = Ebind
		energies[xyz_name] = Ebind

	return energies

if __name__ == "__main__":

	data  = get_energies("train_sn2.txt")

	mols = []

	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
		mol.read_xyz("/home/heinen/PhD/projects/sn_e/paper/raw_data/xyz/" + xyz_file)
		mol.properties = data[xyz_file]
		mols.append(mol)

	bags = {
          "H" :  max([mol.atomtypes.count("H" ) for mol in mols]),
          "C" :  max([mol.atomtypes.count("C" ) for mol in mols]),
          "N" :  max([mol.atomtypes.count("N" ) for mol in mols]),
          "O" :  max([mol.atomtypes.count("O" ) for mol in mols]),
          "F" :  max([mol.atomtypes.count("F" ) for mol in mols]),
          "Cl" : max([mol.atomtypes.count("Cl") for mol in mols]),
          "Br" : max([mol.atomtypes.count("Br") for mol in mols]), }

	for mol in mols:
		mol.generate_bob(asize=bags)

	ll = [1e-1, 1e-3, 1e-5]
	sigma = [0.1*2**i for i in range(5,20)]

	X      = np.asarray([mol.representation for mol in mols])
	Yprime = np.asarray([ mol.properties for mol in mols ])

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
