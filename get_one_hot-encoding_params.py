#!/usr/bin/env python3

import sys
import time
from datetime import datetime
import random
import numpy as np
from copy import deepcopy
import qml
from qml.representations import *
from qml.kernels import gaussian_kernel
from qml.kernels import laplacian_kernel
from qml.math import cho_solve
import itertools
from time import time
from scipy.optimize import minimize
from collections import defaultdict

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
		xyz_name = tokens[0]
		Ebind = float(tokens[2]) #* 627.509
		energies[xyz_name] = Ebind

	return energies

def get_R(FG):
	r = 1
	if FG == "A": R = [r,0,0,0,0]
	if FG == "B": R = [0,r,0,0,0]
	if FG == "C": R = [0,0,r,0,0]
	if FG == "D": R = [0,0,0,r,0]
	if FG == "E": R = [0,0,0,0,r]

	return R

def get_X(LG):
	x = 1
	if LG == "A": X = [x,0,0]
	if LG == "B": X = [0,x,0]
	if LG == "C": X = [0,0,x]

	return X

def get_Y(Nuc):
	y = 1
	if Nuc == "A": Y = [y,0,0,0]
	if Nuc == "B": Y = [0,y,0,0]
	if Nuc == "C": Y = [0,0,y,0]
	if Nuc == "D": Y = [0,0,0,y]

	return Y

def get_one_hot_encoding(names):
	representation = []
	tmp = []
	for name in names:
		tokens = name.split('_')
		FG1 = tokens[0]
		FG2 = tokens[1]
		FG3 = tokens[2]
		FG4 = tokens[3]
		LG	= tokens[4]
		Nuc = tokens[5]

		R1 = get_R(FG1)
		R2 = get_R(FG2)
		R3 = get_R(FG3)
		R4 = get_R(FG4)

		X = get_X(LG)
		Y = get_Y(Nuc)

		representation.append(R1 + R2 + R3 + R4 + X + Y)#int(sys.argv[2])*fukui) #+ rxn)

	return np.asarray(representation)

if __name__ == "__main__":

	# get binding energies
	data  = get_energies("train_sn2.txt")

	mols = []
	names = []

	# read molecules
	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
#		mol.read_xyz("xyz/e2/" + xyz_file + ".xyz")
		mol.properties = data[xyz_file]
		names.append(xyz_file)
		mols.append(mol)

	sigma = [0.1*2**i for i in range(2,10)]
	ll = [0.1, 1e-3,1e-5, 1e-7, 1e-9]

	X      = get_one_hot_encoding(names)
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
