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
	data  = get_energies("train_e2.txt")
	data2 = get_energies( "test_e2.txt")

	mols = []
	mols_test = []
	names = []
	names2 = []

	# read molecules
	for xyz_file in sorted(data.keys()):
		mol = qml.Compound()
#		mol.read_xyz("xyz/e2/" + xyz_file + ".xyz")
		mol.properties = data[xyz_file]
		names.append(xyz_file)
		mols.append(mol)


	for xyz_file in sorted(data2.keys()):
		mol = qml.Compound()
#		mol.read_xyz("xyz/e2/" + xyz_file + ".xyz")
		mol.properties = data2[xyz_file]
		names2.append(xyz_file)
		mols_test.append(mol)

	#print "\n -> calculating the Representation "
	#for mol in mols:
	#		 mol.representation = get_one_hot(mol.name)
	#for mol in mols_test:
	#		 mol.representation = get_one_hot(mol.name)

	N = [125,250,500,1000]
	#N = [225,450,900,1800]
	total = len(mols)
	nModels = 10
	sigma = [0.1*2**i for i in range(10)]
	ll = [0.1, 1e-3,1e-5, 1e-7, 1e-9]

	X = get_one_hot_encoding(names)
	X_test = get_one_hot_encoding(names2)

	Yprime = np.asarray([ mol.properties for mol in mols ])
	Y_test = np.asarray([ mol.properties for mol in mols_test ])

	random.seed(667)

	for j in range(len(sigma)):
		print('\n')
		for l in ll:
			print()
			K =      gaussian_kernel(X, X, sigma[j])
			K_test = gaussian_kernel(X, X_test, sigma[j])
			for train in N:
				maes = []
				for i in range(nModels):
					split = list(range(total))
					random.shuffle(split)

					training_index	= split[:train]

					Y = Yprime[training_index]

					C = deepcopy(K[training_index][:,training_index])
					C[np.diag_indices_from(C)] += l

					alpha = cho_solve(C, Y)

					Yss = np.dot((K_test[training_index]).T, alpha)
					diff = Yss	- Y_test
					mae = np.mean(np.abs(diff))
					maes.append(mae)

				s = np.std(maes)/np.sqrt(nModels)
				print(str(l) + '\t' + str(sigma[j]) +	"\t" + str(train) + "\t" + str(sum(maes)/len(maes)) + " " + str(s))
