#!/usr/bin/env python3

import sys
import numpy as np


def read_data(f):

	lines = open(f, 'r').readlines()

	vals = np.array([])

	for line in lines:
		val = float(line.split()[2])
		vals = np.append(vals, val)

	return vals

if __name__ == '__main__':

	filename = sys.argv[1]
	f2 = sys.argv[2]

	vals = read_data(filename)
	test = read_data(f2)
	mean = np.mean(vals)

	print(mean)

	MAEs = np.array([])

	for t in test:
		diff = np.abs(t - mean)
		MAEs = np.append(MAEs, diff)

	print(np.mean(MAEs))
	print(sum(MAEs)/len(MAEs))
