import math
import os
import sys
import numpy as np
from oeFilter import oeFilter

def fbCreate(numOrient = 8, startSigma = 1, numScales = 2, scaling = math.sqrt(2), \
		elong = 2):

	support = 3
	fb = np.zeros((len(range(1,numOrient+1))*2, len(range(1,numScales+1)))).tolist()
	for idx0, scale in enumerate(range(1,numScales+1)):
		sigma = startSigma * (scaling**(scale-1));
		#ipdb.set_trace()
		for orient in range(1,numOrient+1):
			theta = (orient-1)/float(numOrient) * math.pi
			fb[(2*orient)-2][idx0] = oeFilter([sigma*elong, sigma], support, theta, 2, 0)
			fb[(2*orient)-1][idx0] = oeFilter([sigma*elong, sigma], support, theta, 2, 1)

	return fb

