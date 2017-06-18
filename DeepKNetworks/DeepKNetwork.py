import random
import math
import operator
import numpy as np


class DeepKNetwork:
        def __init__(self, prototypeList):
		self.prototypeList = prototypeList

		self.layers = []
		self.eta = .025
		self.layers.append(np.random.rand(prototypeList[1], prototypeList[0]))
		print('Added first layer ' + str(self.layers[0].shape))
		for i in range(1,len(prototypeList)-1):
			self.layers.append(np.where(np.random.randint(2, \
				size = (prototypeList[i+2], prototypeList[i]))))
			print('Added layer ' + str(i) + ' ' + str(self.layers[i].shape))
		
		print('Network has ' + str(len(self.layers)) + ' layer(s).')

		self.stateScale = .9
		self.bias = np.ones(prototypeList[0])*(1-self.stateScale)*.5
		self.c = [int(prototypeList[i]*self.eta) for i in range(len(prototypeList)-1)]

		


        def GetFeatures(self, data):
                closestPrototypesIndxs = []
		D = self.layers[0] - (np.array(data)*self.stateScale + self.bias)
		D = np.sqrt(sum(D.T**2))    # a bottlenect for sure
		indexes = np.argpartition(D, self.c[0], axis=0)[:self.c[0]]
		
		for i in range(1,len(self.layers)):
			D = np.sum(np.setxor1d(self.layers[i], indexes, True), axis=1)
#			phi = np.zeros(self.prototypeList[i])
#			phi[indexes] = 1
#			D = np.sum(np.logical_xor(self.layers[i], phi), axis=1)
			indexes = np.argpartition(D, self.c[i], axis=0)[:self.c[i]]


				
                return indexes














