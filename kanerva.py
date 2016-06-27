import random
import math
import operator
import numpy as np
import struct

# from kanerva import *
# k = KanervaCoder(50, [0,0, -2, -2 ,-2], [4.5, 4.5, 2 ,2 ,2], 'random', 'hamming')
# observation = [2.0, 4.0, 7.0, 3.0, 9.0]
# k.GetFeatures(observation)

class KanervaCoder:
	distanceMeasure = 'hamming'
	spacingType = 'random'
	numPrototypes = 10000
	dimensions = 1
	maxBitString = 0
	lowerBounds = []
	upperBounds = []
	ranges = []
	threshold = 1
	prototypes = None

	def __init__(self, _numPrototypes, _lowerBounds, _upperBounds, _spacingType, _distanceMeasure):
		if len(_lowerBounds) != len(_upperBounds):
			raise AssertionError('lowerBounds and upperBounds must be the same length')
			return
		if _distanceMeasure != 'hamming' and _distanceMeasure != 'euclidian':
			raise AssertionError('Unknown distance measure ' + str(_distanceMeasure) + '. Use hamming or euclidian.')
			return
		if _spacingType != 'random' and _spacingType != 'uniform' and _spacingType != 'adaptive':
			raise AssertionError('Unknown spacing type ' + str(_spacingType) + '. Use random, uniform, or adaptive.')
			return
		if _numPrototypes < 0:
			raise AssertionError('Need more than 2 prototypes. ' + str(_numPrototypes) + ' given. If 0 given, 10000 prototpyes are used by default.')
			return
		if len(_lowerBounds) < 1:
			raise AssertionError('Need at least one dimension in state space.')
			return

		self.distanceMeasure = _distanceMeasure
		self.numPrototypes = _numPrototypes
		self.lowerBounds = np.array(_lowerBounds)
		self.upperBounds = np.array(_upperBounds)
		self.ranges = self.upperBounds - self.lowerBounds
		self.spacingType = _spacingType
		self.dimensions = len(_lowerBounds)
		self.makePrototypes()

	def floatToBits(self,f):
			s = struct.pack('>f', f)
			return hex(struct.unpack('>l', s)[0])

	def makePrototypes(self):

		if self.spacingType == 'random':
			if self.distanceMeasure == 'euclidian':
				self.prototypes = []
				for i in range(self.numPrototypes):
					tempPrototype = np.random.rand(self.dimensions)*ranges + self.lowerBounds
					if i > 0:
						self.threshold = self.threshold + (np.linalg.num(tempPrototype, self.prototypes[i]) - self.threshold)/(i+1)
					self.prototypes.append(tempPrototype)
				self.prototypes = np.array(self.prototypes)


			elif self.distanceMeasure == 'hamming':
				# find the maximum bit size if every float was represented together
				self.prototypes = []
				for i in range(self.numPrototypes):
					tempPrototype = []
					for j in range(self.dimensions):
						tempDimension = self.floatToBits(random.random()*self.ranges[j] + self.lowerBounds[j])
						print(tempDimension)
						tempPrototype.append(tempDimension)
					if i > 0:
						print('Calculating threshold')
						self.threshold = self.threshold + (self.initHamming(tempPrototype, i-1) - self.threshold)/(i+1)
						print('Calculated threshold')
					print('Appended the prototype')
					self.prototypes.append(tempPrototype)
				self.prototypes = np.array(self.prototypes)
				
		else:
			raise Exception('Not implemented yet. :(')	
			return


	def initHamming(self, data, i):
		"""Calculate the Hamming distance between two bit strings"""
		#assert len(x) == len(prototypes(i))
		prototype = self.prototypes[i]
		count = 0
		print('In calculate threshold')
		for j in range(self.dimensions):
			print('On dimension ' + str(j))
			z = int(data[j],16) & int(prototype[j],16)
			while z:
				count += 1
				z &= z-1 # magic!
		return count

	def computeHamming(self, data, i):
		"""Calculate the Hamming distance between two bit strings"""
		#assert len(x) == len(prototypes(i))
		prototype = self.prototypes[i]
		count = 0
		for j in range(self.dimensions):
			z = int(self.floatToBits(data[j]),16) & int(prototype[j],16)
			while z:
				count += 1
				z &= z-1 # magic!
		return count

	def GetFeatures(self, data):
		if self.distanceMeasure == 'euclidian':
			return np.array([1 if np.linalg.num(data, self.prototypes[i]) < self.threshold else 0 for i in range(len(self.prototypes))])
		else:
			return np.array([1 if self.computeHamming(data,i) < self.threshold else 0 for i in range(len(self.prototypes))])		

