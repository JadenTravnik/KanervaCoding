import random
import math
import operator
import numpy as np


class BaseKanervaCoder:
	def __init__(self, _startingPrototypes, _dimensions):
		self.numPrototypes = _startingPrototypes
		self.dimensions = _dimensions
		self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
		self.updatedPrototypes = []

	def GetFeatures(self, data, update):
		closestPrototypesIndxs = []
		for i in range(self.numPrototypes):
			threshold = .1
			if np.linalg.norm(data - self.prototypes[i]) < threshold:
				closestPrototypesIndxs.append(i)

		return closestPrototypesIndxs


	def updatePrototypes(self):
		self.updatedPrototypes = []