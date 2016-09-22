import random
import math
import numpy as np
from BaseKanervaCoder import BaseKanervaCoder

class VisitCounterKanerva(BaseKanervaCoder):

	def __init__(self, _startingPrototypes, _dimensions):
		BaseKanervaCoder.__init__(self, _startingPrototypes, _dimensions)
		self.visitCounts = np.zeros(self.numPrototypes)
		self.updatedPrototypes = []
		self.minNumberVisited = self.numPrototypes*5
		self.numClosest = 10
		self.beenAroundTheBlock = False

	def GetFeatures(self, data, update):

		tempArr = np.array([[i, np.linalg.norm(data - self.prototypes[i])] for i in range(len(self.prototypes))])

		closestPrototypesIndxs = [int(x[0]) for x in sorted(tempArr, key = lambda x: x[1])[:self.numClosest]]

		if update:
		
			print('Updating visit counter')
			for i in closestPrototypesIndxs:
				self.visitCounts[i] += 1

			if self.beenAroundTheBlock == False: # use this so we dont have to calculated the max every time
				maxVisit = max(self.visitCounts)
				print('Max visit: ' + str(maxVisit)) 
				if maxVisit > self.minNumberVisited:
					self.beenAroundTheBlock = True
					
			if self.beenAroundTheBlock:
				self.updatePrototypes()
		return closestPrototypesIndxs

	def updatePrototypes(self):
		self.updatedPrototypes = []
		mostVisitedPrototypeIndexs = [i[0] for i in sorted(enumerate(self.visitCounts), key=lambda x:x[1])]
		count = 0
		for prototype in range(self.numPrototypes):
			if math.exp(-self.visitCounts[prototype]) > random.random(): # remove with probability e^-m (Equation 4)
				self.visitCounts[prototype] = 0
				replacementPrototypeIndex = mostVisitedPrototypeIndexs[-(count+1)]
				self.prototypes[prototype] = self.prototypes[replacementPrototypeIndex] # add another prototype
				

				for dimension in range(self.dimensions):
					randOffset = (random.random() - .5)/(self.numPrototypes^-self.dimensions)
					self.prototypes[prototype][dimension] += randOffset # change every dimension to something close by

				self.updatedPrototypes.append([prototype, self.prototypes[prototype], replacementPrototypeIndex])
				count += 1				

		self.visitCounts = np.zeros(self.numPrototypes)
		self.beenAroundTheBlock = False													