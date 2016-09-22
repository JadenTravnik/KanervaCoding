import random
import math
import numpy as np
from VisitCounterKanerva import VisitCounterKanerva

class VisitCounterCorrelationKanerva(VisitCounterKanerva):

	def __init__(self, _startingPrototypes, _dimensions):
		VisitCounterKanerva.__init__(self, _startingPrototypes, _dimensions)
		self.prototypePairVisits = np.zeros((self.numPrototypes,self.numPrototypes))
		self.maxCorrelatedPair = []
		self.maxCorrelation = 0
		self.minNumberVisited = self.numPrototypes*5
		self.minPrototypes = 100
		self.maxVisit = 0
		self.addedPrototypes = []
		self.updatedPrototypes = []
		self.deletedPrototypes = []		

	def GetFeatures(self, data, update):

		tempArr = np.array([[i, np.linalg.norm(data - self.prototypes[i])] for i in range(len(self.prototypes))])

		closestPrototypesIndxs = [int(x[0]) for x in sorted(tempArr, key = lambda x: x[1])[:self.numClosest]]

		if update:
			for i in closestPrototypesIndxs:
				self.visitCounts[i] += 1
				for correlatedIndex in closestPrototypesIndxs:
					if correlatedIndex != i:
						self.prototypePairVisits[i][correlatedIndex] += 1
			self.maxCorrelatedPair = np.unravel_index(self.prototypePairVisits.argmax(), self.prototypePairVisits.shape)


			if self.beenAroundTheBlock == False: # use this so we dont have to calculated the max every time
				tempMaxVisit = max(self.visitCounts)
				if self.maxVisit < tempMaxVisit:
					self.maxVisit = tempMaxVisit
				if self.maxVisit > self.minNumberVisited:
					self.beenAroundTheBlock = True
					
			if self.beenAroundTheBlock:
				print('Updating')
				self.updatePrototypes()
		return closestPrototypesIndxs

	def updatePrototypes(self):
		self.addedPrototypes = []
		self.updatedPrototypes = []
		self.deletedPrototypes = []

		mostVisitedPrototypeIndexs = [i[0] for i in sorted(enumerate(self.visitCounts), key=lambda x:x[1])]
		count = 0

		self.numPrototypes = len(self.prototypes)
		print('Num Prototypes before update: ' + str(self.numPrototypes))

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

		self.numPrototypes = len(self.prototypes)
		print('Num Prototypes after update: ' + str(self.numPrototypes))

		if self.numPrototypes > self.minPrototypes:
			print('Adding new prototype to an array of prototypes of length: ' + str(len(self.prototypes)))
			newPrototype = np.array((self.prototypes[self.maxCorrelatedPair[0]] + self.prototypes[self.maxCorrelatedPair[1]])/2.0)
			print('New prototype: ' + str(newPrototype))
			self.addedPrototypes.append(newPrototype)
			print('numprototypes just before add: ' + str(len(self.prototypes)))
			self.prototypes = np.append(self.prototypes,np.array([newPrototype]), axis=0)
			print('Added a new prototype. Array is now length: ' + str(len(self.prototypes)))
			
			self.deletedPrototypes = sorted(self.maxCorrelatedPair, reverse=True)
			self.prototypes = np.delete(self.prototypes, self.deletedPrototypes[0], axis=0)
			self.prototypes = np.delete(self.prototypes, self.deletedPrototypes[1], axis=0)

		self.numPrototypes = len(self.prototypes)
		print('NumPrototypes after add/delete: ' + str(self.numPrototypes))
		self.visitCounts = np.zeros(self.numPrototypes)
		self.beenAroundTheBlock = False	
		self.maxVisit = 0

	def EmptyArrays(self):
		self.addedPrototypes = []
		self.updatedPrototypes = []
		self.deletedPrototypes = []						