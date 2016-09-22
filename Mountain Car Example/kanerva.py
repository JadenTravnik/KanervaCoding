import random
import math
import operator
import numpy as np
import struct

class KanervaCoder:
	distanceMeasure = 'euclidian'
	numPrototypes = 50
	dimensions = 1
	threshold = 0.02
	numClosest = 10
	prototypes = None
	visitCounts = None
	updatePrototypes = None
	minNumberVisited = 50

	updateFunc = None

	activationRadii = 0
	beenAroundTheBlock = False # set to true once a single prototype has been visited the minNumberVisited

	def __init__(self, _numPrototypes, _dimensions, _distanceMeasure):
		if _distanceMeasure != 'hamming' and _distanceMeasure != 'euclidian':
			raise AssertionError('Unknown distance measure ' + str(_distanceMeasure) + '. Use hamming or euclidian.')
			return
		if _numPrototypes < 0:
			raise AssertionError('Need more than 2 prototypes. ' + str(_numPrototypes) + ' given. If 0 given, 50 prototpyes are used by default.')
			return

		self.dimensions = _dimensions
		self.distanceMeasure = _distanceMeasure
		self.numPrototypes = _numPrototypes
		self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
		
		self.visitCounts = np.zeros(self.numPrototypes)
		self.updatedPrototypes = []
		self.minNumberVisited = self.numPrototypes/2
		self.updateFunc = 1
		self.activationRadii = .1
		self.caseStudyN = 5

	def floatToBits(self,f):
			s = struct.pack('>f', f)
			return hex(struct.unpack('>l', s)[0])

	def computeHamming(self, data, i):
		"""Calculate the Hamming distance between two bit strings"""
		prototype = self.prototypes[i]
		count = 0
		for j in range(self.dimensions):
			z = int(self.floatToBits(data[j]),16) & int(prototype[j],16)
			while z:
				count += 1
				z &= z-1 # magic!
		return count


	def GetFeatures(self, data, update):
		if self.distanceMeasure == 'euclidian':
			#tempArr = np.array([1 if np.linalg.norm(data - self.prototypes[i]) < self.threshold else 0 for i in range(len(self.prototypes))])
			
			if self.updateFunc == 0: # XGame Paper

				tempArr = np.array([[i, np.linalg.norm(data - self.prototypes[i])] for i in range(len(self.prototypes))])

				closestPrototypesIndxs = [int(x[0]) for x in sorted(tempArr, key = lambda x: x[1])[:self.numClosest]]

				if update:
				
					print('Updating XGame')
					for i in closestPrototypesIndxs:
						self.visitCounts[i] += 1

					if self.beenAroundTheBlock == False: # use this so we dont have to calculated the max every time
						maxVisit = max(self.visitCounts)
						print('Max visit: ' + str(maxVisit)) 
						if maxVisit > self.minNumberVisited:
							self.beenAroundTheBlock = True
							
					if self.beenAroundTheBlock:
						self.updatePrototypesXGame()

			elif self.updateFunc == 1: # Case Studies

				closestPrototypesIndxs = []
				data = np.array(data)
				for prototype in range(self.numPrototypes):
					diffArr = abs(data - self.prototypes[prototype])
					#closestPrototypesIndxs.append(min([1 - diff/self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr]))
					u = min([1 - diff/self.activationRadii if diff <= self.activationRadii else 0 for diff in diffArr])
					if u > 0:
						closestPrototypesIndxs.append(prototype)

				if update:
					print('Updating Case Studies')
					# if len(closestPrototypesIndxs) < self.caseStudyN:
					# 	for i in range(self.caseStudyN - len(closestPrototypesIndxs)):

			return closestPrototypesIndxs

		else:
			# fuzzy
			#return np.array([self.computeHamming(data,i)/self.threshold for i in range(len(self.prototypes))])
			tempArr =  np.array([1 if self.computeHamming(data,i) < self.threshold else 0 for i in range(len(self.prototypes))])

			return np.where(tempArr == 1)[0]



	def updatePrototypesXGame(self):
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

		print('Done updatedPrototypes: updatedPrototypes: ' + str(self.updatedPrototypes))