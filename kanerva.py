import random
import math
import operator
import numpy as np
import struct

# It is recommended that one read the README before using this class

class SelectiveKanervaCoder:

	def __init__(self, _numPrototypes, _dimensions = 2, _eta = .025, _seed = 0):

		self.numPrototypes = _numPrototypes
		self.dimensions = _dimensions
		self.eta = _eta
		self.c = int(_numPrototypes*_eta)
		self.seed = _seed if _seed != 0 else np.random.random()
		np.random.seed(_seed)
		self.prototypes = np.random.rand(_numPrototypes, _dimensions)

	def getFeatures(self, _input):
		D = self.prototypes - _input
		D = np.sqrt(sum(D.T**2)) # get Euclidian distance
		indexes = np.argpartition(D, self.c, axis=0)[:self.c]
		phi = np.zeros(self.numPrototypes)
		phi[indexes] = 1
		return phi


class KanervaCoder:
	distanceMeasure = 'euclidian' # alternatively hamming distance could be used
	numPrototypes = 50
	dimensions = 1
	threshold = 0.02

	# an alternative to the threshold is to take the X closest points
	numClosest = 10
	prototypes = None
	visitCounts = None
	updatePrototypes = None
	minNumberVisited = 50

	Fuzzy = False

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

		# because each observation is normalized within its range,
		# each prototype can be a random vector where each dimension is within (0-1)
		self.prototypes = np.array([np.random.rand(self.dimensions) for i in range(self.numPrototypes)])
		
		# this is a counter for each prototype that increases each time a prototype is visited 
		self.visitCounts = np.zeros(self.numPrototypes)

		# this is used within the learner to manipulate the prototype location
		self.updatedPrototypes = []

		# this is one thing I have been testing, if we want to manipulate our prototypes (combine/move/add)
		# we should make sure that we have explored the state space sufficinetly enough 
		# minNumberVisited is one way to specify that we want at least one prototype to be visited
		# this number of times before we manipulate our representation
		self.minNumberVisited = self.numPrototypes/2

		# if updateFunc is 0, perform the representation update function found in the XGame paper
		# if updateFunc is 1, perform the representation update function found in the Case Studies paper  
		self.updateFunc = 1

		# the activationRadii is defined in the Case Study paper and is used as a radius to find all 
		# prototypes that are sufficiently close.
		# setting the activationRadii to 0 will include no prototypes, and to 1 will include all of them
		self.activationRadii = .1

		# This is defined in the Case Study paper as a way of limiting how many prototypes should are activated
		# by a given observation
		self.caseStudyN = 5

		# if false, an array of the indexes of the activated prototypes is returned
		# if true, the distance metric for every prototype is returned
		self.Fuzzy = False


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


	# The function to get the features for the observation 'data'
	# the argument 'update' is a boolean which indicates whether the representation should 
	# check for an update condition (such as meeting the minimal amount of prototype visits).
	# This is useful for debugging 
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

		else if self.distanceMeasure == 'hamming':
			# fuzzy
			#return np.array([self.computeHamming(data,i)/self.threshold for i in range(len(self.prototypes))])
			tempArr =  np.array([1 if self.computeHamming(data,i) < self.threshold else 0 for i in range(len(self.prototypes))])

			return np.where(tempArr == 1)[0]


	# the update algorithm defined in the XGame paper
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
