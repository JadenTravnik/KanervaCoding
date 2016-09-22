from prototype import Prototype
from math import exp, sqrt, pi

class KanervaCoding1D:

	def __init__(self, numPrototypes, groups, maxState):

		self.numGroups = groups
		self.numPrototypes = numPrototypes
		self.prototypes = []
		self.stateDimension = 1
		self.maxState = maxState
		self.generatePrototypes(False)
		self.computePrototypeWidth()

	def getWeights(self):
		w = []
		for i in range(self.numPrototypes):
			w.append(self.prototypes[i].getTheta())
		return w


# generates k different prototypes based on the isDifferent metric
	def generatePrototypes(self, isRandom):

		if isRandom:
			for i in range(self.numPrototypes):
				numDifferent = 0
				self.prototypes.append(Prototype(1, self.stateDimension))
				while numDifferent < i:
					if self.prototypes[i].isDifferent(self.prototypes[numDifferent]):
						numDifferent += 1
					else:
						self.prototypes[i].setRandomly()
						numDifferent = 0

		else :
			for i in range(self.numPrototypes):
				groupNum = i % self.numGroups
				p = Prototype(1, self.stateDimension, groupNum)
				p.setFixed([i/float(self.maxState+1) ], 0)
				self.prototypes.append(p)



	def getV(self, state):
		tempPrototype = Prototype(1, self.stateDimension)
		tempPrototype.setFixed([state/float(self.maxState+1)], 0)
		thetaSum = 0

		pIndex = [0]*self.numGroups
		pGrade = self.getPrototypeIndex(state, pIndex)

		for i in range(len(pIndex)):
			thetaSum += self.prototypes[pIndex[i]].getTheta()*pGrade[i]

		return thetaSum

# computes the prototype width which is the variance of distance between all other prototypes
	def computePrototypeWidth(self):
		
		if self.numPrototypes > 1:

			for i in range(self.numPrototypes):
				featureDiff = []
				total, mean, variance = 0,0,0
				for j in range(self.numPrototypes):
					if i != j:
						featureDiff.append(self.prototypes[i].calculateDiff(self.prototypes[j]))

				mean = sum(featureDiff)/(self.numPrototypes - 1)
				total = sum([pow(diff - mean,2) for diff in featureDiff])

				variance = total/ float(self.numPrototypes - 1) ## is this actually how this is calculated

				#self.prototypes[i].setFeatureWidth(1)
				self.prototypes[i].setFeatureWidth(variance)
		else:
			self.prototypes[0].setFeatureWidth(100)

	def getPrototypeIndex(self, state, prototypeIndex):
		tempPrototype = Prototype(1, self.stateDimension)
		tempPrototype.setFixed([state/float(self.maxState + 1)], 0)
		pGrade = [0]*self.numGroups

		for i in range(len(self.prototypes)):
			prototype = self.prototypes[i]
			tempFeatureDiff = tempPrototype.calculateDiff(prototype)

			membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))#/sqrt(2*prototype.getFeatureWidth()*pi)
			if pGrade[prototype.group] < membershipGrade:
				prototypeIndex[prototype.group] = i
				pGrade[prototype.group] = membershipGrade


		return [p/sum(pGrade) for p in pGrade]


	# updates the theta values of the prototypes
	def updateWeights(self, state, delta, alpha):

		pIndex = [0]*self.numGroups
		pGrade = self.getPrototypeIndex(state, pIndex)

		for i in range(len(pIndex)):
			update = self.prototypes[pIndex[i]].getTheta() + alpha * delta * pGrade[i]
			self.prototypes[pIndex[i]].setTheta(update)
