from prototype import Prototype
from math import exp, sqrt, pi

class KanervaCoding1D:

	def __init__(self, numPrototypes, maxState):

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
				p = Prototype(1, self.stateDimension)
				p.setFixed([i/float(self.maxState+1) ], 0)
				self.prototypes.append(p)

	def getV(self, state):
		tempPrototype = Prototype(1, self.stateDimension)
		tempPrototype.setFixed([state/float(self.maxState)], 0)
		thetaSum = 0

		for prototype in self.prototypes:
			tempFeatureDiff = tempPrototype.calculateDiff(prototype)
			membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/(2*prototype.getFeatureWidth())))#/sqrt(2*prototype.getFeatureWidth()*pi)
			thetaSum += prototype.getTheta() * membershipGrade

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
				self.prototypes[i].setFeatureWidth(.1)
		else:
			self.prototypes[0].setFeatureWidth(100)


	# updates the theta values of the prototypes
	def updateWeights(self, state, delta, alpha):

		tempPrototype = Prototype(1, self.stateDimension)
		tempPrototype.setFixed([state/float(self.maxState)], 0)

		for prototype in self.prototypes:
			tempFeatureDiff = tempPrototype.calculateDiff(prototype)

			membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))#/sqrt(2*prototype.getFeatureWidth()*pi)
			prototype.setTheta(prototype.getTheta() + alpha * delta * membershipGrade)
