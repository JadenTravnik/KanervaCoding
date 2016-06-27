import numpy as np 
import random


class Prototpye:
	def __init__(self):
		self.theta_value = 0 # theta value associated with each prototype (feature) # float
		self.frequence_value = 0	# float
		self.featureWidth_value = 0 # float

		self.action = 0 # int
		state = np.array([])

	# feature
	def copy(self, feature2):
		self.setFixed(feature2.getState(), feature2.getAction)

	# void
	def setRandomly(self):
		self.setAction(random.randrange(0,5))
		s = []
		s.append(random.randrange(1,100))
		s.append(random.randrange(1,100))
		s.append(random.randrange(48) + 100)
		s.append(random.randrange(64195) + 1340)
		self.setState(s)

	#void # vector s and int a
	def setFixed(self, s, a):
		self.setState(s)
		self.setAction(a)
		self.setTheta(0.0)
		self.setFrequence(0.0)
		self.setFeatureWidth(0.0)

	#void
	def printPrototpye(self):

		print(str(self.state) + ', ' + str(self.action))

	# void, ofstream
	def exportFeature(self, ftfp):
		print(ftfp)

	# void, vector s
	def setState(self, s):
		self.s = s

	# void, int a
	def setAction(self, a):
		self.action = a

	# vector<double> 
	def getState(self):
		return self.state

	# int 
	def getAction(self):
		return self.action

	# bool, other prototype
	def isNeighbor(self, feature2):
		difference = CalculateDiff(feature2);
		return difference <= 1
			
	# bool, other prototpye
	def isDifferent(self, feature2):
		difference = self.calculateDiff(feature2)
		return difference > 0

	# unsigned int, feature
	def calculateDiff(self, feature2):
		diff = 0 #unsigned int
		actDif = 0 # unsigned int
		stateDif = 0 # unsigned int
		state_4_Dif = 0 # unsigned int

		if self.action == feature2.getAction():
			actDif = 0
		else:
			actDif = 1

		for i in xrange(0,3):
			if (self.state[i] != feature2.getState()[i])
				stateDif++

		local = self.state[3]
		given = feature2.getState()[3]
		big, small = 0,0
		if local >= given:
			big = local
			smal = given
		else:
			big = given
			small = local

		if big - small <= big * .05:
			state_4_Dif = 0
		else:
			state_4_Dif = 1

		difference = actDif + stateDif + state_4_Dif
		return difference


	# void, float theta
	def setTheta(self, theta):
		self.theta_value = theta

	# float
	def getTheta(self):
		return self.theta_value


	# void, float
	def setFrequence(self, frequence):
		self.frequence_value = frequence

	# float
	def getFrequence(self):
		return self.frequence_value


	# void, float
	def setFeatureWidth(self, featureWidth):
		self.featureWidth_value = featureWidth_value

	# float
	def getFeatureWidth(self):
		return self.featureWidth_value




