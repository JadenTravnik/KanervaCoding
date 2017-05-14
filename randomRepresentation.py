import random
import math
import operator
import numpy as np
import struct
import matplotlib.pyplot as plt

# It is recommended that one read the README before using this class

class RandomRepresentation:

	def __init__(self, _numPrototypes, _dimensions, alpha, repLearning=True):
		if _numPrototypes < 0:
			raise AssertionError('Need more than 2 prototypes. ' + str(_numPrototypes) + ' given. If 0 given, 50 prototpyes are used by default.')
			return

		self.alpha = alpha/float(_numPrototypes)

		self.dimensions = _dimensions
		self.numPrototypes = _numPrototypes

		self.prototypes = np.random.random((self.numPrototypes, self.dimensions))
		self.weights = np.zeros(self.numPrototypes + 1)
		
		self.activationPercentage = .25
		self.activationRange = .05
		self.activationTrace = np.zeros(self.numPrototypes)
		self.traceDecay = .99
		self.epsilon = 0.0001

		self.desiredDensity = .25
		self.densityRange = .05
		self.densitySelectionBelowProbabilityThreshold = .01
		self.densitySelectionAboveProbabilityThreshold = .01

		self.prototypeDrift = .02

		self.thresholds = np.zeros(self.numPrototypes)

	def GetFeatures(self, _input):
		Diff = self.prototypes - _input
		D = np.sum(np.abs(Diff)**2,axis=-1)**(1./2)

		activations = D < self.thresholds
		inactivations = -activations + 1

		# calculate trace
		self.activationTrace *= self.traceDecay
		self.activationTrace += (1-self.traceDecay)*activations

		# update thresholds to keep frequency of activation
		below = self.activationTrace < (self.activationPercentage - self.activationRange)
		above = self.activationTrace > (self.activationPercentage + self.activationRange)
		self.thresholds += self.epsilon*below - self.epsilon*above

		# move
		density = sum(activations)/(self.numPrototypes*1.)

		if density < self.desiredDensity - self.densityRange:
			inactivations_indicies = np.transpose(np.nonzero(inactivations))

			selected = np.where(np.random.random(len(inactivations_indicies)) < self.densitySelectionBelowProbabilityThreshold)
			selected = selected[0]

			diffs = Diff[selected]
			avgDiff = np.mean(abs(diffs))
			farInds = abs(diffs) > avgDiff

			for i in range(len(selected)):
				if any(farInds[i,:]):
					choice = np.random.choice([j for j in range(self.dimensions) if farInds[i,j]])
					Index = selected[i]
					direction = cmp(Diff[Index,choice],0)
					self.prototypes[Index, choice] += direction*self.prototypeDrift
		elif density > self.desiredDensity + self.densityRange:
			activations_indicies = np.nonzero(activations)
			selected = np.where(np.random.random(len(activations_indicies)) < self.densitySelectionAboveProbabilityThreshold)
			selected = selected[0]

			# Now need to select a dimension which didnt agree with the input AKA wasnt in the threshold
			diffs = Diff[selected]

			avgDiff = np.mean(abs(diffs))
			closeInds = abs(diffs) < avgDiff

			for i in range(len(selected)):
				if any(closeInds[i,:]):
					choice = np.random.choice([j for j in range(self.dimensions) if closeInds[i,j]])
					Index = selected[i]
					direction = cmp(Diff[Index,choice],0)
					self.prototypes[Index, choice] += -direction*self.prototypeDrift

		activations = activations.astype(int)
		activations = np.append(activations, 1) # add bias

		return activations, density, np.mean(self.activationTrace)

	def GetOutput(self, phi):
		y_hat = np.dot(self.weights, phi)
		return y_hat

	def UpdateWeights(self, phi, y, y_hat):	
		diff = y - (y_hat/sum(phi))

		w_delta = self.alpha*diff*phi
		self.weights += w_delta


numPrototypeArr = [2000]#, 100, 200, 400, 800]
dimensionArr = [4]#, 2, 4, 8, 16]
seedArr = [1]
numSteps = 10000


for numP in numPrototypeArr:
	for dim in dimensionArr:
		for seed in seedArr:
			densityData = np.zeros(numSteps)
			avgActivationData = np.zeros(numSteps)
			predictionError = np.zeros(numSteps)
			Y_Data = np.zeros(numSteps)
			Y_Hat_Data = np.zeros(numSteps)

			np.random.seed(seed=seed)
			RR = RandomRepresentation(numP, dim, .01)

			for i in range(numSteps):
				if not i % 1000:
					print('step: ' + str(i))

				point = np.random.random(dim)
				y = sum(np.sin(point))/float(dim)
				phi, densityData[i], avgActivationData[i] = RR.GetFeatures(point)
				y_hat = RR.GetOutput(phi)
				RR.UpdateWeights(phi, y, y_hat)

				Y_Data[i], Y_Hat_Data[i] = y, y_hat

				predictionError[i] = (y-y_hat)**2

			plt.plot(densityData, color='r')
			plt.plot(avgActivationData, color='g')
			plt.show()

			plt.plot(Y_Data, color='r')
			plt.plot(Y_Hat_Data, color='g')
			plt.show()
			
			plt.plot(predictionError, color='b')
			
