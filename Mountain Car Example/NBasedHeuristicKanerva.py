import random
import math
import numpy as np

class NBasedHeuristicKanerva(BaseKanervaCoder):

	def __init__(self, _startingPrototypes, _dimensions):
		BaseKanervaCoder.__init__(self, _startingPrototypes, _dimensions)
		
		self.activationRadii = .1
		self.caseStudyN = 5

	def GetFeatures(self, data, update):

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


	def updatePrototypes(self):
		self.updatedPrototypes = []
												