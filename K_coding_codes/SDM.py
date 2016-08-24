import numpy as np


class FloatReceptor:
	def __init__(self, _value, _radius):
		self.value = _value
		self.radius = _radius

	def process(self, inputVal):
		try:
			return 1 if abs(inputVal - self.value) < self.radius else 0
		except:
			return 0

class ReceptorLine:
	def __init__(self, numReceptors):
		self.receptors = [FloatReceptor(j/numReceptors, 1/numReceptors) for j in range(numReceptors)]

	def process(self, inputVal):
		return [r.process(inputVal) for r in self.receptors]


class ReceptorLayer:
	def __init__(self, numReceptorsPerDimension, numDimensions):
		self.receptorLines = [ReceptorLine(numReceptorsPerDimension) for j in range(numDimensions)]

	def process(self, inputData):
		response = []
		for i in range(self.receptorsLines):
			response.extend(self.receptorsLines[i].process(inputData[i]))
		return response


class HardLocation:

	def __init__(self, _address, _n):
		self.address = _address
		self.counters = np.array([0]*len(_n))

	def isNear(self, address):

		difference = 0
		# a logically anded array ([true, false, ..]) multiplied by an integer is promoted to an int array.
		# Which then can be sent to packbits, that here returns an array of 1 element, so now z is an integer
		z = np.packbits(np.logical_and(self.address, address)*1)[0]
		while z:
			difference += 1
			z &= z-1 # magic!
		return 1 if difference < self.radius else 0

	def write(self, inputData):
		self.counters += (inputData*2 - 1)


class SDM:

	def __init__(self, numHardLocations, numFloatDimensions, numDimensions):
		self.hardLocations = [HardLocation()]



