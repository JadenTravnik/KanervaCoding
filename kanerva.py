import random
import math
import operator
import numpy as np

_maxLongint = 2147483647                # maximum integer
_maxLongintBy4 = _maxLongint // 4       # maximum integer divided by 4 

distanceMeasure = 'hamming'
spacingType = 'random'
numPrototypes = 10000
dimensions = 1
maxBitString = 0
lowerBounds = []
upperBounds = []
ranges = []

prototypes = None



def Initialize(_numPrototypes, _lowerBounds, _upperBounds, _spacingType, _distanceMeasure):
	if len(_lowerBounds) != len(_upperBounds):
		raise AssertionError('lowerBounds and upperBounds must be the same length')
		return
	if str(_distanceMeasure) != 'hamming' or str(_distanceMeasure) != 'eclidian':
		raise AssertionError('Unknown distance measure ' + str(_distanceMeasure) + '. Use hamming or eclidian.')
		return
	if str(_spacingType) != 'random' or str(_spacingType) != 'uniform' or str(_spacingType) != 'adaptive':
		raise AssertionError('Unknown spacing type ' + str(_spacingType) + '. Use random, uniform, or adaptive.')
		return
	if _numPrototypes < 0:
		raise AssertionError('Need more than 2 prototypes. ' + str(_numPrototypes) + ' given. If 0 given, 10000 prototpyes are used by default.')
		return
	if len(_lowerBounds) < 1:
		raise AssertionError('Need at least one dimension in state space.')
		return

	distanceMeasure = _distanceMeasure
	numPrototypes = _numPrototypes
	lowerBounds = np.array(_lowerBounds)
	upperBounds = np.array(_upperBounds)
	ranges = upperBounds - lowerBounds
	spacingType = _spacingType
	dimensions = len(lowerBounds)
	makePrototypes()


def makePrototypes():
	prototypes = np.array([np.zeros(dimensions) for i in range(numPrototypes)])

	if spacingType == 'random':
		if distanceMeasure == 'eclidian':
			prototypes = np.array([np.random.rand(dimensions)*ranges + lowerBounds for i in range(numPrototypes)])		
		elif distanceMeasure == 'hamming':
			# find the maximum bit size if every float was represented together
			maxBitString = 32*dimensions
			

	else:
		raise Exception('Not implemented yet. :(')	
		return

def GetFeatures(data):
	return np.array([])

