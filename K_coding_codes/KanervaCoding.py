from prototype import Prototype
from math import exp, sqrt
import numpy as np
import time


numPrototypes = 1
numActions = 1
stateDimension = 1
prototypes = []
isRandom = False

# computes the prototype width which is the variance of distance between all other prototypes
def computePrototypeWidth():
	
	if numPrototypes > 1:

		for i in range(numPrototypes):
			featureDiff = []
			total, mean, variance = 0,0,0
			for j in range(numPrototypes):
				if i != j:
					featureDiff.append(prototypes[i].calculateDiff(prototypes[j]))

			mean = sum(featureDiff)/(numPrototypes - 1)
			total = sum([pow(diff - mean,2) for diff in featureDiff])

			variance = total/ float(numPrototypes - 2) ## is this actually how this is calculated

			#prototypes[i].setFeatureWidth(variance)
			prototypes[i].setFeatureWidth(1/variance)
	else:
		prototypes[0].setFeatureWidth(100)

# generates k different prototypes based on the isDifferent metric
def generatePrototypes():

	if isRandom:
		for i in range(numPrototypes):
			numDifferent = 0
			prototypes.append(Prototype(numActions, stateDimension))
			while numDifferent < i:
				if prototypes[i].isDifferent(prototypes[numDifferent]):
					numDifferent += 1
				else:
					prototypes[i].setRandomly()
					numDifferent = 0

	else :
		for i in range(numPrototypes):
			p = Prototype(numActions, stateDimension)
			p.setFixed([i/float(numPrototypes+1) ], 0)
			prototypes.append(p)

			

# returns the Q value for the state action pair
def getQ(state, action):
	tempPrototype = Prototype(numActions, stateDimension)
	tempPrototype.setFixed([state/float(maxState)], action)
	thetaSum = 0

	for prototype in prototypes:
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)
		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		# print('state: ' + str(tempPrototype.state) + '  prototype.state: ' + str(prototype.state) + '  membershipGrade: ' + str(membershipGrade) + '  theta: ' + str(prototype.getTheta()))
		# raw_input("Press Enter to continue...")
		thetaSum += prototype.getTheta() * membershipGrade

	# print('thetaSum: ' + str(thetaSum))
	# raw_input("Press Enter to continue...")

	return thetaSum


def getPrototypeDistribution(i):
	dist = []
	prototype = prototypes[i]
	for state in range(maxState):
		tempPrototype = Prototype(numActions, stateDimension)
		tempPrototype.setFixed([state/float(maxState)], 0)
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)
		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		dist.append(membershipGrade*prototype.getTheta())
	return dist


def getTileQ(state,action):



# updates the theta values of the prototypes
def learn(state1, action1, reward, state2):
	# maxQ = -float('inf')
	# for a in range(numActions):
	# 	tempQ = getQ(state2, a)
	# 	if (maxQ < tempQ):
	# 		maxQ = tempQ

	### sarsa

	maxQ = getQ(state2, 0)

	preQ = getQ(state1, action1)

	delta = reward + gamma*maxQ - preQ

	tempPrototype = Prototype(numActions, stateDimension)
	tempPrototype.setFixed([state1/float(maxState)], action1)

	for prototype in prototypes:
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)

		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		prototype.setTheta(prototype.getTheta() + alpha * delta * membershipGrade/numPrototypes)




numEpisodes = 300
numRuns = 2
maxState = 10
# runValue = [[[0  for i in range(maxState)] for j in range(maxState)] for k in range(numEpisodes)]
# distributionValue = [[[0 for i in range(numRuns * 5 + 5)] for j in range(maxState)] for k in range(numEpisodes)]
error = [0 for k in range(numEpisodes)]

w = [0]*72 # 8 overlapping 9 cell tilings

for run in range(1,numRuns):
	prototypes = []
	numPrototypes = run*5 + 5
	generatePrototypes()
	computePrototypeWidth()

	for episode in range(numEpisodes):
		if episode % 1000 == 0:
			print('episode ' + str(episode))
		lastState = 0
		nextState = 0
		while lastState < maxState-1:
			nextState = lastState + 1 # simple state transition
			learn(lastState, 0, 1, nextState) # the reward is +1 on the next state, only one action ("go forward")
			lastState = nextState

			# q = []
			# for i in range(maxState):
			# 	qi = getQ(i,0)
			# 	q.append(qi)
			# 	runValue[episode][lastState][i] = qi

			# for j in range(numPrototypes):
			# 	distributionValue[episode][lastState][j] = getPrototypeDistribution(j)

		for i in range(maxState):

			error[episode] += sqrt(pow((maxState - i) - getQ(i,0),2))

	# runValue = np.array(runValue)
	# distributionValue = [np.array([np.array(di) for di in d]) for d in distributionValue]

	# np.save('runValue-' + str(run*5 + 5) + '.npy', runValue)
	# np.save('distributionValue-' + str(run*5 + 5) + '.npy', distributionValue)


plt.plot(error)
plt.show()
