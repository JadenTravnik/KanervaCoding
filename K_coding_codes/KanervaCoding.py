from prototype import Prototype
from math import exp
import numpy as np

gamma = 0.999
epsilon = .2
alpha = .1

AlphaFactor = 0.99995
EpsilonFactor = 0.9995

numPrototypes = 30
numActions = 1
stateDimension = 1
prototypes = []

# computes the prototype width which is the variance of distance between all other prototypes
def computePrototypeWidth():
	featureDiff = []
	for i in range(numPrototypes):
		total, mean, variance = 0,0,0
		for j in range(numPrototypes):
			if j != k:
				featureDiff[idx] = prototypes[i].calculateDiff(prototypes[j])

		mean = sum(featureDiff)/(numPrototypes - 1)
		total = sum([pow(diff - mean,2) for diff in featureDiff])

		variance = total/ (numPrototpyes - 1 - 1) ## is this actually how this is calculated
		prototype[i].setFeatureWidth(variance)

# generates k different prototypes based on the isDifferent metric
def generatePrototypes(numPrototypes):
	for i in range(numPrototypes):
		numDifferent = 0
		prototypes.append(Prototype(numActions, stateDimension))
		while numDifferent < i:
			if prototypes[i].isDifferent(prototypes[numDifferent]):
				numDifferent += 1
			else:
				prototypes[i].setRandomly()
				numDifferent = 0

# returns the Q value for the state action pair
def getQ(state, action):
	tempPrototype = Prototype(state, action)
	thetaSum = 0

	for prototype in prototypes:
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)
		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		thetaSum += prototype.getTheta() * membershipGrade

	return thetaSum

# updates the theta values of the prototypes
def learn(state1, action1, reward, state2):
	maxQ = -float('inf')
	for a in range(numActions):
		tempQ = getQ(state2, a)
		if (maxQ < tempQ):
			maxQ = tempQ

	preQ = getQ(state1, action1)
	delta = reward + gamma*maxQ - preQ

	tempPrototype = Prototype(state1, action1)

	for prototype in prototypes:
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)
		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		prototype.setTheta(prototype.getTheta() + alpha * delta * membershipGrade/numPrototypes)

def chooseAction(state):
	if random.random() > epsilon:
		return random.randrange(0,numActions)
	else:
		return chooseBestAction(state) 

def chooseBestAction(state):
	action = 0
	maxQ = -inf
	q_list = []
	bestActionList = []

	for a in range(numActions):
		qValue = getQ(state, a)
		q_list.append(qValue)

	for i in range(len(q_list)):
		if (maxQ < q_list[i]):
				maxQ = q_list[i]
				bestActionList = [] # clear the best values

		if maxQ == q_list:
			bestActionList.append(i) # add the action to the best value array


	if bestV.size() > 1: # if there is more than 1 best value
			index = rand() % bestV.size() # choose randomly
			action - bestActionList[index]
	else:
		action = bestActionList[0]

	return action


generatePrototypes(numPrototypes)

print(prototypes)


numEpisodes = 50
numRuns = 1
maxState = 5
runValue = [[[0 for i in range(maxState)] for i in range(numEpisodes)] for i in range(numRuns)]

for run in range(numRuns):
	print('Run: ' + str(run) + '. New prototypes generated')
	prototypes = []
	generatePrototypes(numPrototypes)

	for episode in range(numEpisodes):
		lastState = 0
		print('Episode: ' + str(episode))
		while lastState < maxState:
			print('State: ' + str(lastState))
			nextState = lastState + 1 # simple state transition
			learn(lastState, 0, 1, nextState) # the reward is +1 on the next state, only one action ("go forward")
			runValue[run][episode][lastState] = getQ(lastState,0)
			lastState = nextState


		print(runValue)