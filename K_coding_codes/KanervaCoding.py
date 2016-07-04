from prototype import Prototype
from math import exp
import numpy as np
import matplotlib.pyplot as plt
import time

gamma = 1
epsilon = 0
alpha = .1

AlphaFactor = 0.99995
EpsilonFactor = 0.9995

numPrototypes = 10
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
			p.setFixed([(i+1)/float(numPrototypes+1) ], 0)
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
	# print('state1: ' + str(state1) + '  preq: ' + str(preQ) + ' state2: ' + str(state2))
	# raw_input("Press Enter to continue...")

	delta = reward + gamma*maxQ - preQ
	# print('state1: ' + str(state1) + ' state2: ' + str(state2) +  '  preQ: ' + str(preQ) + '  reward: ' + str(reward) + ' maxQ: ' + str(maxQ) + '  delta: ' + str(delta))

	tempPrototype = Prototype(numActions, stateDimension)
	tempPrototype.setFixed([state1/float(maxState)], action1)
	# print('state1: ' + str(state1) + '  tempPrototype.state: ' + str(tempPrototype.state))

	# print([p.getTheta() for p in prototypes])

	for prototype in prototypes:
		tempFeatureDiff = tempPrototype.calculateDiff(prototype)
		# print('Prototype: ' + str(prototype.state) + ' tempPrototype: ' + str(tempPrototype.state) + '  tempFeatureDiff: ' + str(tempFeatureDiff) + ' FeatureWidth: ' + str(prototype.getFeatureWidth()))

		membershipGrade = float(exp(-(tempFeatureDiff*tempFeatureDiff)/2*prototype.getFeatureWidth()))
		# print('membershipGrade: ' + str(membershipGrade) + ' theta: ' + str(prototype.getTheta()) + ' delta: ' + str(delta) + ' update: ' + str(prototype.getTheta() + alpha * delta * membershipGrade/numPrototypes))
		prototype.setTheta(prototype.getTheta() + alpha * delta * membershipGrade/numPrototypes)

		# raw_input("Press Enter to continue...")

	# print([p.getTheta() for p in prototypes])

	# raw_input("Press Enter to continue...")

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


numEpisodes = 300
numRuns = 1
maxState = 10
runValue = [[[[0  for i in range(maxState)] for j in range(maxState)] for k in range(numEpisodes)] for l in range(numRuns)]
distributionValue = [[[[0 for i in range(numPrototypes)] for j in range(maxState)] for k in range(numEpisodes)] for l in range(numRuns)]


for run in range(numRuns):
	prototypes = []
	generatePrototypes()
	computePrototypeWidth()

	for episode in range(numEpisodes):
		lastState = 0
		nextState = 0
		while lastState < maxState-1:
			nextState = lastState + 1 # simple state transition
			learn(lastState, 0, 1, nextState) # the reward is +1 on the next state, only one action ("go forward")
			lastState = nextState

			q = []
			for i in range(maxState):
				qi = getQ(i,0)
				q.append(qi)
				runValue[run][episode][lastState][i] = qi

			for j in range(numPrototypes):
				distributionValue[run][episode][lastState][j] = getPrototypeDistribution(j)



raw_input("Press Enter to graph")

x = np.linspace(0,maxState,maxState)
y = x
plt.ion()
fig = plt.figure()

ax = fig.add_subplot(111)
plt.ylim([-3,maxState+3])
plt.xlim([0,maxState])
line1, = ax.plot(x,y, 'r-')
distributions = []
for p in range(numPrototypes):
	tempLine, = ax.plot(x,y,label=str(p))
	distributions.append(tempLine)

#plt.legend(loc='best')

for i in range(numEpisodes):
	for j in range(maxState):
		line1.set_ydata(np.array(runValue[run][i][j]))
		for k in range(numPrototypes):
			distributions[k].set_ydata(np.array(distributionValue[run][i][j][k]))
		fig.canvas.draw()