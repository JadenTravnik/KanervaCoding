from math import exp, sqrt
import numpy as np
import matplotlib.pyplot as plt
import time

from KC import KanervaCoding1D
from Tilecoder import TileCoder1D


gamma = 1
epsilon = 0
alpha = .1

stateDimension = 1

numEpisodes = 30000
numRuns = 1
maxState = 10
# runValue = [[[0  for i in range(maxState)] for j in range(maxState)] for k in range(numEpisodes)]
# distributionValue = [[[0 for i in range(numRuns * 5 + 5)] for j in range(maxState)] for k in range(numEpisodes)]
errorTC = [0 for i in range(numEpisodes)]
errorKC = [0 for i in range(numEpisodes)]

tc = TileCoder1D(3, 8, maxState)
kc = KanervaCoding1D(20, maxState)

def learn(state1, reward, state2):

	maxVTilecode = tc.getV(state2)
	preVTilecode = tc.getV(state1)

	delta = reward + gamma*maxVTilecode - preVTilecode
	tc.updateWeights(state1, delta, alpha)

	maxVKanervaCode = kc.getV(state2)
	preVKanervaCode = kc.getV(state1)

	delta = reward + gamma*maxVKanervaCode - preVKanervaCode
	kc.updateWeights(state1, delta, alpha)

for run in range(numRuns):

	for episode in range(numEpisodes):
		if episode % 1000 == 0:
			print('episode ' + str(episode))
		lastState = 0
		nextState = 0
		while lastState < maxState-1:
			nextState = lastState + 1 # simple state transition
			learn(lastState, 1, nextState) # the reward is +1 on the next state
			lastState = nextState

		for i in range(maxState):

			errorTC[episode] += sqrt(pow((maxState - i) - tc.getV(i),2))
			errorKC[episode] += sqrt(pow((maxState - i) - kc.getV(i),2))

plt.gca().set_color_cycle(['red', 'green'])

plt.plot(errorTC)
plt.plot(errorKC)
plt.show()
