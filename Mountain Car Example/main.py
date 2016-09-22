import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from ACRL import ACRL
from Tilecoder import numTiles as n
from pylab import *
import numpy as np
import random
import math
import kanerva
from BaseKanervaCoder import BaseKanervaCoder as bkc
from VisitCounterKanerva import VisitCounterKanerva as vck
from VisitCounterCorrelationKanerva import VisitCounterCorrelationKanerva as vcck
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

n = 500

numEpisodes = 200
runSum = 0.0

timeSteps = np.zeros((1,numEpisodes))
returns = np.zeros((1,numEpisodes))

returnSum = 0.0

k = vcck(n,2)

fig, ax = plt.subplots()

prototypePlot, acrl = {}, {}

x, y, s, t = [], [], [], []
steps = 0
episodeNum = 0
gotOut = False
S = [0,0]

for prototype in k.prototypes:
	x.append(prototype[0] * 1.7 - 1.2)
	y.append(prototype[1] * .14 - .07)
	t.append(0)
	s.append(5)

x.append(0)
t.append(0)
y.append(0)
s.append(15)


acrl = ACRL()

def setupPlot():
	global x, y, t, prototypePlot

	prototypePlot = ax.scatter(x, y, c=t, s=10, animated=True, cmap = plt.matplotlib.cm.jet)
	ax.axis=([-1.5, .8, -.1, .1])
	return prototypePlot,

def clamp(arr, minn, maxn):
	for i in range(len(arr)):
		arr[i] = maxn + arr[i]
	return arr

def get_features(S, update):
    #F = np.zeros(numTilings)
    #tilecode(S[0],S[1],F)
    #return F

	features = k.GetFeatures([(S[0]+1.2)/1.7,(S[1]+.07)/.14], update)

	return features

def updatePrototypePlacements():
	for prototype in k.updatedPrototypes:
		try:
			x[prototype[0]] = (prototype[1][0] * 1.7) - 1.2
			y[prototype[0]] = (prototype[1][1] * .14) - .07
			acrl.w[prototype[0]] = acrl.w[prototype[2]]			
		except:
			print('prototype[0]: ' + str(prototype[0]))
			print('prototype[1]: ' + str(prototype[1]))
			print('len(x): ' + str(len(x)))


	wAvg = 0
	u_muAvg = 0
	u_sigmaAvg = 0
	e_muAvg = 0
	evAvg = 0
	e_sigmaAvg = 0
	compatibleFeatures_muAvg = 0
	compatibleFeatures_sigmaAvg = 0
	for prototype in k.deletedPrototypes:
		del x[prototype]
		del y[prototype]
		wAvg += acrl.w[prototype]
		u_muAvg += acrl.u_mu[prototype]
		u_sigmaAvg += acrl.u_sigma[prototype]
		e_muAvg += acrl.e_mu[prototype]
		evAvg += acrl.ev[prototype]
		e_sigmaAvg += acrl.e_sigma[prototype]
		compatibleFeatures_muAvg += acrl.compatibleFeatures_mu[prototype]
		compatibleFeatures_sigmaAvg += acrl.compatibleFeatures_sigma[prototype]
		print('Got avgs')
		acrl.w = np.delete(acrl.w, prototype, axis=0)
		acrl.u_mu = np.delete(acrl.u_mu, prototype, axis=0)
		acrl.u_sigma = np.delete(acrl.u_sigma, prototype, axis=0)
		acrl.e_mu = np.delete(acrl.e_mu, prototype, axis=0)
		acrl.ev = np.delete(acrl.ev, prototype, axis=0)
		acrl.e_sigma = np.delete(acrl.e_sigma, prototype, axis=0)
		acrl.compatibleFeatures_mu = np.delete(acrl.compatibleFeatures_mu, prototype, axis=0)
		acrl.compatibleFeatures_sigma = np.delete(acrl.compatibleFeatures_sigma, prototype, axis=0)
		acrl.n -= 1		

	avg = len(k.deletedPrototypes)*1.0
	for prototype in k.addedPrototypes:
		x.append(prototype[0])
		y.append(prototype[1])
		acrl.w = np.append(acrl.w, np.array([wAvg/avg]), axis=0)
		acrl.u_mu = np.append(acrl.u_mu, np.array([u_muAvg/avg]), axis=0)
		acrl.u_sigma = np.append(acrl.u_sigma, np.array([u_sigmaAvg/avg]), axis=0)
		acrl.e_mu = np.append(acrl.e_mu, np.array([e_muAvg/avg]), axis=0)
		acrl.ev = np.append(acrl.ev, np.array([evAvg/avg]), axis=0)
		acrl.e_sigma = np.append(acrl.e_sigma, np.array([e_sigmaAvg/avg]), axis=0)
		acrl.compatibleFeatures_mu = np.append(acrl.compatibleFeatures_mu, np.array([compatibleFeatures_muAvg/avg]), axis=0)
		acrl.compatibleFeatures_sigma = np.append(acrl.compatibleFeatures_sigma, np.array([compatibleFeatures_sigmaAvg/avg]), axis=0)
		acrl.n += 1

	k.EmptyArrays()


def updatePlot(i):
	global prototypePlot, steps, S, x, y, s, gotOut

	if gotOut:
		speed = 1
	else:
		speed = 1

	for i in range(speed):
		res = loop()

		if steps >= 5000 or res:
			S = mountaincar.init()
			print('init mc')
			steps = 0
			acrl.Erase_Traces()
			break

	s = [50]*n +[200] # reset sizes

	for index in get_features(S, False): # change sizes of the close prototypes
		s[index] = 100

	prototypePlot._sizes = s

	x[-1], y[-1] = S[0], S[1] # update the state
	prototypePlot.set_offsets(zip(x,y)) # display the points
	values = clamp(acrl.w + [acrl.value], 0, 1)
	prototypePlot.set_array(values) # display the values

	return prototypePlot,	

def loop():
	global S, acrl, steps, episodeNum, gotOut

	prev_features = get_features(S, True)
	updatePrototypePlacements()

	if steps % 100 == 0:
		print('num steps: ' + str(steps))


	A = acrl.getAction(prev_features)
	R,Snext = mountaincar.sample(S,A)
	acrl.R = R
	if steps >= 5000 or Snext == None or isnan(Snext[0]) or isnan(Snext[1]):
		print('num steps: ' + str(steps))
		print('Snext: ' + str(Snext))
		print('Breaking from episode: ' + str(episodeNum))
		episodeNum += 1
		if Snext == None or isnan(Snext[0]) or isnan(Snext[1]):
			gotOut = True
		return True
	acrl.Value(prev_features)
	acrl.Delta()
	next_features = get_features(Snext, False)
	acrl.Next_Value(next_features)
	acrl.Delta_Update()
	acrl.Average_Reward_Update()
	acrl.Trace_Update_Critic(prev_features)
	acrl.Weights_Update_Critic()
	acrl.Compatible_Features(A,prev_features)
	acrl.Trace_Update_Actor()
	acrl.Weights_Update_Actor()
	S = Snext
	steps += 1
	return False

if __name__ == '__main__':

	ani = animation.FuncAnimation(fig, updatePlot, interval=1, init_func=setupPlot, blit=True)
	plt.show()

	print "Average return:", returnSum/numEpisodes
	runSum += returnSum
	print "Overall average return:", runSum/numEpisodes

