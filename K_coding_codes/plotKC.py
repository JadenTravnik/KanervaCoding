import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import math

maxState = 50
x = np.linspace(0,maxState,maxState)
y = [maxState - i + 1 for i in range(maxState)]

iterationNum = 1
numEpisodes = 200

runValue = np.load('runValue-' + str(iterationNum*5 + 5) + '.npy')
distributionValue = np.load('distributionValue-' + str(iterationNum*5 + 5) + '.npy')



#plt.ion()
fig = plt.figure()

ax = fig.add_subplot(111)
plt.ylim([-3,maxState+5])
plt.xlim([-1,maxState+1])
line0, = ax.plot(x,y, linewidth=5, linestyle='-', c='purple')
line1, = ax.plot(x,y, linewidth=5, linestyle='-', c='red')

distributions = []
for p in range(iterationNum*5 + 5):
	tempLine, = ax.plot(x,y,label=str(p))
	distributions.append(tempLine)

#plt.legend(loc='best')

def init():
	return line1,

def animate(iteration):
	global maxState
	j = iteration % maxState
	i =  int(math.floor(iteration/maxState))

	line1.set_ydata(np.array(runValue[i][j]))
	for k in range(iterationNum*5 + 5):

		distributions[k].set_ydata(np.array(distributionValue[i][j][k]))
	
	return [line1] + distributions

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=numEpisodes*maxState, interval=1, blit=True)

anim.save('KanervaCoding-' + str(iterationNum)  + '.mp4', fps=50, extra_args=['-vcodec', 'libx264'])

# for i in range(numEpisodes):
# 	for j in range(maxState): # maxstate
# 		line1.set_ydata(np.array(runValue[i][j]))
# 		for k in range(iterationNum*5 + 5):
# 			distributions[k].set_ydata(np.array(distributionValue[i][j][k]))
# 		fig.canvas.draw()
