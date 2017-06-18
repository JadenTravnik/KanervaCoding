import mountaincar
import invmountaincar
from ACRL import ACRL
import numpy as np
import random
import math
import kanerva
from DeepKNetwork import DeepKNetwork



class Experiment():
	def __init__(self):
		self.runs = 1
		self.numEpisodes = 2000
		self.maxEpisodeLen = 200
		self.data = np.zeros((self.runs, self.numEpisodes, self.maxEpisodeLen))

		self.prototypeList = [2, 2000]

		self.log_rate = int(self.maxEpisodeLen*.25)

		self.loop()
		np.save('deepkreward-2', self.data)


	def get_features(self, state):
		return self.deepKnet.GetFeatures([(state[0] + 1.2, (state[1] + .07) / .14)])


	def loop(self):
		for i_run in range(self.runs):
			self.deepKnet = DeepKNetwork(self.prototypeList)
			acrl = ACRL(n=self.prototypeList[-1])
			for i_episode in range(self.numEpisodes):
				state = invmountaincar.init()
#				self.acrl.Erace_Traces()
				for i_step in range(self.maxEpisodeLen):
					if i_step % self.log_rate == 0:
						print('run ' + str(i_run + 1) + '/' + str(self.runs) \
							 + ', episode ' + str(i_episode + 1) + '/' + str(self.numEpisodes) \
							 + ', step ' + str(i_step + 1) + '/' + str(self.maxEpisodeLen)) 
					prev_features = self.get_features(state)

					action = acrl.getAction(prev_features)
					reward, next_state = invmountaincar.sample(state, action)
					acrl.R = reward
			
					self.data[i_run, i_episode, i_step] = reward

	
					if i_step >= self.maxEpisodeLen \
							or next_state == None \
							or np.isnan(next_state[0]) \
							or np.isnan(next_state[1]):
						print('num steps: ' + str(i_step))
						print('Snext: ' + str(next_state))
						print('Breaking from episode: ' + str(i_episode))
						gotOut =  next_state == None \
							  or np.isnan(next_state[0]) \
							  or np.isnan(next_state[1])
						print('Ended episode ' + str(i_episode + 1) + '/' + str(self.numEpisodes) + ' on step ' + str(i_step + 1) + '/' + str(self.maxEpisodeLen)) 
							
						break
				
					acrl.Value(prev_features)
					acrl.Delta()
					next_features = self.get_features(next_state)
					acrl.Next_Value(next_features)
					acrl.Delta_Update()
					acrl.Average_Reward_Update()
					acrl.Trace_Update_Critic(prev_features)
					acrl.Weights_Update_Critic()
					acrl.Compatible_Features(action, prev_features)
					acrl.Trace_Update_Actor()
					acrl.Weights_Update_Actor()
					state = next_state

if __name__ == '__main__':
	Experiment()
