import numpy as np
from math import *
import random


class ACRL():
	def __init__(self,gamma = 1, alphaR = 0, alphaV = 0.001, alphaU = 0.0001, lmbda = 0.75, n=1000):
		self.gamma = gamma
		self.alphaR = alphaR
		self.alphaV = alphaV
		self.alphaU = alphaU
		self.lmbda = lmbda
		self.action = 0
		
		self.avgR = 0
		self.ev = np.zeros(n)
		self.e_mu = np.zeros(n)
		self.e_sigma = np.zeros(n)
		
		self.w = np.zeros(n)
		self.u_mu = np.zeros(n)
		self.u_sigma = np.zeros(n)
		
		self.delta = 0.0
		self.R = 0.0
		self.value = 0.0
		self.nextValue = 0.0
		
		self.compatibleFeatures_mu = np.zeros(n)
		self.compatibleFeatures_sigma = np.zeros(n)
		
		self.mean = 0.0
		self.sigma = 1.0
		self.n = n
    
	def Value(self,features):
		Val = 0.0
		for index in features:
			Val += self.w[index]
		self.value = Val
	
	def Next_Value(self,features):
		Val = 0.0
		for index in features:
			Val += self.w[index]
		self.nextValue = Val
	
	def Delta(self):
		self.delta = self.R - self.avgR - self.value
    
	def Delta_Update(self):
		self.delta += self.gamma*self.nextValue
    
	def Trace_Update_Critic(self,features):
		self.ev = self.gamma*self.lmbda*self.ev
		for index in features:
			self.ev[index] += 1
    
	def Trace_Update_Actor(self):
		self.e_mu = self.gamma * self.lmbda * self.e_mu + self.compatibleFeatures_mu
		self.e_sigma = self.gamma * self.lmbda * self.e_sigma + self.compatibleFeatures_sigma

	def Weights_Update_Critic(self):
		self.w += self.alphaV * self.delta * self.ev
	
	def Weights_Update_Actor(self):
		self.u_mu += self.alphaU * self.delta * self.e_mu
		self.u_sigma += self.alphaU * self.delta * self.e_sigma 
	
	def Compatible_Features(self,action,features):

		self.compatibleFeatures_mu = np.zeros(self.n)
		self.compatibleFeatures_sigma = np.zeros(self.n)
	

		# scaling factor of mean
		mcf = ((self.action - self.mean)) #/(pow(self.sigma,2))) 

		#scaling factor of sigma
		scf = (pow((self.action - self.mean),2)) -(pow(self.sigma,2)) # - 1

		if scf > 1000:
			print('scf: ' + str(scf) + ' mcf: ' + str(mcf) + '  td error: ' + str(self.delta))

		for index in features:
			self.compatibleFeatures_mu[index] = mcf
			self.compatibleFeatures_sigma[index] = scf
	    
	def Average_Reward_Update(self):
		self.avgR += self.alphaR * self.delta

	def getAction(self,features):
		self.mean = 0.0
		self.sigma = 0.0
		for index in features:
			self.mean += self.u_mu[index]
			self.sigma += self.u_sigma[index]
	
		#print self.sigma
		#print('First: ' + str(self.sigma))
		#if isnan(self.sigma) or abs(self.sigma) > 10:
	#		self.sigma = .5
		self.sigma = exp(self.sigma)
		#print('Second: ' + str(self.sigma))
		if self.sigma < 1:
		    self.sigma = 1  	 
		#print self.mean,self.sigma 	
		a = np.random.normal(self.mean,self.sigma)
		self.action = a
		#a /= 100


		if a > 1:
		    a = 0.99
		if a < -1:
		    a = -0.99
		#print a
		return a
	
	def Erase_Traces(self):
		self.e_mu = np.zeros(self.n)
		self.ev = np.zeros(self.n)
		self.e_sigma = np.zeros(self.n)
