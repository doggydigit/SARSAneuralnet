from math import exp
import numpy as np


class InputNeuron():
    def __init__(self,x,y):
	    self.w = np.array([1.0, 1.0, 1.0]) # list of weight (1x3)
	    self.x = x # prefered position
	    self.y = y # prefered velocity
	    self.activation = 0.0 # activation of neuron
	    self.e = np.array([0.0, 0.0, 0.0])
        
    def UpdateNeuron(self, x, y, sigmax, sigmay):
	    self.activation = exp(-(((self.x-x)/(sigmax))**2)-(((self.y-y)/(sigmay))**2))
		
    def GetContribution(self, outputneuron):
	    return self.w[outputneuron]*self.activation
	    
    def UpdateTrace(self, action, lamgam):
        for i in range(3):
		    self.e[i] *= lamgam
	    
        self.e[action] += self.activation
	
    def ResetTrace(self):
		self.e = np.array([0.0, 0.0, 0.0])
	    
    def UpdateWeights(self, deltaQ):
	    for i in range(3):
		    self.w[i] += deltaQ * self.e[i]
	
    def GetMaxWeight(self):
		
		maxi = 0.0
		index = 1
		suum = 0.0
		for i in range(3):
			suum += self.w[i]
			if(self.w[i]>maxi):
				maxi = self.w[i]
				index = i
		return  (index-1)*maxi/suum

class Network():

    def __init__(self, nx = 5, ny = 5, tau = 0.0006, lam = 0.95, eta = 0.001, gamma = 0.95):#best is tau = 0.01 and eta = 0.001 with both decaying over time
        self.nx = nx # number neurons for abscisse ( = position)
        self.ny = ny # number neurons for ordinate ( = position derivative)
        self.sigmax = 180/(nx-1)
        self.sigmay = 30/(ny-1)
        self.tau = tau
        self.inputlayer = np.array([np.array([InputNeuron(-150+180/(nx-1)*i, -15+30/(ny-1)*j) for i in range(nx)]) for j in range(ny)])
        self.outputlayer = np.array([0.0, 0.0, 0.0])
        self.lastoutput = 0.0
        self.lam = lam
        self.gamma = gamma
        self.eta = eta
        
    def UpdateNeurons(self, x, y):
		self.outputlayer = [0.0, 0.0, 0.0]
		for i in range(self.nx):
		    for j in range(self.ny):
			    self.inputlayer[i][j].UpdateNeuron(x, y, self.sigmax, self.sigmay)
                for k in range(3):
				    self.outputlayer[k] += self.inputlayer[i][j].GetContribution(k) 
				       

    def UpdateWeights(self, action, reward):
		deltaQ = self.eta*(reward - self.lastoutput + self.gamma*self.outputlayer[action])
		
		for i in range(self.nx):
		    for j in range(self.ny):
				self.inputlayer[i][j].UpdateWeights(deltaQ)
		self.lastoutput = self.outputlayer[action]
        
    def UpdateTrace(self, action):
	    lamgam = self.lam*self.gamma
			
	    for i in range(self.nx):
		    for j in range(self.ny):
			    self.inputlayer[i][j].UpdateTrace(action, lamgam)
			    
    def ResetTraces(self):
		for i in range(self.nx):
			for j in range(self.ny):
				self.inputlayer[i][j].ResetTrace()
				
    def GetMaxAction(self, x, y):
		self.UpdateNeurons(x, y)
		maxi = 0.0
		a = 0.0
		for i in range(3):
			if(self.outputlayer[i]>maxi):
				maxi = self.outputlayer[i]
				a = i
		return a
		
    def GetAction(self, x, y):
	    self.UpdateNeurons(x, y)
	    
	    try:
			probability = [0,0,0]
			suum = 0.0
			for j in range(3):
			    suum += exp(self.outputlayer[j]/self.tau)
					
						
			for i in range(3):	    
				probability[i] = exp(self.outputlayer[i]/self.tau)/suum
			p = np.random.random()
			for i in range(3):
				p = p - probability[i]
				if p <= 0.0:
					self.UpdateTrace(i)
					return i
	    except OverflowError:
		    return np.argmax(self.outputlayer)
			    
    def save(self,nr):
		for k in range(3):
			arr = np.array([np.array([self.inputlayer[i][j].w[k] for i in range(self.nx)]) for j in range(self.ny)])
			np.savetxt('Memory/weights/w' +str(nr) + '-' + str(k) + '.txt', arr, fmt='%f')
		
    def load(self):
		for k in range(3):
			arr = np.loadtxt('Memory/test' + str(k) + '.txt')
			for i in range(self.nx):
				for j in range(self.ny):
					self.inputlayer[i][j].w[k] = arr[i][j]
				
		print('loaded')
            
