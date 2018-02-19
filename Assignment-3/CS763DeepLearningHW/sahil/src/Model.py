import numpy
import torch
import math

class Model():
	"""docstring for Model"""
	def __init__(self):
		super(Model, self).__init__()
		self.Layers = []
		self.isTrain = True
	
	def addLayer(self,Layer):
		self.Layers.append(Layer)	
	
	def forward(self,myinput):
		lenn = len(self.Layers)
		self.inputs = [None]*(1+lenn)
		self.inputs[0] = myinput
		for i in range(lenn):
			self.inputs[i+1] = self.Layers[i].forward(self.inputs[i])
		return self.inputs[lenn]
	def backward(self,myinput,gradOutput): # input is something extra we dont wanna know
		#output = self.forward(myinput)
		curr_grad = gradOutput.clone()
		lenn = len(self.Layers)
		for i in range(lenn-1,-1,-1):
			curr_grad = (self.Layers[i]).backward(self.inputs[i], curr_grad)
		#return curr_grad
		return 
	def dispGradParam(self):
		lenn = len(self.Layers)
		for i in range(lenn-1,-1,-1):
			print("Layer : %d"%(i))
			print(self.Layers[i])
			if (self.Layers[i].isTrainable):
				self.Layers[i].print_param()
	def clearGradParam(self):
		for Layer in self.Layers:
			Layer.clear_grad()	

	def saveMeanVariance(self, mean, variance):
		self.dataMean = mean
		self.dataVariance = variance