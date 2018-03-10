import torch
from RNN import *
from Linear import *

class Model(object):
	"""docstring for Model"""
	def __init__(self, nLayers, H, V, D, isTrain):
		super(Model, self).__init__()
		self.nLayers = nLayers
		self.H = H
		self.V = V
		self.D = D
		self.isTrain = isTrain
		self.RNN = RNN(D,H)
		self.fc1 = Linear(H,2)

	def forward(self,input):
		self.out1 = self.RNN.forward(input)
		self.out2 = self.fc1.forward(out1)
		return out2

	def backward(self,input,gradOutput):
		gradOut1 = self.fc1(self.out1,gradOutput)
		return self.RNN(input,gradOut1)