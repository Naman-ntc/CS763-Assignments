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
		self.fcin = Linear(V,D)
		self.fc1 = Linear(H,2)

	def forward(self, input):
		self.wordVec = self.fcin.forward(input)
		self.out1 = self.RNN.forward(self.wordVec)
		self.out2 = self.fc1.forward(self.out1)
		return self.out2

	def backward(self,input,gradOutput):
		gradOut1 = self.fc1(self.out1,gradOutput)
		gradWordVec = self.RNN(self.word_vec,gradOut1)
		_ = self.fcin(input,gradWordVec)
		return