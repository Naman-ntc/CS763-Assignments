import torch
from RNN import *

class Model(object):
	"""docstring for Model"""
	def __init__(self, nLayers, H, V, D, isTrain):
		super(Model, self).__init__()
		self.nLayers = nLayers
		self.H = H
		self.V = V
		self.D = D
		self.isTrain = isTrain
		self.build()

	def build(self):
		self.cells = [None]*self.nLayers
		for i in range(self.nLayers):
			self.cells[i] = RNN(D,H)

	def forward(self,input):
		for cell in self.cells:
			