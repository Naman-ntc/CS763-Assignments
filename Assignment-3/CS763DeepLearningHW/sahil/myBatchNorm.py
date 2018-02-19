from imports import *
import torch
import numpy as np 

class BatchNorm():
	"""docstring for batchnorm"""

	def __init__(self, inputSize):
		self.isTrainable = False
		self.size = inputSize
		self.mean = torch.zeros(1, inputSize)
		self.std = torch.ones(1, inputSize)
		self.lam = 0.2
		return

	def __str__(self):
		return "A batch norm layer which uses a running mean and std, giving " + str(self.lam * 100) + "% weight to the current values"

	def forward(self, input):
		return (input - self.mean)/self.std

	def backward(self, input, upstreamGrad):
		currentMean = input.mean(dim = 0)
		currentMean = currentMean.view(1, self.size)
		self.mean = (1 - self.lam)*self.mean + self.lam * currentMean

		currentStd = input.std(dim = 0)
		currentStd = currentStd.view(1, self.size)
		self.std = ((1 - self.lam)**2)*((self.std)**2) + (self.lam ** 2)*(currentStd**2)
		self.std = torch.sqrt(self.std)
		return upstreamGrad / self.std

	def resetParam(self, mean = 0, std = 1, inputSize = None):
		if inputSize == None:
			inputSize = self.size
		self.mean = torch.zeros(1, inputSize)
		self.std = torch.ones(1, inputSize)
		return 


	def clear_grad(self):
		return