import numpy
import torch
from math import sqrt

class BatchNorm(object):
	"""docstring for BatchNorm"""
	def __init__(self, isTrainable,input_dim,momentum=0.9,gamma='DEFAULT',beta='DEFAULT'):
		super(BatchNorm, self).__init__()
		self.isTrainable = isTrainable
		self.running_mean = torch.zeros(1, inputSize)
		self.running_var = torch.ones(1, inputSize)
		self.momentum = momentum
		if (gamma=='DEFAULT'):
			gamma = running_var
		if (beta=='DEFAULT'):
			beta = running_mean	
		self.gamma = gamma
		self.beta=beta
	def forward(self,input,mode='train'):
		if (mode=='train'):
			self.sample_mean = input.mean(dim=0)
			self.sample_var = input.var(dim=0)
			input_hat = (input - sample_mean) / torch.sqrt(sample_var + 1e-6)		
			self.running_mean = momentum * self.running_mean + (1 - momentum) * self.sample_mean
			self.running_var = momentum * self.running_var + (1 - momentum) * self.sample_var
			self.input_hat = input_hat
			return (gamma*input_hat + beta)
		elif (mode=='test'):
			return (gamma * (input - running_mean) / torch.sqrt(running_var + 1e-6) + beta)
		else :
			return "No such mode :/ "	
	def backward(self,input,gradOutput):
		self.gradBeta = gradOutput.sum(dim=0)
		self.gradGamma = torch.sum(gradOutput * self.input_hat,dim=0)
		gradVar = (gradOutput*gamma*(input-self.sample_mean)*-0.5*torch.pow(self.sample_var + 1e-6,-1.5)).sum(dim=0)
		gradMean1 = -1*(gradOutput*gamma/torch.sqrt(sample_var + 1e-6)).sum(dim=0)
		gradMean2 = gradVar * (-2 * (input-self.sample_mean)).sum(dim=0)
		gradMean = gradMean1 + gradMean2
		gradInput1 = (gradOutput*gamma/torch.sqrt(sample_var + 1e-6))
		gradInput2 = 1./N * (torch.ones(input.size)) * gradMean
		gradInput3 = 2./N * (input-self.sample_mean) * gradVar
		gradInput = gradInput1 + gradInput2 + gradInput3
		return gradInput
	def clear_grad(self):
		self.gradBeta = 0
		self.gradGamma = 0
		return	