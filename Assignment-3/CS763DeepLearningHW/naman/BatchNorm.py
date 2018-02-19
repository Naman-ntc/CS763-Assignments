import numpy
import torch
from math import sqrt

class BatchNorm(object):
	"""docstring for BatchNorm"""
	def __init__(self, input_dim,isTrainable=True,momentum=0.9,gamma=1,beta=0):
		super(BatchNorm, self).__init__()
		self.isTrainable = isTrainable
		self.running_mean = torch.zeros(input_dim).mean(dim=0).type(torch.DoubleTensor)
		self.running_var = torch.ones(input_dim).mean(dim=0).type(torch.DoubleTensor)
		self.momentum = momentum
		if ((gamma==1).all()):
			print("#")
			gamma = self.running_var
		if ((beta==0).all()):
			print("#")
			beta = self.running_mean	
		self.gamma = gamma
		self.beta=beta
	def forward(self,input,mode='train'):
		if (mode=='train'):
			self.sample_mean = input.mean(dim=0)
			self.sample_var = input.var(dim=0,unbiased=False)
			input_hat = (input - self.sample_mean) / torch.sqrt(self.sample_var + 1e-6)		
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.sample_mean
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.sample_var
			self.input_hat = input_hat
			return (self.gamma*self.input_hat + self.beta)
		elif (mode=='test'):
			return (self.gamma * (self.input - self.running_mean) / torch.sqrt(self.running_var + 1e-6) + self.beta)
		else :
			return "No such mode :/ "	
	def backward(self,input,gradOutput):
		N = input.shape[0]
		self.gradBeta = gradOutput.sum(dim=0)
		self.gradGamma = torch.sum(gradOutput * self.input_hat,dim=0)
		gradVar = (gradOutput*self.gamma*(input-self.sample_mean)*-0.5*torch.pow(self.sample_var + 1e-6,-1.5)).sum(dim=0)
		gradMean1 = -1*(gradOutput*self.gamma/torch.sqrt(self.sample_var + 1e-6)).sum(dim=0)
		gradMean2 = gradVar * (-2 * (input-self.sample_mean)).sum(dim=0)
		gradMean = gradMean1 + gradMean2
		print(gradMean.type())
		gradInput1 = (gradOutput*self.gamma/torch.sqrt(self.sample_var + 1e-6))
		gradInput2 = (torch.ones(input.size())).type(torch.DoubleTensor) * gradMean * 1./N  
		gradInput3 = (input-self.sample_mean) * gradVar * 2./N
		gradInput = gradInput1 + gradInput2 + gradInput3
		return (gradInput,self.gradGamma,self.gradBeta)
	def clear_grad(self):
		self.gradBeta = 0
		self.gradGamma = 0
		return	
	def reset_running_pars(self):
		self.running_mean = torch.zeros(input_dim).mean(dim=0).type(torch.DoubleTensor)
		self.running_var = torch.ones(input_dim).mean(dim=0).type(torch.DoubleTensor)