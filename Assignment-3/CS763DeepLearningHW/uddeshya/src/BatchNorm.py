import numpy
import torch
from math import sqrt

class BatchNorm():
	"""BatchNorm Layer"""
	def __init__(self, input_dim, eps=1e-6):
		super(BatchNorm, self).__init__()
		self.weight = torch.rand(1,input_dim)
		self.bias = torch.rand(1,input_dim)
		self.eps = eps
		self.isTrainable = True
		return

	def standardise(self, input):
		N,D = input.size()
		mu = (1./N)*input.mean(dim=0)
		xmu = input - mu
		sq = xmu**2
		var = (1./N)*sq.sum(dim=0)
		sqrtvar = var.sqrt()+self.eps
		ivar = 1./sqrtvar
		xhat = xmu*ivar
		return (xhat, xmu, ivar, sqrtvar, var)
	
	def forward(self,input):
		xhat,_,_,_,_ = self.standardise(input)
		gammax = self.weight * xhat
		self.output = gammax + self.bias
		return self.output
	
	def backward(self, input, gradOutput):
		N,D = gradOutput.size()
		xhat, xmu, ivar, sqrtvar, var = self.standardise(input)
		self.gradBias = gradOutput.sum(dim=0)
		gradGammaX = gradOutput
		t1 = gradGammaX*xhat
		self.gradWeight = t1.sum(dim=0)
		gradXhat = self.gradWeight*self.weight

		t2 = gradXhat*xmu
		gradIvar = t2.sum(dim=0)
		gradXmu1 = gradXhat*ivar

		gradSqrtvar = -1./(sqrtvar**2) * gradIvar

		t3 = var+self.eps
		gradVar = 0.5*1./t3.sqrt() * gradSqrtvar

		gradSq = 1./N * torch.ones(N,D) * gradVar

		gradXmu2 = 2*xmu*gradSq

		gradX1 = (gradXmu1 + gradXmu2)
		gradMu = -1*gradX1.sum(dim=0)

		gradX2 = 1./N * torch.ones(N,D) * gradMu

		self.gradInput = gradX1+gradX2
		return self.gradInput
	
	def __str__(self):
		string = "Batchnorm Layer with gamma %d beta %d eps %d"%(self.weight,self.bias, self.eps)
		return 	string
	
	def print_param(self):
		print("gamma :")
		print(self.weight)
		print("beta :")
		print(self.bias)
		print('eps :')
		print(self.eps)
	
	def clear_grad(self):
		self.gradInput = 0
		self.gradWeight = 0	
		self.gradBias= 0
		return