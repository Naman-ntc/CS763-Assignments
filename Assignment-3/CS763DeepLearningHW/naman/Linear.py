import numpy
import torch
from math import sqrt

class Linear():
	"""docstring for Linear"""
	def __init__(self, input_dim,output_dim,initialization='He'):
		super(Linear, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim 
		self.weight = torch.randn(input_dim,output_dim)*sqrt(2/(input_dim))
		self.bias = torch.randn(1,output_dim)*sqrt(2/(input_dim+output_dim))
		self.isTrainable = True
		return
	def forward(self,input):
		self.output = input.mm(self.weight) + self.bias
		return self.output
	def backward(self, input, gradOutput):
		self.gradInput = gradOutput.mm(self.weight.t())
		self.gradWeight = input.t().mm(gradOutput)
		self.gradBias = gradOutput.sum(dim=0).view(1,self.output_dim)
		return self.gradInput
	def __str__(self):
		string = "Linear Layer with input dimensions %d and output dimensions %d"%(self.input_dim,self.output_dim)
		return 	string
	def print_param(self):
		print("Weight :")
		print(self.weight)
		print("Bias :")
		print(self.bias)
	def clear_grad(self):
		self.gradInput = 0
		self.gradWeight = 0	
		self.gradBias = 0
		return
	def weights_norm(self):
		return torch.norm(self.weight) + torch.norm(self.bias)