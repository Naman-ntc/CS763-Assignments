import numpy
import torch
from math import sqrt

class Linear():
	"""docstring for Linear"""
	def __init__(self, input_dim, output_dim, initialization='Xavier'):
		super(Linear, self).__init__()
		self.input_dim = input_dim #batch first
		self.output_dim = output_dim #batch first
		self.weight = torch.randn(input_dim, output_dim)*sqrt(2/(input_dim+output_dim))
		self.bias = torch.randn(1, output_dim)*sqrt(2/(input_dim+output_dim))
		self.isTrainable = True
		return
	def forward(self,input):
		#self.output = self.weight.mm(self.input.t()) + self.bias
		self.output = input.mm(self.weight) + self.bias
		return self.output
	def backward(self, input, gradOutput):
		#gradOutput/ gradInput sampe dimenstions as output/ input
		self.gradInput = gradOutput.mm(self.weight.t())
		self.gradWeight = input.t().mm(gradOutput)
		self.gradBias = gradOutput.sum(dim=0).view(1,self.output_dim)
		return self.gradInput
	def __str__(self):
		str_out = 'Linear Layer with input dimensions (batch firts) {} and output dimensions (batch first) {}'.format(
			self.input_dim, self.output_dim)
		return 	str_out
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