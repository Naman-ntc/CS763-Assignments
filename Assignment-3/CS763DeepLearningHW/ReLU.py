import numpy
import torch
import math

class ReLU():
	"""docstring for Linear"""
	def __init__(self):
		super(ReLU, self).__init__()
		self.isTrainable = False
	def forward(self,input):
		self.output = input.clamp(min=0)
		return self.output
	def backward(self, input, gradOutput):
		self.gradInput = gradOutput.clone()
		self.gradInput[self.output==0] = 0
		return self.gradInput
	def __str__(self):
		string = "ReLU Layer"
	def clear_grad(self):
		self.gradInput = 0
		return