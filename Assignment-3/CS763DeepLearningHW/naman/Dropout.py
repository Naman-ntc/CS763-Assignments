import numpy
import torch
from math import sqrt

class Dropout(object):
	"""docstring for Dropout"""
	def __init__(self):
		super(Dropout, self).__init__()
		self.isTrainable = False
	def forward(self,input,keep_prob):
		self.mask = torch.rand(input.size()) < keep_prob
		output = self.mask * input
		return output
	def backward(self,input,gradOutput):
		gradInput = self.mask * gradOutput
		return gradInput
		