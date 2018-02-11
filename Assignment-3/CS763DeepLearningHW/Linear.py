import numpy
import torch

class Linear(object):
	"""docstring for Linear"""
	def __init__(self, input_dim,output_dim):
		super(Linear, self).__init__()
		self.input_dim = input_dim
		self.output_dim = output_dim 
	
	def forward(self,input):
			