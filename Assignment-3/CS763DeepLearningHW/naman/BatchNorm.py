import numpy
import torch
from math import sqrt

class BatchNorm(object):
	"""docstring for BatchNorm"""
	def __init__(self, isTrainable):
		super(BatchNorm, self).__init__()
		self.isTrainable = isTrainable
	def forward(self,input,output):
				
		