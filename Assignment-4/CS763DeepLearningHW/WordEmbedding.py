import torch
import numpy as np 

class WordEmbedding(object):
	"""docstring for WordEmbedding"""
	def __init__(self, V,D):
		super(WordEmbedding, self).__init__()
		self.V = V
		self.D = D
		self.EmbeddingMatrix = torch.randn(V,D)

	def forward(self, input):
		return self.EmbeddingMatrix(input.numpy().astype(int))

	def backward(self, gradOutput):
			