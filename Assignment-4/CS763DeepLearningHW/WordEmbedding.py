import torch
import numpy as np 

class WordEmbedding(object):
	"""docstring for WordEmbedding"""
	def __init__(self, V,D):
		super(WordEmbedding, self).__init__()
		self.V = V
		self.D = D
		self.EmbeddingMatrix = torch.ones(V,D)
		self.gradEmbeddingMatric = torch.zeros(V,D)
		self.isTrainable = False

	def forward(self, input):
		return self.EmbeddingMatrix[(input.numpy()).astype(int),:].view(-1,self.D)
			