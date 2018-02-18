import matplotlib
#matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt

class PCA():
	"""docstring for PCA"""
	def __init__(self, X):
		super(PCA, self).__init__()
		self.m = X.size()[0]
		self.n = X.size()[1]
		self.X_mean = torch.mean(X,dim=0)
		self.make_eigenvec(X-self.X_mean)
	def make_eigenvec(self,X):
		_, self.eigenvalues,self.eigenbasis = torch.svd(X)
		self.eigenvalues = self.eigenvalues * self.eigenvalues
		self.plot_eigenvalues()
	def plot_eigenvalues(self):
		plt.plot(self.eigenvalues)
	def give_basis_dim(self,dim):
		self.dim = dim
		self.eigenbasis = self.eigenbasis[:,:dim]
	def convert_data(self,X):
		X = X - self.X_mean
		return X.mm(self.eigenbasis)	