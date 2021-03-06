import torch
from math import sqrt 

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, input_dim, hidden_dim):
		super(RNN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.weight = torch.randn((input_dim+hidden_dim),hidden_dim) * sqrt(1.0/(input_dim+hidden_dim))
		self.bias = torch.randn(hidden_dim) * sqrt(1.0/(input_dim+hidden_dim))
		self.h0 = torch.randn(hidden_dim)
		self.gradBias = torch.zeros(hidden_dim)
		self.gradWeight = torch.zeros((input_dim+hidden_dim),hidden_dim)
		self.isTrainable = True

	def forward(self,input):
		"""
		Takes a input a sequence of T vectors each of D dimension T x D
		ncells equals number of timesteps we propogate calculated using 
		"""
		ncells = input.size()[0]
		self.hidden = torch.zeros(ncells+1,self.hidden_dim)
		self.hidden[0,:] = self.h0
		for i in range(ncells):
			#print(input[i,:].size(),self.hidden[i,:].size())
			temp_input = torch.cat((input[i,:],self.hidden[i,:])).view(1,-1)
			self.hidden[i+1,:] = temp_input.mm(self.weight) + self.bias
			self.hidden[i+1,:] = torch.tanh(self.hidden[i+1,:])
		#self.h0 = self.hidden[ncells,:]
		return self.hidden[ncells,:]

	def backward(self,input,gradOutput):
		"""
		Takes a input a sequence of T vectors each of D dimension T x D
		Also takes as input a gradOutput of 1xH dimensions
		ncells equals number of timesteps we backpropogate
		"""
		truncating = 20
		ncells = input.size()[0]
		gradHidden = torch.zeros(ncells+1,self.hidden_dim)
		gradHidden[ncells,:] = gradOutput
		gradInput = torch.zeros(input.size())
		tempgradBias = torch.zeros(self.hidden_dim)
		tempgradWeight = torch.zeros((self.input_dim+self.hidden_dim),self.hidden_dim)
		for i in reversed(range(ncells)):
			temp_input = torch.cat((input[i,:],self.hidden[i,:])).view(1,-1)
			dtemp = ((1-(self.hidden[i+1,:]**2))*gradHidden[i+1,:]).view(1,-1)
			if (ncells-i<= truncating):
				tempgradBias += torch.sum(dtemp,dim=0)
				tempgradWeight += temp_input.view((self.hidden_dim+self.input_dim),1).mm(dtemp.view(1,self.hidden_dim))
			gradHidden[i,:] = dtemp.mm(self.weight[self.input_dim:,:].t())
			gradInput[i,:] = dtemp.mm(self.weight[:self.input_dim,:].t())
		tempgradWeight /= truncating
		tempgradBias /= truncating	
		self.gradWeight += tempgradWeight
		self.gradBias += tempgradBias	
		# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		# print("gradWeight :")
		# print(tempgradWeight.max(),tempgradWeight.min(),tempgradWeight.mean(),tempgradWeight.median())
		# print("weight :")
		# print(self.weight.max(),self.weight.min(),self.weight.mean(),self.weight.median())
		# print("gradBias :")
		# print(tempgradBias.max(),tempgradBias.min(),tempgradBias.mean(),tempgradBias.median())
		# print("bias :")
		# print(self.bias.max(),self.bias.min(),self.bias.mean(),self.bias.median())
		# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
		return gradInput
	
	def clear_grad(self):
		self.gradWeight = 0
		self.gradBias = 0
		return

	def __str__(self):
		string = "RNN layer weight variable number of cells. Takes input a sequence (-1 , %d ) and outputs the last hidden state (%d)!!"%(self.input_dim,self.hidden_dim)
		return string

	def print_param(self):
		print("Weight :")
		print(self.weight)
		print("Bias :")
		print(self.bias)

	