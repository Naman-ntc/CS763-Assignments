import torch

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, input_dim,hidden_dim):
		super(RNN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.W = torch.randn((input_dim+hidden_dim),hidden_dim)
		self.B = torch.randn(hidden_dim)
		self.h0 = torch.zeros(hidden_dim)
		self.gradB = torch.zeros(hidden_dim)
		self.gradW = torch.zeros((input_dim+hidden_dim),hidden_dim)
		
	def forward(self,input):
		"""
		Takes a input a sequence of T vectors each of D dimension T x D
		ncells equals number of timesteps we propogate calculated using 
		"""
		ncells = input.size()[0]
		self.hidden = torch.zeros(ncells+1,self.hidden_dim)
		self.hidden[0,:] = self.h0
		for i in range(ncells):
			temp_input = torch.cat(input[i,:],hidden[i,:])
			self.hidden[i+1,:] = temp_input.mm(self.W) + self.B
			self.hidden[i+1,:] = torch.tanh(self.hidden[i+1,:])
		self.h0 = self.hidden[ncells,:]
		return self.h0

	def backward(self,input,gradOutput):
		"""
		Takes a input a sequence of T vectors each of D dimension T x D
		Also takes as input a gradOutput of 1xH dimensions
		ncells equals number of timesteps we backpropogate
		"""
		ncells = input.size()[0]
		gradHidden = torch.zeros(ncells+1,self.hidden_dim)
		gradHidden[ncells,:] = gradOutput
		gradInput = torch.zeros(input.shape())
		for i in reversed(range(ncells)):
			temp_input = torch.cat(input[i,:],hidden[i,:])
			dtemp = (1-torch.square(self.hidden[i+1,:]))*gradHidden[i+1,:]
			self.gradB += torch.sum(dtemp,dim=0)
			self.gradW += temp_input.view((self.hidden_dim+self.input_dim),1).mm(dtemp.view(1,self.hidden_dim))
			gradHidden[i,:] = dtemp.mm(self.W[input_dim:,:].t())
			gradInput[i,:] = dtemp.mm(self.W[:input_dim,:].t())
		return gradInput
	
	def clear_grad(self):
		self.gradW = 0
		self.gradB = 0
		return

	def __str__(self):
		string = "RNN layer with variable number of cells. Takes input a sequence (-1 , %d ) and outputs the last hidden state (%d)!!"%(input_dim,hidden_dim)
		return string

	def print_param(self):
		print("Weight :")
		print(self.W)
		print("Bias :")
		print(self.B)

	