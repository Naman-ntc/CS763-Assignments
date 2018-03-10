import torch

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, input_dim,hidden_dim):
		super(RNN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.W = torch.randn((input_dim+hidden_dim),hidden_dim)
		self.b = torch.randn(hidden_dim,1)
		self.h0 = torch.zeros(hidden_dim)

	def forward(self,input):
		"""
		Takes a input a sequence of T vectors each of D dimension T x D
		ncells equals number of timesteps we propogate
		"""
		ncells = input.size()[0]
		self.hidden = torch.zeros(ncells+1,self.hidden_dim)
		self.hidden[0,:] = self.h0
		for i in range(ncells):
			temp_input = torch.cat(input,hidden[i,:])
			self.hidden[i+1,:] = temp_input.mm(self.W) + self.b
			self.hidden[i+1,:] = torch.tanh(self.hidden[i+1,:])
		self.h0 = self.hidden[ncells,:]
		return self.h0

	def backward(self,input,gradOutput):
		self.gradB = gradOutput.sum(dim=0)
		self.gradW = input.t().mm(gradOutput)
		self.gradInput = gradOutput.mm(self.output.t())
		return self.gradInput
		