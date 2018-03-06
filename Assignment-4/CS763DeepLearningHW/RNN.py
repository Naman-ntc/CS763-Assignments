import torch

class RNN(object):
	"""docstring for RNN"""
	def __init__(self, input_dim,hidden_dim):
		super(RNN, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.W = torch.randn((input_dim+hidden_dim),hidden_dim)
		self.b = torch.randn(hidden_dim,1)

	def forward(self,input):
		self.output = input.mm(W) + self.b
	
	def backward(self,input,gradOutput):
		self.gradB = gradOutput.sum(dim=0)
		self.gradW = input.t().mm(gradOutput)
		self.gradInput = gradOutput.mm(self.output.t())
		return self.gradInput
		