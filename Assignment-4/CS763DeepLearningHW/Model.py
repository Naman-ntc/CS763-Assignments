import torch
from RNN import *
from Linear import *
from WordEmbedding import *

class Model(object):
	"""docstring for Model"""
	def __init__(self, nLayers, H, V, D, isTrain):
		super(Model, self).__init__()
		self.nLayers = nLayers
		self.H = H
		self.V = V
		self.D = D
		self.isTrain = isTrain
		#self.embedding = WordEmbedding(V,D)
		self.RNN = RNN(D,H)
		self.fc1 = Linear(H,2)
		self.Layers = [self.fc1,self.RNN]
		
	def forward(self, input):
		#self.wordVec = self.embedding.forward(input)
		input = input.view(-1)
		n = input.size()[0]
		self.wordVec = torch.zeros(n,self.D)
		self.wordVec[np.arange(n),input.numpy()]
		self.out1 = self.RNN.forward(self.wordVec)
		self.out2 = self.fc1.forward(self.out1.view(1,-1))
		return self.out2

	def backward(self,input,gradOutput):
		gradOut1 = self.fc1.backward(self.out1.view(1,-1),gradOutput.view(1,-1))
		gradWordVec = self.RNN.backward(self.wordVec,gradOut1)
		#_ = self.embedding(input,gradWordVec)
		return

	def clearGradParam(self):
		for Layer in self.Layers:
			if Layer.isTrainable:
				Layer.clear_grad()		

	def dispGradParam(self):
		lenn = len(self.Layers)
		for i in range(lenn-1,-1,-1):
			print("Layer : %d"%(i))
			print(self.Layers[i])
			if (self.Layers[i].isTrainable):
				self.Layers[i].print_param()