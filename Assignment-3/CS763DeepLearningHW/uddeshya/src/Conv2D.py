import numpy as np
import torch
from math import sqrt

class Conv2D():
	"""Conv2d layer supporting output_chnl_dim:1, stride=1, ksize=odd"""
	def __init__(self, input_chnl_dim, output_chnl_dim, kernel_size=3, stride=1, initialization='Gauss'):
		super(Conv2D, self).__init__()
		self.input_chnl_dim = input_chnl_dim 
		self.output_chnl_dim = output_chnl_dim
		self.k_size = kernel_size
		self.stride = stride
		self. padding = int(self.k_size/2)
		if initialization == 'Id':
			self.kernel = torch.zeros(output_chnl_dim, input_chnl_dim, kernel_size, kernel_size)
			self.kernel[:, :, int(kernel_size/2), int(kernel_size/2)] = 1
		else:
			self.kernel = torch.randn(output_chnl_dim, input_chnl_dim, kernel_size, kernel_size)
		self.bias = torch.randn(1, output_chnl_dim)
		print(self.kernel)
		self.isTrainable = True
		return

	def forward(self,input):
		batch_sz, _ , ih, iw = input.size()[0], input.size()[1], input.size()[2], input.size()[3]
		padded_inp = torch.zeros(batch_sz, self.input_chnl_dim, ih+2*self.padding, iw+2*self.padding)
		padded_inp[:,:, self.padding:(ih+self.padding), self.padding:(iw+self.padding)] = input
		#print(padded_inp[:,:,1:(ih+self.padding), 1:(iw+self.padding)])
		#print(padded_inp)
		#print(padded_inp[0,:,1:(ih+self.padding), 1:(iw+self.padding)])
		self.output = torch.zeros(batch_sz, self.output_chnl_dim, ih, iw)
		for img in range(batch_sz):
			Xneighb_mat = torch.zeros(ih*iw, self.k_size*self.k_size*self.input_chnl_dim)
			for i1 in range(ih):
				for j1 in range(iw):
					i = i1+self.padding
					j = j1+self.padding
					#print(i-int(self.k_size/2),i+int(self.k_size/2)+1, j-int(self.k_size/2),j+int(self.k_size/2)+1)
					patch = padded_inp[img, :, i-int(self.k_size/2):i+int(self.k_size/2)+1, j-int(self.k_size/2):j+int(self.k_size/2)+1].contiguous()
					#print(patch)
					patch_vec = patch.view(1,-1)
					#	print('#### patch_vec now', patch_vec)
					nidx = (i-self.padding)*iw + j - self.padding
					Xneighb_mat[nidx, :] = patch_vec
			#print('##### Xneighb_mat', Xneighb_mat)
			y1 = Xneighb_mat.mm(self.kernel.view(self.k_size*self.k_size*self.input_chnl_dim, -1)).t() + self.bias
			#print('##### y1', y1)
			y2 = y1.view(1, ih, iw)
			self.output[img] = y2
		return self.output 

	def backward(self, input, gradOutput):
		#gradOutput/ gradInput sampe dimenstions as output/ input
		self.gradInput = gradOutput.mm(self.weight.t())
		self.gradWeight = input.t().mm(gradOutput)
		self.gradBias = gradOutput.sum(dim=0).view(1,self.output_dim)
		return self.gradInput
	def __str__(self):
		str_out = 'Linear Layer with input dimensions (batch firts) {} and output dimensions (batch first) {}'.format(
			self.input_dim, self.output_dim)
		return 	str_out
	def print_param(self):
		print("Weight :")
		print(self.weight)
		print("Bias :")
		print(self.bias)
	def clear_grad(self):
		self.gradInput = 0
		self.gradWeight = 0	
		self.gradBias = 0
		return