import numpy
import torch

class Conv2D(object):
	"""docstring for Conv2D"""
	def __init__(self,kernel,in_image,in_channels,out_channels,stride,padding='VALID',):
		super(Conv2D, self).__init__()
		self.in_image = in_image
		self.kernel = kernel
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.stride = stride
		if (padding[0]=='VALID'):
			padding[0] = 0
			while((image[0]-kernel[0]+2*padding[0]) % stride[0] != 0):
				padding[0] += 1
		if (padding[1]=='VALID'):
			padding[1] = 0
			while((image[1]-kernel[1]+2*padding[1]) % stride[1] != 0):
				padding[1] += 1		 
		self.padding = padding
		self.out_image = [((image[0]-kernel[0]+2*padding[0])//stride[0]) + 1,((image[1]-kernel[1]+2*padding[1])//stride[1])+1]
		self.filters = torch.randn(out_channels,in_channels,kernel[0],kernel[1]) * torch.sqrt(2/(image[0]*image[1]))
	def forward(self,input):
		"""
		input is an tensor of dimensions N in_channels in_image[0] in_image[1]
		output will be tensor of dimensions N out_channels out_image[0] out_image[1]
		"""
		#input_cols = im2col_efficient(input,)	