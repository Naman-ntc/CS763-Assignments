import numpy
import torch
from math import sqrt
from utils import *

momentum = 0.8

class Linear():
    """docstring for Linear"""
    def __init__(self, input_dim,output_dim,initialization='He'):
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.weight = torch.randn(output_dim, input_dim)*sqrt(2/(input_dim))
        self.bias = torch.randn(output_dim, 1)*sqrt(2/(input_dim+output_dim))
        self.isTrainable = True
        self.momentumWeight = torch.zeros(self.weight.size()).type(torch.FloatTensor)
        self.momentumBias = torch.zeros(self.bias.size()).type(torch.FloatTensor)
        return

    def forward(self,input):
        self.output = torch.mm(self.weight, input) + self.bias
        self.output = softmax(self.output)
        return self.output

    def backward(self, input, gradOutput):
        global momentum
        self.gradInput = torch.mm(self.weight.t(), gradOutput)
        self.gradWeight = torch.mm(gradOutput, input.t())
        self.gradBias = gradOutput.sum(dim=1).view(self.output_dim,1)
        self.momentumWeight = momentum*self.momentumWeight + (1- momentum)*self.gradWeight
        self.momentumBias = momentum*self.momentumBias + (1- momentum)*self.gradBias
        return self.gradInput

    def __str__(self):
        string = 'LINEAR'
        return 	string

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

    def weights_norm(self):
        return torch.norm(self.weight) + torch.norm(self.bias)