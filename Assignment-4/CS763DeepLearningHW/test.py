import torch
from Model import *
from readData import *
import numpy as np

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,128,268,200,1)
#print(model.forward(data[0].type(torch.DoubleTensor)))
#print(model.backward(data[0].type(torch.DoubleTensor),torch.randn(1,2)))

trial_data = learningRate = 1e-1

learningRate = 1e-1
