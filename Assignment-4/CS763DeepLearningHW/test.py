import torch
from Model import *
from readData import *
import numpy as np


model = Model(-1,128,268,200,1)
temp_data = torch.zeros(data[0].size()[0],268)
temp_data[np.arange(data[0].size()[0]),data[0].numpy()] = 1
model.forward(data[0])