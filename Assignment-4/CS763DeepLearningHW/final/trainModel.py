import torch
import sys
sys.path.append('src')
from imports import *
import numpy as np
import os

batchSize = 12  ## Sentence Length
lossClass = Criterion()
learningRate = 1

torch.set_default_tensor_type('torch.DoubleTensor')

def printAcc(start,batch_size):
	count = 0
	for i in range(batch_size):
		trial_data = data[start+i].view(1,-1)
		yPred = model.forward(trial_data)
		count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[start+i]))
	print(count/batch_size)

def saveModel(fileToSave):
	global model
	file = open(fileToSave, 'wb')
	torch.save(model, file)
	file.close()
	return 

def train(epoches,lr):
		