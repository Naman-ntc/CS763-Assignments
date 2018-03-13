import torch
from Model import *
from readData import *
import numpy as np
from Criterion import *

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,128,333,200,1)

lossClass = Criterion()

learningRate = 1e-1

for i in range(10):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
	if i%1 == 0:
		print(i, loss)
	model.clearGradParam()
	model.backward(trial_data, lossGrad)
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight -= learningRate*layer.gradWeight
			layer.bias -= learningRate*layer.gradBias