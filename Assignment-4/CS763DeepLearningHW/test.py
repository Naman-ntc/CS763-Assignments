import torch
from Model import *
from readData import *
import numpy as np
from Criterion import *

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,128,333,200,1)

lossClass = Criterion()

learningRate = 1e-5

count = 0

for i in range(25):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	count += (yPred.view(1,-1).max(dim=1)[1]==int(labels[i]))
print(count)

for kkk in range(200):
	iterator = np.arange(25)
	np.random.shuffle(iterator)
	batch_loss = 0
	for j in range(25):
		i = iterator[j]
		trial_data = data[i].view(1,-1)
		yPred = model.forward(trial_data)
		lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
		batch_loss += (loss)
		model.clearGradParam()
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*layer.gradWeight
				layer.bias -= learningRate*layer.gradBias
	print(batch_loss)			
	model.backward(trial_data, lossGrad)			

print("\n\nLets check accuracy")

count = 0

for i in range(25):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	count += (yPred.view(1,-1).max(dim=1)[1]==int(labels[i]))
print(count)