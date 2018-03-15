import torch
from Model import *
from readData4 import *
import numpy as np
from Criterion import *

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,128,153,153,1)

lossClass = Criterion()

learningRate = 1e-3
count = 0

for i in range(50):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[i]))
print(count)

for kkk in range(30):
	batch_loss = 0
	for j in range(50):
		i = j
		trial_data = data[i].view(1,-1)
		yPred = model.forward(trial_data)
		lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
		batch_loss += (loss)
		model.backward(trial_data, lossGrad)
		
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight -= learningRate*layer.gradWeight
			layer.bias -= learningRate*layer.gradBias	
	print(kkk,batch_loss/50,model.Layers[1].gradWeight.min(),model.Layers[1].gradWeight.max())		
	model.clearGradParam()			

count = 0

for i in range(50):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[i]))
print(count)

learningRate/=10

for kkk in range(50):
	batch_loss = 0
	for j in range(50):
		i = j
		trial_data = data[i].view(1,-1)
		yPred = model.forward(trial_data)
		lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
		batch_loss += (loss)
		model.backward(trial_data, lossGrad)

		
		
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight -= learningRate*layer.gradWeight
			layer.bias -= learningRate*layer.gradBias
	print(kkk,batch_loss/50,(model.Layers[1].gradWeight).min(),model.Layers[1].gradWeight.max())			
	model.clearGradParam()			


# model.dispGradParam()


print("\n\nLets check accuracy")

count = 0

for i in range(50):
	trial_data = data[i].view(1,-1)
	yPred = model.forward(trial_data)
	count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[i]))
print(count)