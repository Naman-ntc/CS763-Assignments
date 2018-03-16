import torch
from Model import *
from readData4 import *
import numpy as np
from Criterion import *

torch.set_printoptions(precision=3)

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,128,153,153,1)

lossClass = Criterion()


def printAcc(batch_size):
	print("\nPrinting Accuracy Now~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	count = 0
	for i in range(batch_size):
		trial_data = data[i].view(1,-1)
		yPred = model.forward(trial_data)
		count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[i]))
		#print(int(yPred.view(1,-1).max(dim=1)[1]),int(labels[i]),yPred.tolist())
	print(count/batch_size)
	print("\nAccuracy block over~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")




learningRate = 1
batch_size = 50
printAcc(batch_size)

for kkk in range(100):
	batch_loss = 0
	for j in range(batch_size):
		i = j
		trial_data = data[i].view(1,-1)
		yPred = model.forward(trial_data)
		#print(yPred.tolist())
		lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
		#print(lossGrad.tolist())
		batch_loss += (loss)
		model.backward(trial_data, lossGrad)
		# print(yPred.tolist())
		
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight -= learningRate*(layer.gradWeight/batch_size)
			layer.bias -= learningRate*(layer.gradBias/batch_size)
	print(kkk,batch_loss/batch_size)		
	model.clearGradParam()	

		
printAcc(batch_size)
