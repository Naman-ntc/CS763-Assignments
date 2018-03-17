import torch
from Model import *
from readData4 import *
import numpy as np
from Criterion import *
import sys

torch.set_printoptions(precision=3)

torch.set_default_tensor_type('torch.DoubleTensor')
model = Model(-1,256,153,153,1)

lossClass = Criterion()


def printAcc(start,batch_size):
	print("\nPrinting Accuracy Now~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
	count = 0
	for i in range(batch_size):
		trial_data = data[start+i].view(1,-1)
		yPred = model.forward(trial_data)
		count += (int(yPred.view(1,-1).max(dim=1)[1])==int(labels[start+i]))
		#print(int(yPred.view(1,-1).max(dim=1)[1]),int(labels[i]),yPred.tolist())
	print(count/batch_size)
	print("\nAccuracy block over~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")


def submitPred(str):
	total_test = 395
	temp_stdout = sys.stdout
	sys.stdout = open(str, "w")
	print("id,label\n")
	for i in range(total_test):
		test_data = test[i].view(1,-1)
		yPred = model.forward(test_data)
		print("%d,%d"%(i,int(yPred.view(1,-1).max(dim=1)[1])))
	sys.stdout = temp_stdout



learningRate = 1
total_train = 1170
total_test = 14

batch_size = 12


def train(epoches,lr):
	for kkk in range(int(epoches)):
		batch_loss = 0
		permed = torch.randperm(total_train)
		counter = 0
		for j in range(int(total_train)):
			i = permed[j]
			trial_data = data[i].view(1,-1)
			yPred = model.forward(trial_data)
			#print(yPred.tolist())
			lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
			#print(lossGrad.tolist())
			batch_loss += (loss)
			model.backward(trial_data, lossGrad)
			# print(yPred.tolist())
			counter+=1
			if counter==batch_size :
				for layer in model.Layers:
					if layer.isTrainable:
						layer.weight -= learningRate*(layer.gradWeight/batch_size)
						layer.bias -= learningRate*(layer.gradBias/batch_size)
				print((total_train*kkk+j)//batch_size,batch_loss/batch_size)		
				model.clearGradParam()
				counter = 0	
				batch_loss = 0
	# batch_loss = 0
	# permed = torch.randperm(total_train)
	# counter = 0
	# for j in range(int(total_train * (epoches%1))):
	# 	i = permed[j]
	# 	trial_data = data[i].view(1,-1)
	# 	yPred = model.forward(trial_data)
	# 	#print(yPred.tolist())
	# 	lossGrad, loss = lossClass.backward(yPred, torch.DoubleTensor([labels[i]]))
	# 	#print(lossGrad.tolist())
	# 	batch_loss += (loss)
	# 	model.backward(trial_data, lossGrad)
	# 	# print(yPred.tolist())
	# 	counter+=1
	# 	if counter==batch_size :
	# 		for layer in model.Layers:
	# 			if layer.isTrainable:
	# 				layer.weight -= learningRate*(layer.gradWeight/batch_size)
	# 				layer.bias -= learningRate*(layer.gradBias/batch_size)
	# 		print(int((total_train*(epoches//1)+j)//batch_size),batch_loss/batch_size)		
	# 		model.clearGradParam()
	# 		counter = 0	
	# 		batch_loss = 0			

printAcc(0,total_train)
printAcc(1170,total_test)


train(10,1)
train(3,1e-1)
printAcc(0,total_train)
printAcc(1170,total_test)
submitPred("output1.dat")		
train(3,1e-2)
train(3,1e-3)
printAcc(0,total_train)
printAcc(1170,total_test)
submitPred("output2.dat")		
train(3,1e-3)
printAcc(0,total_train)
printAcc(1170,total_test)
submitPred("output3.dat")		

"""
86.29
train(10,1)
train(3,1e-1)
printAcc(0,total_train)
printAcc(1100,total_test)
submitPred("output1.dat")		
train(3,1e-2)
train(5,1e-3)
printAcc(0,total_train)
printAcc(1100,total_test)
submitPred("output2.dat")		
"""

"""
85.78
train(10,1)
train(1,1e-1)
printAcc(0,total_train)
printAcc(1100,total_test)
submitPred("output1.dat")		
train(2,1e-2)
train(5,1e-3)
printAcc(0,total_train)
printAcc(1100,total_test)
submitPred("output2.dat")		

"""