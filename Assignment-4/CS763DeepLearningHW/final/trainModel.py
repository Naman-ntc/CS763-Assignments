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

def getData(pathToData, pathToLabels):
	global data, labels
	train_data = open(pathToData,"r")
	train_labels = open(pathToLabels,"r")
	dictionary_file = open("for_dict.txt","r")

	dictionary = {}

	dictionary_opened = dictionary_file.readlines()

	for i in range(153):
		dictionary[int(dictionary_opened[i].split()[0])] = i

	data = train_data.readlines()
	labels = train_labels.readlines()
	num_sequences = len(data)
	
	for i in range(num_sequences):
		data[i] = torch.DoubleTensor([dictionary[int(x)] for x in data[i].split()])
		labels[i] = int(labels[i].split()[0])
			
	labels = torch.DoubleTensor(labels)

	train_data.close()
	train_labels.close()
	
		
def train(epoches,lr):
	global model
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


def makeBestModel():
	model = Model(-1,256,153,153,1)
	return model


def trainModel():
	train(10,1)
	train(3,1e-1)
	printAcc(0,total_train)
	printAcc(1100,total_test)
	train(3,1e-2)
	train(5,1e-3)
	printAcc(0,total_train)
	printAcc(1100,total_test)


def saveModel(fileToSave):
	global model
	file = open(fileToSave, 'wb')
	torch.save(model, file)
	file.close()
	return 
	
argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]
model = makeBestModel()

trainModel()

command = "mkdir " + arguments["-modelName"]
os.system(command)
fileToSave = arguments["-modelName"] + "/model.bin"
saveModel(fileToSave)


