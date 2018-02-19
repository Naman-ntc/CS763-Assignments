import torch
import sys
sys.path.append('src')
from imports import *
import torchfile as tf
import numpy as np
import os

batchSize = 128
plotIndex = 0
losses = []
plotIndices = []
lossClass = Criterion()
learningRate = 1e-6
data = None
labels = None


def train(iterations, whenToPrint):
	global learningRate
	global model, dataSize, batchSize, plotIndex, losses, plotIndices
	for i in range(iterations):
		indices = (torch.randperm(dataSize)[:batchSize]).numpy()
		currentData = data[indices, :]
		currentLabels = labels.view(dataSize, 1)[indices, :]
		yPred = model.forward(currentData)
		lossGrad, loss = lossClass.backward(yPred, currentLabels)
		if i%whenToPrint == 0:
			print(i, loss)
			losses.append(loss)
			plotIndices.append(plotIndex)
		model.clearGradParam()
		model.backward(currentData, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*layer.gradWeight
				layer.bias -= learningRate*layer.gradBias
		plotIndex += 1

def trainAcc():
	yPred = model.forward(data)
	N = data.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum())/N

def valAcc():
	yPred = model.forward(valData)
	N = valData.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == valLabels.type(torch.LongTensor)).sum())/N

def makeBestModel():
	model = Model()
	###Add all the layers that you need
	return model

def trainModel():
	global model
	###Actually train the model (using the train function)
	return 


def getData(pathToData, pathToLabels):
	global data, labels

	npLabels = tf.load(pathToLabels)
	npData = tf.load(pathToData)

	totalLabels = torch.from_numpy(npLabels)
	totalData = torch.from_numpy(npData)

	totalLabels = totalLabels.type(torch.DoubleTensor)
	totalData = totalData.contiguous().view(totalData.size()[0], -1).type(torch.DoubleTensor)

	data = totalData[:]
	labels = totalLabels[:]

	dataSize = data.size()[0]
	# data = data.contiguous().view(dataSize, -1)

	dataMean = data.mean(dim=0)
	data = data - dataMean
	data = data/data.std(dim=0,keepdim=True)

	return

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
getData(arguments["-data"], arguments["-target"])

command = "mkdir " + arguments["-modelName"]
os.system(command)
fileToSave = arguments["-modelName"] + "/model.bin"
saveModel(fileToSave)