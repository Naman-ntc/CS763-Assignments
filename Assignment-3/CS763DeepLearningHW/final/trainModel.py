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
reg = 1e-3 
batchSize = 64
dataSize = 0


def train(model,lossClass,iterations, whenToPrint, batchSize, learningRate, par_regularization):
	global dataSize, plotIndex, losses, plotIndices, labels, data
	dataSize = data.size()[0]
	for i in range(iterations):
		indices = (torch.randperm(dataSize)[:batchSize]).numpy()
		currentData = data[indices, :]
		currentLabels = labels.view(dataSize, 1)[indices, :]
		yPred = model.forward(currentData)
		lossGrad, loss = lossClass.backward(yPred, currentLabels)
		if i%whenToPrint == 0:
			reg_loss = model.regularization_loss(par_regularization)
			print("Iter - %d : Training-Loss = %.4f Regularization-Loss = %.4f and Total-loss = %.4f"%(i, loss,reg_loss,loss+reg_loss))
			#losses.append(loss)
			#plotIndices.append(plotIndex)

		model.clearGradParam()
		model.backward(currentData, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*((1-momentum)*layer.gradWeight + momentum*layer.momentumWeight) + par_regularization*layer.weight
				layer.bias -= learningRate*((1-momentum)*layer.gradBias + momentum*layer.momentumBias) + par_regularization*layer.bias
				#layer.weight -= (learningRate*layer.gradWeight + par_regularization*layer.weight)
				#layer.bias -= (learningRate*layer.gradBias + par_regularization*layer.bias)
		if i%(whenToPrint*10) == 0:
			print(trainAcc())	
		plotIndex += 1

def trainAcc():
	global model, data, label
	yPred = model.forward(data)
	N = data.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum())/N

def makeBestModel():
	model = Model()
	model.addLayer(Linear(108*108, 900))
	model.addLayer(ReLU())
	model.addLayer(Linear(900, 6))
	model.addLayer(ReLU())
	return model

def trainModel():
	global model, batchSize, reg, learningRate, lossClass
	iterations_count = 128*8000//batchSize
	lr_decay_iter = iterations_count//8
	reg_zero = 2*iterations_count//10

	for i in range(8):
		train(model,lossClass,lr_decay_iter,10, batchSize ,learningRate, reg)
		learningRate /= 10
		reg/=10
		print(trainAcc())
	return 


def getData(pathToData, pathToLabels):
	global data, labels, dataSize

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
	dataStd = data.std(dim = 0, keepdim = True)
	data = data/dataStd
	return dataMean, dataStd

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
dataMean, dataStd = getData(arguments["-data"], arguments["-target"])
model.saveMeanVariance(dataMean, dataStd)
trainModel()

command = "mkdir " + arguments["-modelName"]
os.system(command)
fileToSave = arguments["-modelName"] + "/model.bin"
saveModel(fileToSave)












