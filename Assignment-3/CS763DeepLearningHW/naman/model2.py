import torch 
from readData import *
from imports import *


dataSize = data.size()[0]
#testDataSize = test.size()[0]
data = data.contiguous().view(dataSize, -1)
#test = test.contiguous().view(testDataSize, -1)
valData = data[5000:]
valLabels = labels[5000:]

data = data[:27000]
labels = labels[:27000]

dataMean = data.mean(dim=0)
data = data - dataMean
valData = valData - dataMean
#test = test - dataMean


valData = valData/data.std(dim=0,keepdim=True)
#test = test/data.std(dim=0,keepdim=True)
data = data/data.std(dim=0,keepdim=True)
##make the model
model = Model()
model.addLayer(Linear(108*108, 200))
model.addLayer(ReLU())
model.addLayer(Linear(200, 6))
# model.addLayer(ReLU())
# model.addLayer(Linear(200, 50))
# model.addLayer(ReLU())
# model.addLayer(Linear(50, 6))

lossClass = Criterion()

learningRate = 1e-4

def train(iterations, whenToPrint):
	global learningRate
	global model
	for i in range(iterations):
		yPred = model.forward(data)
		lossGrad, loss = lossClass.backward(yPred, labels)
		if i%whenToPrint == 0:
			print(i, loss)
		model.clearGradParam()
		model.backward(data, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*layer.gradWeight
				layer.bias -= learningRate*layer.gradBias

def trainAcc():
	yPred = model.forward(data)
	N = data.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum())/N

def valAcc():
	yPred = model.forward(valData)
	N = valData.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == valLabels.type(torch.LongTensor)).sum())/N

def submitPrediction():
	import sys
	yPred = model.forward(test)
	yPred = yPred.max(dim=1)[1]
	N = valData.size()[0]
	sys.stdout = open("output.dat", "w")
	print("id,labels\n")
	for i in range(N):
		print(i,yPred[i])



def saveModel():
	import pickle
	pickle.save(model1,open('output.dat','wb'))

def useOldModel():
	import pickle
	pickle.load(open('model1.pickle',"rb"))