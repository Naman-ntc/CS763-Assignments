import matplotlib
#matplotlib.use('Agg')
import torch 
from readData import *
from imports import *
import matplotlib.pyplot as plt


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
model.addLayer(BatchNorm(200))
model.addLayer(ReLU())
model.addLayer(Linear(200, 50))
model.addLayer(BatchNorm(50))
model.addLayer(ReLU())
model.addLayer(Linear(50, 30))
model.addLayer(BatchNorm(30))
model.addLayer(ReLU())
model.addLayer(Linear(30, 20))
model.addLayer(BatchNorm(20))
model.addLayer(ReLU())
model.addLayer(Linear(20, 6))

lossClass = Criterion()

learningRate = 1e-4

# def train(iterations, whenToPrint):
# 	global learningRate
# 	global model
# 	for i in range(iterations):
# 		yPred = model.forward(data)
# 		lossGrad, loss = lossClass.backward(yPred, labels)
# 		if i%whenToPrint == 0:
# 			print(i, loss)
# 		model.clearGradParam()
# 		model.backward(data, lossGrad)
# 		for layer in model.Layers:
# 			if layer.isTrainable:
# 				layer.weight -= learningRate*layer.gradWeight
# 				layer.bias -= learningRate*layer.gradBias

batchSize = 128
plotIndex = 0
losses = []
plotIndices = []

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

def submitPrediction():
	import sys
	yPred = model.forward(test)
	yPred = yPred.max(dim=1)[1]
	N = valData.size()[0]
	sys.stdout = open("output.dat", "w")
	print("id,labels\n")
	for i in range(N):
		print(i,yPred[i])

def makePlot():
	global losses, plotIndices
	plt.plot(plotIndices, losses)
	plt.show()

def saveModel():
	import pickle
	pickle.save(model1,open('output.dat','wb'))

def useOldModel():
	import pickle
	pickle.load(open('model1.pickle',"rb"))



