import torch 
from readData import *
from imports import *

dataSize = data.size()[0]
data = data.contiguous().view(dataSize, -1)


##Initialize hyperparameters
learningRate = 1e-4
momentum = 0.8

##make the model
model = Model()
model.addLayer(Linear(108*108, 600))
model.addLayer(ReLU())
model.addLayer(Linear(600, 66))
model.addLayer(ReLU())
model.addLayer(Linear(66, 6))
# model.addLayer(ReLU())
# model.addLayer(Linear(50, 6))

lossClass = Criterion()



def train(iterations, whenToPrint):
	global learningRate
	global model
	global momentum
	for i in range(iterations):
		yPred = model.forward(data)
		lossGrad, loss = lossClass.backward(yPred, labels)
		if i%whenToPrint == 0:
			print(i, loss)
		model.clearGradParam()
		model.backward(data, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*((1-momentum)*layer.gradWeight + momentum*layer.momentumWeight)
				layer.bias -= learningRate*((1-momentum)*layer.gradBias + momentum*layer.momentumBias)
