import torch 
from readData import *
from imports import *

dataSize = data.size()[0]
data = data.contiguous().view(dataSize, -1)

##make the model
model = Model()
model.addLayer(Linear(108*108, 6))
# model.addLayer(ReLU())
# model.addLayer(Linear(500, 200))
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
