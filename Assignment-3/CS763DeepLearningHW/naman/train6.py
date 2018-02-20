import matplotlib
matplotlib.use('Agg')

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



def train(model,lossClass,iterations, whenToPrint, batchSize, learningRate, par_regularization):
	global dataSize, plotIndex, losses, plotIndices
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
		plotIndex += 1


def trainAcc(model):
	yPred = model.forward(data)
	N = data.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == labels.type(torch.LongTensor)).sum())/N

def valAcc(model):
	yPred = model.forward(valData)
	N = valData.size()[0]
	return ((yPred.max(dim=1)[1].type(torch.LongTensor) == valLabels.type(torch.LongTensor)).sum())/N

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




def Try_em_all():
	learningRate = 1e-2
	par_regularization = [1e-2,1e-3,1e-4] 
	batchSize = [128,64,32]
	plotIndex = 0
	losses = []
	plotIndices = []
	
	for reg in par_regularization:
		for bs in batchSize:
			stringg = "Model6"+"-"+str(par_regularization)+"-"+str(batchSize)
			model = Model()	
			model.addLayer(Linear(108*108, 2400))
			model.addLayer(BatchNorm(2400))
			model.addLayer(ReLU())
			model.addLayer(Linear(2400, 600))
			model.addLayer(BatchNorm(600))
			model.addLayer(ReLU())
			model.addLayer(Linear(600, 60))
			model.addLayer(ReLU())

			lossClass = Criterion()

			iterations_count = 128*6000/bs
			lr_decay_iter = iterations_count/10
			reg_zero = 2*iterations_count/10

			for i in range(10):
				train(model,lossClass,lr_decay_iter,10, bs ,learningRate, reg)
				learningRate /= 10
				par_regularization/=10
				print(trainAcc(model))
				print(valAcc(model))	
			torch.save(model,open(stringg,'wb'))