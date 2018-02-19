from imports import *
import torch
import sys
import torchfile as tf

def makeModel(pathToFile):
	file = open(pathToFile, 'r')
	fileData = ""
	for chunk in file:
		fileData+= chunk
	fileData = fileData.split('\n')
	while(fileData[-1] == ''):
		fileData = fileData[:-1]
	file.close()
	pathToWeight = fileData[-2]
	pathToBias = fileData[-1]
	numberOfLayers = fileData[0]
	fileData = fileData[1:-2]

	model = Model()

	for element in fileData:
		currentData = element.split()
		if currentData[0] == "linear":
			model.addLayer(Linear(int(currentData[1]), int(currentData[2])))
		elif currentData[0] == "relu":
			model.addLayer(ReLU())
		# elif currentData[0] == "batchnorm":
		# 	model.addLayer(BatchNorm(int(currentData[1])))

	weightData = tf.load(pathToWeight)
	biasData = tf.load(pathToBias)
	index = 0
	for layer in model.Layers:
		if layer.isTrainable:
			layer.weight = torch.from_numpy(weightData[index]).t()
			layer.bias = torch.from_numpy(biasData[index])
			index += 1

	return model


def readInput(pathToInput):
	input = tf.load(pathToInput)
	input = torch.from_numpy(input)
	inputSize = input.size()
	input = input.view(inputSize[0], -1)
	return input

def readInputGrad(pathToInputGrad):
	inputGrad = tf.load(pathToInputGrad)
	inputGrad = torch.from_numpy(inputGrad)
	return inputGrad

def saveToFile(tensor, pathToFile):
	file = open(pathToFile, 'wb')
	torch.save(tensor, file)
	file.close()
	return 
#########################################################################################


argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]

#print('')
model = makeModel(arguments["-config"])
#print('')
input = readInput(arguments["-i"])
inputGrad = readInputGrad(arguments["-ig"])
output = model.forward(input)
saveToFile(output, arguments["-o"])
model.clearGradParam()
model.backward(input, inputGrad)
weightGrads = []
biasGrads = []
for layer in model.Layers:
	if layer.isTrainable:
		weightGrads.append(layer.gradWeight.t())
		biasGrads.append(layer.gradBias)
saveToFile(weightGrads, arguments["-ow"])
saveToFile(biasGrads, arguments["-ob"])
finalGrad = model.Layers[0].gradInput
saveToFile(finalGrad, arguments["-og"])
#print('')








