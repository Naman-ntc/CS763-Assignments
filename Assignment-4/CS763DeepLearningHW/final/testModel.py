import torch
import sys
sys.path.append('src')
from imports import *
import numpy as np
import os

def getTarget (pathToInput):
	dictionary_file = open("../Small-Data/for_dict.txt")
	dictionary = {}
	dictionary_opened = dictionary_file.readlines()
	for i in range(153):
		dictionary[int(dictionary_opened[i].split()[0])] = i
	test_data = open(pathToInput,"r")
	test = test_data.readlines()
	num_sequences = len(test)
	for i in range(num_sequences):
		test[i] = torch.DoubleTensor([dictionary[int(x)] for x in test[i].split()])
	test_data.close()		
	return test

def getModel(pathToModel):
	os.system("cp " + pathToModel + " visiondata.bin")
	model = torch.load("visiondata.bin")
	os.system("rm visiondata.bin")
	return model


def getPredictions(model,testData):
	totalTest = len(testData)
	predictions = torch.zeros(totalTest)
	for i in range(totalTest):
		test_data = testData[i].view(1,-1)
		yPred = model.forward(test_data)
		predictions[i] = int(yPred.view(1,-1).max(dim=1)[1])
	return predictions	

argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]


modelFile = arguments["-modelName"] + "/model.bin"

model = getModel(modelFile)
testData = getTarget(arguments["-data"])

yPred = getPredictions(model,testData)

file = open("testPredications.bin", 'wb')
torch.save(ypred, file)
file.close()
