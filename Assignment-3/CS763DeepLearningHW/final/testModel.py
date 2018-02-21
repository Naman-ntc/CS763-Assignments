import torch
import sys
sys.path.append('src')
from imports import *
import torchfile as tf
import numpy as np
import os

def getTarget (pathToInput):
	input = tf.load(pathToInput)
	input = torch.from_numpy(input)
	input = input.type(torch.DoubleTensor)
	return input

def getModel(pathToModel):
	os.system("cp " + pathToModel + " visiondata.bin")
	model = torch.load("visiondata.bin")
	os.system("rm visiondata.bin")
	return model

argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]


modelFile = arguments["-modelName"] + "/model.bin"

model = getModel(modelFile)
testData = getTarget(arguments["-data"])

testData = (testData - model.dataMean)/model.dataVariance

ypred = model.forward(testData)
ypred = ypred.max(dim = 1)[1]
file = open("testPredications.bin", 'wb')
torch.save(ypred, file)
file.close()
