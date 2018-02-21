import torch 
import torchfile as tf 
from imports import *
import sys


def getInput (pathToInput):
	input = tf.load(pathToInput)
	input = torch.from_numpy(input)
	input = input.type(torch.DoubleTensor)
	return input

def getTarget(pathToTarget):
	targets = tf.load(pathToTarget)
	targets = torch.from_numpy(targets)
	targets = targets - 1 # subtracted 1 because indices are 1-indexed in this case, while 0-indexed in the training set
	targets = targets.type(torch.DoubleTensor)
	return targets

def saveToFile(tensor, pathToFile):
	file = open(pathToFile, 'wb')
	torch.save(tensor, file)
	file.close()
	return 

argumentList = sys.argv[1:]
arguments = {}
for i in range(int(len(argumentList)/2)):
	arguments[argumentList[2*i]] = argumentList[2*i + 1]


input = getInput(arguments["-i"])
targets = getTarget(arguments["-t"])
lossClass = Criterion()
loss = lossClass.forward(input, targets)
print(loss)
grad , _ = lossClass.backward(input, targets)
saveToFile(grad, arguments["-og"])