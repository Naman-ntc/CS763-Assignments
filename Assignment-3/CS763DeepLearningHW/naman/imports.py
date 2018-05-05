from Model import *
from Criterion import *
from Linear import *
from ReLU import *
from BatchNorm import *


def saveModel(model, nameOfFile = "model.bin"):
	file = open(nameOfFile, 'wb')
	torch.save(model, file)
	file.close()
	return

def loadModel(nameOfFile = "model.bin"):
	file = open(nameOfFile)
	model = torch.load(file)
	file.close()
	return model

