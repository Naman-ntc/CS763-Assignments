import torch
from readData import *

number = 1000

testLabels = torch.from_numpy(npLabels[-number:])
testData = torch.from_numpy(npData[- number:])

testLabels = testLabels.type(torch.FloatTensor)
testData = testData.type(torch.FloatTensor)

testDataSize = testData.size()[0]
testData = testData.contiguous().view(testDataSize, -1)

def test(model):
	global testData, testLabels, number
	pred = model.forward(testData)
	pred = pred.max(dim = 1)[1]
	correct = (pred == testLabels.type(torch.LongTensor)).sum()
	return float(correct)/ float(number)



