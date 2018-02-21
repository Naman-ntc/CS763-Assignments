from imports import *
import sys
import torchfile as tf

def saveTestResults(model):
	import readData
	meann = readData.data.mean(dim = 0)
	stdd = readData.data.std(dim = 0, keepdim = True)
	nptest = tf.load('../../Data/test.bin')
	testdata = torch.from_numpy(nptest)
	testdata = testdata.contiguous().view(testdata.size()[0], -1).type(torch.DoubleTensor)
	testdatanorm = (testdata - meann)/stdd
	ypred = model.forward(testdatanorm)
	ypred = ypred.max(dim = 1)[1]
	N = testdatanorm.size()[0]
	sys.stdout = open("output.dat", "w")
	print("id,label\n")
	for i in range(N):
	    print(str(i)+","+str(ypred[i]))
