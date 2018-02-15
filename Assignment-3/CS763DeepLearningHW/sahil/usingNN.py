import torch
from torch.autograd import Variable
from readData import *
import torch.optim as optim


model = torch.nn.Sequential(
          torch.nn.Linear(108*108, 200),
          torch.nn.ReLU(),
          torch.nn.Linear(200, 6),
)

x = Variable(data)
y = Variable(labels, requires_grad=False)

lossClass = CrossEntropyLoss(size_average = True)

optimiser = optim.SGD(model.parameters(), lr = 1e-4, momentum = 0)

def train(num, whenToPrint):
	global optimiser, lossClass, x, y, model
	for i in range(num):
		lossClass
		if i%whenToPrint == 0:
			print(loss)