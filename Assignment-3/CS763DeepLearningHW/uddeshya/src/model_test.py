import torch 
from DataLoader import *
from imports import *
lpath = '../Data/labels.bin'
dpath = '../Data/data.bin'
batch_sz = 200
dl = DataLoader(lpath, dpath, batch_sz)
gen_d = dl.get_batch()
print('loaded data')

model = Model()
model.addLayer(Linear(108*108, 200))
model.addLayer(ReLU())
model.addLayer(Linear(200, 6))

lossClass = Criterion()

learningRate = 1e-4

def train(iterations, whenToPrint):
	print('training started...')
	global learningRate
	global model, gen_d, dl
	for i in range(iterations):
		# indices = (torch.randperm(dataSize)[:batchSize]).numpy()
		# currentData = data[indices, :]
		# currentLabels = labels.view(dataSize, 1)[indices, :]

		currentData, currentLabels = next(gen_d)
		
		yPred = model.forward(currentData)
		lossGrad, loss = lossClass.backward(yPred, currentLabels)
		print(lossGrad, loss)	
		
		if i%whenToPrint == 0:
			print(i, loss)
			#losses.append(loss)
			#plotIndices.append(plotIndex)
		model.clearGradParam()
		model.backward(currentData, lossGrad)
		for layer in model.Layers:
			if layer.isTrainable:
				layer.weight -= learningRate*layer.gradWeight
				layer.bias -= learningRate*layer.gradBias
		#plotIndex += 1

train(10, 3)

