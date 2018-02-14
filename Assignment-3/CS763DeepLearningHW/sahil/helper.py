import torch

def avgMag(model):
	for layer in model.Layers:
		if layer.isTrainable:
			print(layer.weight.abs().mean(), layer.gradWeight.abs().mean() )


def testTrained(model, data, labels):
	print ((model.forward(data).max(dim =1)[1] == labels.type(torch.LongTensor)).sum())