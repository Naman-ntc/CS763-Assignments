from Linear import *
from ReLU import *
from Model import *
from Criterion import *

"""
a = Linear(2,3)

temp = a.forward(torch.rand(5,2))
print(a.backward(torch.rand(5,2),torch.rand(5,3)))
"""

"""
b = ReLU()
inputt = torch.randn(5,2)
temp = b.forward(inputt)
print(inputt)
print(temp)
gradout = torch.rand(5,2)
print(gradout)
print(b.backward(gradout))
"""

inp = torch.randn(30, 100)
out = (torch.rand(30)*10).floor()
mymodel = Model()
linear = Linear(100, 25)
mymodel.addLayer(linear)
relu = ReLU()
mymodel.addLayer(relu)
linear2 = Linear(25, 10)
mymodel.addLayer(linear2)
lossClass = Criterion()

print(inp)
print(out)
print(mymodel.forward(inp))

learningRate = 1e-1
for i in range(10000):
	yPred = mymodel.forward(inp)
	lossGrad, loss = lossClass.backward(yPred, out)
	if i%200 == 0:
		print(i, loss)
	mymodel.clearGradParam()
	mymodel.backward(inp, lossGrad)
	for layer in mymodel.Layers:
		if layer.isTrainable:
			layer.weight -= learningRate*layer.gradWeight
			layer.bias -= learningRate*layer.gradBias