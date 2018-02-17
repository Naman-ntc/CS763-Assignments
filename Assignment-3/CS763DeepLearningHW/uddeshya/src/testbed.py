from Linear import *
from ReLU import *
from Conv2D import *
from BatchNorm import *

L1 = Linear(5, 2)
A1 = ReLU()

x1 = torch.rand(3, 5)

y1 = L1.forward(x1)
z1 = A1.forward(y1)

print('in', x1)
print('out', y1)
print('a_out', z1)

print('################################################')

# L2 = Conv2D(1,1,kernel_size=5,initialization='Id')
# L3 = Conv2D(1,1,kernel_size=5,initialization='Gauss')
# x2 = torch.rand(1, 1, 10, 10)
# y2 = L2.forward(x2)
# y3 = L3.forward(x2)
# print(y2-x2)
# print(y3)

L2 = BatchNorm(eps=1e-6)
x2 = torch.rand(10,5)*100
y2 = L2.forward(x2)
print(x2)
print(y2)
z2 = L2.backward(x2, y2)
print(z2)