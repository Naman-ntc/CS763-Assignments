from Linear import *
from ReLU import *
from Conv2D import *

L1 = Linear(5, 2)
A1 = ReLU()

x1 = torch.rand(3, 5)

y1 = L1.forward(x1)
z1 = A1.forward(y1)

print('in', x1)
print('out', y1)
print('a_out', z1)

print('################################################')

L2 = Conv2D(1,1,3,initialization='Id')

x2 = torch.rand(1, 1, 10, 10)
y2 = L2.forward(x2)
print(y2-x2)