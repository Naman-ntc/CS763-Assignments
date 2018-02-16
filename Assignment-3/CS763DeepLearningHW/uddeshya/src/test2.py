from Criterion import *

lossClass = Criterion()
inp = torch.FloatTensor([[-1.8168,  0.3020],
 [0.6831,  0.8920],
[-1.3641 ,-1.1230]])
target = torch.LongTensor([0,1,0])
print(lossClass.forward(inp, target))